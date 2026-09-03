#%% load library
import numpy as np
import pickle
import copy
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.models.glm as glm
from cedalion.sigproc import frequency
from statsmodels.gam.smooth_basis import BSplines
from scipy.signal import butter, sosfiltfilt, filtfilt, windows

#%% find subjects with fNIRS and enough EEG epochs
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')
_MIN_EPOCHS = 500

_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf'))
    if re.search(r'sub-(\d+)', f)
}

_gradcpt_fifs = sorted(glob.glob(os.path.join(_eeg_deriv, 'sub-*', '*task-gradCPT*preproc_eeg.fif')))
_subj_to_fifs = {}
for _f in _gradcpt_fifs:
    _m = re.search(r'sub-(\d+)', _f)
    if _m:
        _subj_to_fifs.setdefault(_m.group(1), []).append(_f)

_subj_epoch_counts = {}
for _sid in sorted(_subj_to_fifs):
    _total = 0
    for _fif in sorted(_subj_to_fifs[_sid]):
        _events_tsv = _fif.replace('_preproc_eeg.fif', '_events.tsv')
        if not os.path.exists(_events_tsv):
            continue
        _ev_df = pd.read_csv(_events_tsv, sep='\t')
        _onsets = _ev_df['onset'].values
        if len(_onsets) == 0:
            continue
        _raw = mne.io.read_raw_fif(_fif, preload=True, verbose=False)
        _events_arr = np.column_stack([
            (_onsets * _raw.info['sfreq']).astype(int),
            np.zeros(len(_onsets), dtype=int),
            np.ones(len(_onsets), dtype=int),
        ])
        _valid = (_events_arr[:, 0] >= 0) & (_events_arr[:, 0] < _raw.n_times)
        _events_arr = _events_arr[_valid]
        if len(_events_arr) == 0:
            continue
        _epochs = mne.Epochs(_raw, _events_arr, event_id=1,
                             tmin=-0.2, tmax=1.0,
                             baseline=None, preload=True, verbose=False)
        _epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)
        _total += len(_epochs)
    _subj_epoch_counts[_sid] = _total

_enough_sids = {sid for sid, n in _subj_epoch_counts.items() if n >= _MIN_EPOCHS}
subj_id_array = [int(s) for s in sorted(_fnirs_subjects & _enough_sids)]

# check if any of subject in subj_id_array is in the excluded_subj
subj_id_array = [x for x in subj_id_array if f'sub-{x}' not in excluded_subj]

#%% select model type
eeg_reg_type = 'trial_EEG_cz'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = False # If True, highpass fNIRS by 1/len_delay (Hz)
is_norm = False # If True, z-score regressors.
is_plot = False # If True, generate visualization plots
select_chromo = 'HbO'
select_parcel = 'DefaultA_PFCd_1_LH'
USE_GSR = True
is_GSR_then_Others = False # If True, use OLS to regress out GSR first, then use the residuals to fit other regressors.
cfg_GLM['do_GSR'] = USE_GSR
len_delay = 15 # Delay time in HRF (sec), also used as the per-trial epoch length
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)

# which trial types to include as separate EEG delay-FIR regressor sets, one block per
# entry (mirrors run_model_cont_EEG_city_mnt.py's trial_type_prefixes). Each name must be
# one of the stim-locked keys eeg_epoch_subj_level/get_valid_event_idx produce:
# 'city_correct', 'city_incorrect', 'mnt_correct', 'mnt_incorrect'.
# Set to 'all' to fit a single unified regressor set over every trial (no per-type split),
# matching run_model_cont_EEG_fNIRS.py.
select_trial_types = 'all'
# select_trial_types = ['mnt_correct', 'mnt_incorrect']

_ALL_TRIAL_TYPES = ['city_correct', 'city_incorrect', 'mnt_correct', 'mnt_incorrect']
if select_trial_types == 'all':
    trial_type_list = _ALL_TRIAL_TYPES
else:
    trial_type_list = select_trial_types

# map epoch-dict event key -> (raw trial_type, response_code condition) used to rebuild
# the per-trial onset table from each run's events.tsv
_trial_type_to_condition = {
    'city_correct':   lambda df: (df['trial_type'] == 'city') & (df['response_code'] > 0),
    'city_incorrect': lambda df: (df['trial_type'] == 'city') & (df['response_code'] < 0),
    'mnt_correct':    lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] == 0),
    'mnt_incorrect':  lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),
}

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
    betas_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% RUN PREPROCESSING
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

    print('LOADING PREPROCESSED CHANNEL DATA')
    with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        results = pickle.load(f)

    all_chs_pruned = results['chs_pruned']
    all_stims = results['stims']
    geo3d = results['geo3d']

    print('LOADING IMAGE SPACE RESULTS')
    folder = os.path.join(der_dir, subject)
    filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'

    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)

    all_runs = image_results['parcel_ts']
    vv = image_results['vertex_mse']

    n_runs = len(vv)
    vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
    vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

    #%% optional highpass and parcel/chromo selection (channel-space stims kept as-is;
    # per-trial-type onset tables are rebuilt later directly from each run's events.tsv)
    all_runs_tmp = []
    for run in all_runs:
        if F_MIN > 0:
            run.time.attrs['units'] = units.s
            run_filt = frequency.freq_filter(run, F_MIN * units.Hz, F_MAX * units.Hz)
            all_runs_tmp.append(run_filt)
        else:
            all_runs_tmp.append(run)
    all_runs = all_runs_tmp

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel=run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp

    if select_chromo is not None:
        all_runs = [x.sel(chromo=[select_chromo]) for x in all_runs]

    ori_all_runs = all_runs.copy()

    #%% get continuous EEG
    eeg_der_dir = os.path.join(project_path, "derivatives", "eeg")
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    # check if Cz exists
    if 'cz' in eeg_reg_type:
        cz_removed = any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]]
                        for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3'])
        if cz_removed:
            print(f"sub-{subj_id}: Cz was removed in at least one run, skipping subject.")
            continue

    # epoch EEG per trial type/run; used only to determine which trials pass artifact
    # rejection (drop_log), via model.get_valid_event_idx below
    single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = \
        utils.eeg_epoch_subj_level(subject, single_subj_EEG_dict, preproc_params)

    valid_idx_by_type = {tt: model.get_valid_event_idx(tt, single_subj_epoch_dict) for tt in trial_type_list}

    # match each fNIRS run in all_runs to its EEG run (gradcpt1/2/3) via first stim onset in events.tsv
    eeg_ev_files = {
        run_key: os.path.join(eeg_der_dir, subject, f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv")
        for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']
    }
    eeg_ev_dfs = {run_key: pd.read_csv(f, sep='\t') for run_key, f in eeg_ev_files.items()}

    nirs_ev_files = sorted(glob.glob(os.path.join(project_path, subject, 'nirs', f"{subject}_task-gradCPT_run-*_events.tsv")))
    nirs_ev_dfs = {f: pd.read_csv(f, sep='\t') for f in nirs_ev_files}

    matched_nirs_file = dict()
    matched_t_offset = dict()  # run_key -> t_offset (nirs_time = eeg_time + t_offset)
    for run_key, ev_df in eeg_ev_dfs.items():
        run_num = f"{run_key[-1]:0>2}"  # e.g. 'gradcpt1' -> '01'
        for nirs_file, nirs_df in nirs_ev_dfs.items():
            if f"run-{run_num}" in os.path.basename(nirs_file):
                matched_nirs_file[run_key] = nirs_file
                matched_t_offset[run_key] = nirs_df['onset'].values[0] - ev_df['onset'].values[0]
                break

    run_key_to_run_idx = dict()
    for run_key, nirs_file in matched_nirs_file.items():
        nirs_onset0 = nirs_ev_dfs[nirs_file]['onset'].values[0]
        for r_i, stim in enumerate(all_stims):
            if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onset0, atol=0.01):
                run_key_to_run_idx[run_key] = r_i
                break

    #%% fNIRS sampling rate (all_runs' time coordinate is in seconds)
    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()
    # matches get_cont_EEG_regressor's own n_delay = round(delay*sfreq), so that a
    # 2*n_delay-sample EEG epoch yields exactly len_epoch_sample output rows (see below)
    n_delay = np.round(len_delay * fnirs_sfreq).astype(int)
    len_epoch_sample = n_delay

    #%% for each run: crop EEG/fNIRS to the shared event window, resample EEG to fNIRS
    # rate, and build a local (run-relative, zero-based) onset table per trial type,
    # exactly like run_model_cont_EEG_city_mnt.py -- but here the truncated/resampled
    # signals are only used as the source to slice per-trial epochs from below, not fed
    # continuously into the GLM.
    l_cutoff = np.round(1 / len_delay, decimals=2)

    eeg_list = []
    fnirs_raw_list = []
    run_onsets = dict()  # run_key -> {trial_type: array of local onset times (sec, run-relative)}
    eeg_to_fnirs_offset_by_run = dict()  # run_key -> offset s.t. fnirs_local_time = eeg_local_time + offset
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_idx = run_key_to_run_idx[run_key]
        fnirs_run_raw = ori_all_runs[run_idx]

        if is_hp_fNIRS:
            fnirs_units = fnirs_run_raw.pint.units
            sos = butter(4, l_cutoff, btype='highpass', fs=fnirs_sfreq, output='sos')
            fnirs_run_raw = xr.apply_ufunc(
                sosfiltfilt, sos, fnirs_run_raw.pint.dequantify(),
                input_core_dims=[[], ['time']],
                output_core_dims=[['time']],
                exclude_dims={'time'},
            ).transpose(*fnirs_run_raw.dims).pint.quantify(fnirs_units)
            fnirs_run_raw = fnirs_run_raw.assign_coords({'time': ori_all_runs[run_idx].time})

        # EEG <-> fNIRS clock offset for this run (nirs_time = eeg_time + t_offset)
        eeg_ev_df = eeg_ev_dfs[run_key]
        nirs_ev_df = nirs_ev_dfs[matched_nirs_file[run_key]]
        t_offset = matched_t_offset[run_key]

        # window from the first event onset to the last event's onset + len_delay, in fNIRS time.
        # EEG needs an extra len_delay seconds of history *before* nirs_t_start, since each
        # trial's EEG epoch reaches back to (onset - len_delay) to give the delay-FIR
        # regressor a full n_delay-sample history at every output timepoint (see below).
        nirs_t_start = nirs_ev_df['onset'].values[0]
        nirs_t_stop = (nirs_ev_df['onset']).values[-1] + len_delay  # second
        eeg_t_start = nirs_t_start - t_offset - len_delay
        eeg_t_stop = nirs_t_stop - t_offset

        EEG = single_subj_EEG_dict[run_key]
        # EEG crop starts len_delay seconds before nirs_t_start (see above), clamped to the
        # recording's own bounds
        EEG_raw = EEG.copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))
        _eeg_crop_tmin = max(eeg_t_start, 0)  # actual local-time zero point of EEG_raw, in eeg-run time

        # truncate fNIRS to the same shared event window (fNIRS local time zero = nirs_t_start)
        fnirs_run = fnirs_run_raw.sel(time=slice(max(nirs_t_start, fnirs_run_raw.time.values[0]),
                                                   min(nirs_t_stop, fnirs_run_raw.time.values[-1])))

        # downsample EEG to fNIRS sampling rate (kept at its own, longer length -- no
        # sample-count matching against fNIRS here, since EEG_resample intentionally spans
        # len_delay seconds more than fnirs_run, at the front)
        EEG_resample = EEG_raw.copy()
        EEG_resample.resample(fnirs_sfreq, npad='auto')

        # reset fNIRS time base to 0 (run-local time, relative to nirs_t_start)
        fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
        # offset (sec) between EEG_resample's local time (0 = _eeg_crop_tmin, in eeg-run time)
        # and fNIRS local time (0 = nirs_t_start, in nirs-run time): fnirs_local = eeg_local + eeg_to_fnirs_offset
        eeg_to_fnirs_offset = (_eeg_crop_tmin + t_offset) - nirs_t_start

        # build per-trial-type local onset tables (run-relative, zero-based, in fNIRS time),
        # keeping only trials that were 'preserved' by EEG epoch QC (get_valid_event_idx).
        # get_valid_event_idx's 'preserved' indices index into the *boundary-filtered*
        # per-type event list that eeg_epoch_subj_level/epoch_by_select_event build
        # internally (events too close to the EEG run's start/end for the ERP baseline/
        # tmax window are dropped before drop_log is computed) -- so the same boundary
        # filter must be replicated here on trials_this_type before preserved_idx can be
        # used to index into it.
        eeg_run_key = f"run{int(run_key[-1]):02d}"
        _sfreq_full = EEG.info['sfreq']
        _n_samples_full = len(EEG.times)
        run_onsets[run_key] = dict()
        for tt in trial_type_list:
            trials_this_type = eeg_ev_df[_trial_type_to_condition[tt](eeg_ev_df)].reset_index(drop=True)
            if len(trials_this_type) > 0:
                _event_duration = 1.6 if tt.endswith('response') else 1.8
                _baseline_length = -0.8 if tt.endswith('response') else -0.2
                _tmax = _event_duration + _baseline_length
                _samples_before = int(np.abs(_baseline_length) * _sfreq_full)
                _samples_after = int(_tmax * _sfreq_full)
                _onset_samples = (trials_this_type['onset'].values * _sfreq_full).astype(int)
                _boundary_mask = (_onset_samples >= _samples_before) & (_onset_samples <= _n_samples_full - _samples_after)
                trials_this_type = trials_this_type[_boundary_mask].reset_index(drop=True)
            preserved_idx = valid_idx_by_type[tt].get(eeg_run_key, {}).get('preserved', [])
            if len(preserved_idx) == 0 or len(trials_this_type) == 0:
                run_onsets[run_key][tt] = np.array([])
                continue
            preserved_idx = np.asarray(preserved_idx)
            preserved_idx = preserved_idx[preserved_idx < len(trials_this_type)]
            onsets = trials_this_type['onset'].values[preserved_idx]
            # shift from eeg-run time -> nirs-run time -> local (run-relative, zero-based) time
            local_onsets = onsets + t_offset - nirs_t_start
            run_onsets[run_key][tt] = local_onsets

        eeg_list.append(EEG_resample)
        fnirs_raw_list.append(fnirs_run)
        eeg_to_fnirs_offset_by_run[run_key] = eeg_to_fnirs_offset

    #%% Regress GSR out before fitting EEG (on the full continuous run, before epoching)
    if USE_GSR and is_GSR_then_Others:
        gs_regressors = model.get_global_mean_regressor(fnirs_raw_list)
        resid_runs = []
        for run, gs_dm in zip(fnirs_raw_list, gs_regressors):
            gsr_results = glm.fit(run, gs_dm, noise_model=cfg_GLM['noise_model'])
            gsr_pred = glm.predict(run, gsr_results.sm.params, gs_dm)
            gsr_pred = gsr_pred.pint.dequantify().pint.quantify('molar')
            resid = run - gsr_pred
            resid = resid.transpose(*run.dims)
            resid_runs.append(resid)
        fnirs_raw_list = resid_runs

    #%% Extract continuous EEG values per run (for slicing into per-trial epochs below)
    eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]

    #%% Epoch EEG and fNIRS at each preserved trial's onset. fNIRS epoch: len_delay seconds
    # starting at trial onset (Y to be explained). EEG epoch: 2*len_delay seconds, starting
    # len_delay seconds *before* trial onset (i.e. [onset-len_delay, onset+len_delay)), so
    # that get_cont_EEG_regressor -- which drops the first n_delay output samples to give
    # every regressor row a full real-EEG delay history -- yields exactly len_epoch_sample
    # output rows aligned with the fNIRS epoch. Trials whose window would run past either
    # signal's bounds are dropped.
    # eeg_epochs_by_type[tt] / fnirs_epochs_by_type[tt]: list of 1D EEG arrays / fNIRS
    # DataArrays, one per selected trial (across all runs, for that trial type).
    eeg_epochs_by_type = {tt: [] for tt in trial_type_list}
    fnirs_epochs_by_type = {tt: [] for tt in trial_type_list}
    for run_i, run_key in enumerate(['gradcpt1', 'gradcpt2', 'gradcpt3']):
        eeg_sample_times = eeg_list[run_i].times
        eeg_sig = eeg_reg_value_list[run_i]
        fnirs_run = fnirs_raw_list[run_i]
        n_fnirs_samples = len(fnirs_run.time)
        n_eeg_samples = len(eeg_sample_times)
        offset = eeg_to_fnirs_offset_by_run[run_key]  # fnirs_local = eeg_local + offset

        for tt in trial_type_list:
            for onset in run_onsets[run_key][tt]:
                if onset < 0:
                    continue
                fnirs_i = np.searchsorted(fnirs_run.time.values, onset)
                if fnirs_i + len_epoch_sample > n_fnirs_samples:
                    continue
                eeg_local_onset = (onset - len_delay) - offset
                eeg_i = np.searchsorted(eeg_sample_times, eeg_local_onset)
                if eeg_i < 0 or eeg_i + 2 * n_delay > n_eeg_samples:
                    continue
                eeg_epochs_by_type[tt].append(eeg_sig[eeg_i:eeg_i + 2 * n_delay])
                fnirs_epochs_by_type[tt].append(
                    fnirs_run.isel(time=slice(fnirs_i, fnirs_i + len_epoch_sample))
                )

    for tt in trial_type_list:
        n_trials = len(eeg_epochs_by_type[tt])
        print(f"{subject} {tt}: {n_trials} trials selected for design matrix.")
        if n_trials == 0:
            print(f"{subject} {tt}: no trials found, regressor will be dropped for this trial type.")

    #%% Build per-trial-type EEG delay-FIR regressors (one design-matrix "run" per
    # selected trial), then concatenate the trials, matching get_cont_EEG_regressor's
    # per-run FIR construction but with each selected trial acting as its own "run"
    per_type_regressors = dict()
    per_type_fnirs = dict()
    for tt in trial_type_list:
        if len(eeg_epochs_by_type[tt]) == 0:
            per_type_regressors[tt] = []
            per_type_fnirs[tt] = []
            continue
        per_type_regressors[tt] = model.get_cont_EEG_regressor(
            eeg_epochs_by_type[tt], fnirs_sfreq, delay=len_delay,
            name_prefix=f'{tt}_', z_score=is_norm)
        per_type_fnirs[tt] = fnirs_epochs_by_type[tt]

    #%% flatten per-trial-type (trial-level) design matrices + fNIRS epochs into one
    # flat list of "runs" (one per selected trial, across all included trial types), so
    # they can go through model.concatenate_runs_dms exactly like a normal run list.
    # Trial types not present for a given trial get all-zero regressor columns (handled
    # by fillna(0) after concatenation), keeping the & combine below well-defined even
    # when trial counts differ across types.
    if len(trial_type_list) == 1:
        # single trial type selected (or effectively 'all' collapsed to one block):
        # no need to align/zero-pad across types
        tt = trial_type_list[0]
        flat_fnirs = per_type_fnirs[tt]
        flat_dms = per_type_regressors[tt]
    else:
        flat_fnirs = []
        flat_dms = []
        for tt in trial_type_list:
            for fnirs_epoch, eeg_dm in zip(per_type_fnirs[tt], per_type_regressors[tt]):
                flat_fnirs.append(fnirs_epoch)
                flat_dms.append(eeg_dm)

    if len(flat_fnirs) == 0:
        print(f"{subject}: no trials selected for any trial type, skipping subject.")
        continue

    # concatenate all selected trial epochs (as if each were its own run) and their dms
    Y_all, dm_all, runs_updated = model.concatenate_runs_dms(flat_fnirs, flat_dms)

    #%% Low-rank representation of Delay using BSpline (fit independently per trial type,
    # so each trial type's FIR delay curve is smoothed on its own low-rank subspace rather
    # than sharing a single basis across all trial types' concatenated delay indices)
    all_regressor_names = dm_all.common.regressor.values
    basis_blocks = []
    for tt in trial_type_list:
        type_regressor_names = [name for name in all_regressor_names
                                 if name.startswith(f'{tt}_delay')]
        if len(type_regressor_names) == 0:
            continue
        n_type_regressor = len(type_regressor_names)
        delay_idx = np.arange(n_type_regressor)
        bspline_basis = BSplines(delay_idx, df=[n_bspline_basis], degree=[bspline_degree],
                                  include_intercept=True).basis  # (n_type_regressor, n_bspline_basis)
        basis_blocks.append(xr.DataArray(
            bspline_basis,
            dims=("regressor", "component"),
            coords={"regressor": type_regressor_names,
                    "component": [f"{tt}_bspline{i}" for i in range(n_bspline_basis)]},
        ))
    basis_da = xr.concat(basis_blocks, dim="component").fillna(0)
    basis_da = basis_da.sel(regressor=all_regressor_names)

    # project the full-rank delay design matrix onto the low-rank spline basis
    dm_all.common = xr.dot(dm_all.common, basis_da, dims="regressor").rename({"component": "regressor"})

    #%% Combine drift and GSR regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = model.get_drift_regressors(runs_updated, cfg_GLM)
        dm_all &= model.reduce(model.operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = model.get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dm_all &= model.reduce(model.operator.and_, drift_regressors)

    if cfg_GLM['do_GSR'] and not is_GSR_then_Others:
        gsr = model.get_global_mean_regressor(runs_updated)
        dm_all &= model.reduce(model.operator.and_, gsr)

    dm_all.common = dm_all.common.fillna(0)

    # normalized regressors if required
    if is_norm:
        dm_all.common = (dm_all.common - dm_all.common.mean('time')) / dm_all.common.std('time')

    #%% select HbO to fasten training process
    dm_all.common = dm_all.common.sel(chromo=[select_chromo])

    #%% get GLM fitting results for each subject
    print(f"Start trial_EEG GLM fitting ({subject})")
    glm_results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    # extract HRF (delay-regressor betas) per parcel and per trial type, then expand the
    # low-rank bspline coefficients back to full per-delay resolution via the same basis
    betas_all = glm_results.sm.params.copy()
    betas_eeg_per_type = dict()
    for tt in trial_type_list:
        eeg_reg = [p for p in betas_all.regressor.values if p.startswith(f'{tt}_') and 'bspline' in p]
        if len(eeg_reg) == 0:
            continue
        betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
        type_basis_da = basis_da.sel(component=eeg_reg)
        betas_eeg = xr.dot(betas_bspline, type_basis_da, dims="component")
        type_regressor_names = [name for name in all_regressor_names
                                 if name.startswith(f'{tt}_delay')]
        betas_eeg = betas_eeg.sel(regressor=type_regressor_names)
        betas_eeg_per_type[tt] = betas_eeg

    #%% visual check fit results and HRF
    if is_plot:
        plt_type = trial_type_list[0]
        parcel_names = [p for p in betas_eeg_per_type[plt_type].parcel.values if not p.startswith('Background+FreeSurfer')]
        select_network = select_parcel.split('_')[0]
        net_parcels = [p for p in parcel_names if p.split('_')[0] == select_network]

        vis_betas = glm_results.sm.params.copy()
        eeg_reg = [p for p in vis_betas.regressor.values if 'bspline' in p]
        vis_betas = vis_betas.sel(regressor=eeg_reg)
        y_hat = xr.dot(dm_all.common, vis_betas, dims='regressor')

        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=False)
        y_true = Y_all.sel(parcel=select_parcel).values.flatten()
        y_hat_vals = y_hat.sel(parcel=select_parcel).values.flatten()

        axs[0].plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=2)
        axs[0].plot(y_hat.time.values, y_hat_vals, 'b', label='y_hat (EEG)', alpha=0.5)
        axs[0].set_ylabel('HbO concentration')
        axs[0].set_title(f'Parcel activities estimation ({select_parcel})')
        axs[0].legend()
        axs[0].grid()
        t_betas = np.arange(0, len_delay, 1 / fnirs_sfreq)
        for tt in trial_type_list:
            if tt not in betas_eeg_per_type:
                continue
            betas_eeg = betas_eeg_per_type[tt]
            axs[1].plot(t_betas, betas_eeg.sel(parcel=select_parcel).values.flatten(), label=f'HRF {tt} ({select_parcel})')
            axs[1].plot(t_betas, betas_eeg.sel(parcel=net_parcels).mean('parcel').values.flatten(), '--', label=f'HRF {tt} ({select_network})')
        axs[1].set_title('HRF estimation using Cz EEG, trial-epoch based')
        axs[1].legend()
        axs[1].grid()

    #%% save betas for later visualization
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg_per_type'] = betas_eeg_per_type
        betas_dict['basis_da'] = basis_da
        betas_dict['n_trials_per_type'] = {tt: len(eeg_epochs_by_type[tt]) for tt in trial_type_list}
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)
