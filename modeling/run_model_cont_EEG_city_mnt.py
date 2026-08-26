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
# eeg_reg_type = 'cont_EEG_allBandPower-bandpass'
eeg_reg_type = 'cont_EEG_city_mnt_cz'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = False # If True, highpass fNIRS by 1/len_delay (Hz)
is_norm = False # If True, z-score regressors.
is_plot = False # If True, generate visualization plots
select_chromo='HbO'
# select_parcel='DorsAttnA_ParOcc_1_RH'
select_parcel='DefaultA_PFCd_1_LH'
USE_GSR=True
is_GSR_then_Others = False # If True, use OLS to regress out GSR first, then use the residuals to fit other regressors.
cfg_GLM['do_GSR']=USE_GSR
len_delay = 15 # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)

# trial types being compared -- each gets its own set of EEG delay-FIR regressors,
# active (non-zero) only during that trial type's window (onset -> next trial onset)
trial_type_prefixes = ['city', 'mnt']

# sliding-window bandpower parameters (used when eeg_reg_type ends with 'power')
power_bands = {
    # 'delta': (1, 4),
    # 'theta': (4, 8),
    'alpha': (8, 13),
    # 'beta': (13, 30),
}
power_win_len = 1  # sliding window length (sec); if None, defaults to 1/fnirs_sfreq at runtime
power_method = 'bandpass'  # 'bandpass' (filter + square) or 'multitaper' (DPSS PSD)
sum_method = np.mean # method for summarizing power across channels
power_bandwidth = None  # multitaper frequency smoothing (Hz); None uses MNE's default

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
    with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        results = pickle.load(f)

    # all_runs = results['runs']
    all_chs_pruned = results['chs_pruned']
    all_stims = results['stims']
    geo3d = results['geo3d']

    print('LOADING IMAGE SPACE RESULTS')
    folder =  os.path.join(der_dir, subject)
    filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'

    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)

    all_runs = image_results['parcel_ts']
    vv = image_results['vertex_mse']

    n_runs = len(vv)
    vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
    vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

    #%% label city_correct / mnt_correct trials per run (all other trials are dropped from
    # the trial-type windowing below; they contribute no EEG regressor window)
    stims_pruned_list = []

    all_runs_tmp = []
    for stim, run in zip(all_stims, all_runs):
        city_correct_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] > 0)].copy()
        city_correct_trials['trial_type'] = 'city-correct'

        mnt_correct_trials = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
        mnt_correct_trials['trial_type'] = 'mnt-correct'

        city_mnt_trials = pd.concat([city_correct_trials, mnt_correct_trials], ignore_index=False).sort_index()

        if F_MIN > 0:
            # TODO HP the timeseries data
            run.time.attrs['units'] = units.s
            run_filt = frequency.freq_filter(run,
                                        F_MIN*units.Hz,
                                        F_MAX*units.Hz)
            all_runs_tmp.append(run_filt)
        else:
            all_runs_tmp.append(run)

        stims_pruned_list.append(city_mnt_trials)

    # run_ts_list = [image_results['parcel_ts_weights']]
    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel = run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp.copy()

    # select only one parcel and one chromo
    if select_chromo is not None:
        all_runs = [x.sel(chromo=[select_chromo]) for x in all_runs]

    ori_all_runs = all_runs.copy()

    #%% get continous EEG
    eeg_der_dir = os.path.join(project_path, "derivatives", "eeg")
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    # check if Cz exists
    if 'cz' in eeg_reg_type:
        cz_removed = any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]]
                        for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3'])
        if cz_removed:
            print(f"sub-{subj_id}: Cz was removed in at least one run, skipping subject.")
            continue

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

    # check if run_key repeat
    # print(run_key_to_run_idx.items())

    #%% bandpass and resample EEG
    # fNIRS sampling rate (all_runs' time coordinate is in seconds)
    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()
    # get highpass filter frequency
    l_cutoff = np.round(1/len_delay,decimals=2)

    eeg_list = []
    eeg_raw_list = []
    all_runs_truncated = []
    fnirs_raw_list = []
    run_time_windows = dict()  # run_key -> (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)
    stims_pruned_truncated = dict()  # run_key -> city/mnt-correct trials df, onsets shifted to the truncated run's local time
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_idx = run_key_to_run_idx[run_key]
        fnirs_run = ori_all_runs[run_idx].copy()
        fnirs_run_raw = ori_all_runs[run_idx]

        # highpass fNIRS to remove drift
        if is_hp_fNIRS:
            fnirs_units = fnirs_run.pint.units
            sos = butter(4, l_cutoff, btype='highpass', fs=fnirs_sfreq, output='sos')
            fnirs_run = xr.apply_ufunc(
                sosfiltfilt, sos, fnirs_run.pint.dequantify(),
                input_core_dims=[[], ['time']],
                output_core_dims=[['time']],
                exclude_dims={'time'},
            ).transpose(*fnirs_run.dims).pint.quantify(fnirs_units)
            fnirs_run = fnirs_run.assign_coords({'time': all_runs[run_idx].time})

        # EEG <-> fNIRS clock offset for this run (nirs_time = eeg_time + t_offset)
        eeg_ev_df = eeg_ev_dfs[run_key]
        nirs_ev_df = nirs_ev_dfs[matched_nirs_file[run_key]]
        t_offset = matched_t_offset[run_key]

        # window from the first event onset to the last event's end (onset + duration), in fNIRS time
        nirs_t_start = nirs_ev_df['onset'].values[0]
        # nirs_t_stop = (nirs_ev_df['onset'] + nirs_ev_df['duration']).values[-1]
        nirs_t_stop = (nirs_ev_df['onset']).values[-1]+len_delay # second
        eeg_t_start = nirs_t_start - t_offset
        eeg_t_stop = nirs_t_stop - t_offset

        EEG = single_subj_EEG_dict[run_key].copy()
        EEG_raw = single_subj_EEG_dict[run_key].copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))

        # truncate fNIRS to the same shared event window
        fnirs_run = fnirs_run.sel(time=slice(max(nirs_t_start, fnirs_run.time.values[0]),
                                            min(nirs_t_stop, fnirs_run.time.values[-1])))
        n_fnirs_samples = len(fnirs_run.time)
        fnirs_run_raw = fnirs_run_raw.sel(time=slice(max(nirs_t_start, fnirs_run_raw.time.values[0]),
                                            min(nirs_t_stop, fnirs_run_raw.time.values[-1])))

        # downsample EEG to match the number of sample points in fNIRS
        EEG_resample = EEG_raw.copy()
        EEG_resample.resample(fnirs_sfreq, npad='auto')

        # enforce exact sample-count match with the truncated fNIRS run
        if EEG_resample.n_times > n_fnirs_samples:
            EEG_resample.crop(tmax=EEG_resample.times[n_fnirs_samples - 1])
            EEG_raw.crop(tmax=EEG_resample.times[n_fnirs_samples - 1])
        elif EEG_resample.n_times < n_fnirs_samples:
            fnirs_run = fnirs_run.isel(time=slice(0, EEG_resample.n_times))
            n_fnirs_samples = EEG_resample.n_times
            fnirs_run_raw = fnirs_run_raw.isel(time=slice(0, EEG_resample.n_times))

        # truncate fNIRS so the delay at the beginning of the recording is removed
        fnirs_run = fnirs_run.isel(time=slice(np.round(len_delay*fnirs_sfreq).astype(int), n_fnirs_samples))
        fnirs_run_raw = fnirs_run_raw.isel(time=slice(np.round(len_delay*fnirs_sfreq).astype(int), n_fnirs_samples))

        # reset fnirs_run.time to 0
        fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
        fnirs_run_raw = fnirs_run_raw.assign_coords(time=fnirs_run_raw.time.values - fnirs_run_raw.time.values[0])

        # remember this run's time window so it can be reapplied to an all-parcel copy later
        run_time_windows[run_key] = (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)

        # shift city/mnt-correct trial onsets (originally in nirs_run time) into this run's
        # truncated, zero-based local time frame: local_t = nirs_onset - nirs_t_start - len_delay
        trials_this_run = stims_pruned_list[run_idx].copy()
        trials_this_run['onset'] = trials_this_run['onset'] - nirs_t_start - len_delay
        stims_pruned_truncated[run_key] = trials_this_run

        # append data
        fnirs_raw_list.append(fnirs_run_raw)
        all_runs_truncated.append(fnirs_run)
        eeg_list.append(EEG_resample)
        eeg_raw_list.append(EEG_raw)

    all_runs = all_runs_truncated

    #%% Regress GSR out before fitting EEG
    if USE_GSR and is_GSR_then_Others:
        gs_regressors = model.get_global_mean_regressor(all_runs)
        pred_runs = []
        resid_runs = []
        for run, gs_dm in zip(all_runs, gs_regressors):
            gsr_results = glm.fit(run, gs_dm, noise_model=cfg_GLM['noise_model'])
            gsr_pred = glm.predict(run, gsr_results.sm.params, gs_dm)
            pred_runs.append(gsr_pred)
            # change unit to molar
            gsr_pred = gsr_pred.pint.dequantify().pint.quantify('molar')
            resid = run - gsr_pred
            resid = resid.transpose(*run.dims)
            resid_runs.append(resid)
        all_runs = resid_runs

        # fNIRS check
        if is_plot:
            run_i = 0
            raw_run = fnirs_raw_list[run_i]
            hpf_run = all_runs_truncated[run_i]
            range_i = [0,1000]
            plt.figure()
            plt.plot(raw_run.time[range_i[0]:range_i[1]], raw_run.sel(parcel=select_parcel).values.flatten()[range_i[0]:range_i[1]], label='Raw fNIRS', alpha=0.7)
            plt.plot(hpf_run.time[range_i[0]:range_i[1]], hpf_run.sel(parcel=select_parcel).values.flatten()[range_i[0]:range_i[1]], label='HRF fNIRS', alpha=0.7)
            plt.plot(pred_runs[run_i].time[range_i[0]:range_i[1]], pred_runs[run_i].sel(parcel=select_parcel).values.flatten()[range_i[0]:range_i[1]], label='GSR pred', alpha=0.7)
            plt.plot(resid_runs[run_i].time[range_i[0]:range_i[1]], resid_runs[run_i].sel(parcel=select_parcel).values.flatten()[range_i[0]:range_i[1]], label='Resid', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.title(f'GSR on HRF fNIRS (sub-{subj_id}_run-0{run_i+1})')
            plt.grid()
            plt.legend()

    #%% Extract EEG values for DM
    if 'pc1' in eeg_reg_type:
        # use the first PC across all EEG channels
        eeg_reg_value_list = []
        for x in eeg_list:
            eeg_data = x.get_data(picks='eeg')  # channels x samples
            eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
            u, s, vt = np.linalg.svd(eeg_data, full_matrices=False)
            pc1 = vt[0] * s[0]
            eeg_reg_value_list.append(pc1.flatten())
    elif 'power' in eeg_reg_type.lower():
        # sliding-window bandpower (mean across all EEG channels), centered on each fNIRS sample.
        # Use eeg_list's (untrimmed) sample times, matching what pc1/cz feed into
        # get_cont_EEG_regressor -- that function trims off the leading n_delay
        # samples itself, so the input here must NOT be pre-trimmed like all_runs_truncated.
        win_len = power_win_len if power_win_len is not None else 1 / fnirs_sfreq
        eeg_reg_value_dict = {band: [] for band in power_bands}
        for eeg_raw_run, eeg_resample_run in zip(eeg_raw_list, eeg_list):
            # sample centers in EEG_raw's local time frame (both start at the same crop point)
            sample_times = eeg_resample_run.times
            band_power = model.bandpower_sliding_window(eeg_raw_run, sample_times, win_len,
                                                          power_bands, picks='eeg',
                                                          method=power_method,
                                                          bandwidth=power_bandwidth,
                                                          sum_method=sum_method)
            for band, power_ts in band_power.items():
                eeg_reg_value_dict[band].append(power_ts)

    else:
        # extract EEG signal for creating DesignMatrix
        eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]

    #%% Build a per-run, per-trial-type sample mask over the *untrimmed* eeg_list time axis
    # (get_cont_EEG_regressor trims off the leading n_delay samples itself, so masks here
    # must be built against the same untrimmed sample_times used above, matching how
    # eeg_reg_value_list / eeg_reg_value_dict are indexed).
    # Each trial owns samples from its onset up to the next qualifying (city/mnt-correct)
    # trial's onset; samples before the first trial or after the last trial's window are
    # masked out (0) for both trial types.
    trial_type_masks = {tt: [] for tt in trial_type_prefixes}
    for run_i, run_key in enumerate(['gradcpt1', 'gradcpt2', 'gradcpt3']):
        sample_times = eeg_list[run_i].times
        trials_df = stims_pruned_truncated[run_key].sort_values('onset').reset_index(drop=True)

        run_masks = {tt: np.zeros(len(sample_times), dtype=bool) for tt in trial_type_prefixes}
        for t_i in range(len(trials_df)):
            onset = trials_df.loc[t_i, 'onset']
            offset = trials_df.loc[t_i + 1, 'onset'] if t_i + 1 < len(trials_df) else sample_times[-1] + 1
            trial_label = trials_df.loc[t_i, 'trial_type']  # 'city-correct' or 'mnt-correct'
            prefix = trial_label.split('-')[0]  # 'city' or 'mnt'
            if prefix not in run_masks:
                continue
            in_window = (sample_times >= onset) & (sample_times < offset)
            run_masks[prefix] |= in_window

        for tt in trial_type_prefixes:
            if not run_masks[tt].any():
                print(f"{subject} {run_key}: no '{tt}' trials found, regressor will be all-zero for this run.")
            trial_type_masks[tt].append(run_masks[tt])

    # create EEG regressors: one delay-FIR regressor set per trial type, masked to that
    # trial type's windows. z-score (if requested) is computed on the full run signal
    # BEFORE masking, so per-trial-type regressors share the same scale/units, then the
    # already-z-scored signal is passed through with z_score=False.
    per_type_regressors = dict()
    for prefix in trial_type_prefixes:
        if 'power' in eeg_reg_type.lower():
            per_run_band_regressors = []
            for band in power_bands:
                masked_runs = []
                for run_i, power_ts in enumerate(eeg_reg_value_dict[band]):
                    sig = power_ts.copy()
                    if is_norm:
                        sig = (sig - np.nanmean(sig)) / np.nanstd(sig)
                    sig = np.where(trial_type_masks[prefix][run_i], sig, 0.0)
                    masked_runs.append(sig)
                per_run_band_regressors.append(
                    model.get_cont_EEG_regressor(masked_runs, fnirs_sfreq, delay=len_delay,
                                                  name_prefix=f'{prefix}_{band}_', z_score=False)
                )
            type_regressors = []
            for band_dms in zip(*per_run_band_regressors):
                run_dm = band_dms[0]
                for band_dm in band_dms[1:]:
                    run_dm = model.combine_dm(run_dm, band_dm)
                type_regressors.append(run_dm)
            per_type_regressors[prefix] = type_regressors
        else:
            masked_runs = []
            for run_i, sig in enumerate(eeg_reg_value_list):
                sig = sig.copy()
                if is_norm:
                    sig = (sig - sig.mean()) / sig.std()
                sig = np.where(trial_type_masks[prefix][run_i], sig, 0.0)
                masked_runs.append(sig)
            per_type_regressors[prefix] = model.get_cont_EEG_regressor(
                masked_runs, fnirs_sfreq, delay=len_delay, name_prefix=f'{prefix}_', z_score=False)

    # combine trial-type regressor sets (per run) into one design matrix per run
    eeg_regressors = []
    for run_i in range(len(all_runs)):
        run_dm = per_type_regressors[trial_type_prefixes[0]][run_i]
        for prefix in trial_type_prefixes[1:]:
            run_dm = model.combine_dm(run_dm, per_type_regressors[prefix][run_i])
        eeg_regressors.append(run_dm)

    # concatenate all runs and dms
    Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, eeg_regressors)

    #%% Low-rank representation of Delay using BSpline (fit independently per trial type,
    # so each trial type's FIR delay curve is smoothed on its own low-rank subspace rather
    # than sharing a single basis across both trial types' concatenated delay indices)
    all_regressor_names = dm_all.common.regressor.values
    basis_blocks = []
    n_regressor_per_type = dict()
    for prefix in trial_type_prefixes:
        if 'power' in eeg_reg_type.lower():
            for band in power_bands:
                band_regressor_names = [name for name in all_regressor_names
                                         if name.startswith(f'{prefix}_{band}_delay')]
                n_band_regressor = len(band_regressor_names)
                delay_idx = np.arange(n_band_regressor)
                bspline_basis = BSplines(delay_idx, df=[n_bspline_basis], degree=[bspline_degree],
                                          include_intercept=True).basis  # (n_band_regressor, n_bspline_basis)
                basis_blocks.append(xr.DataArray(
                    bspline_basis,
                    dims=("regressor", "component"),
                    coords={"regressor": band_regressor_names,
                            "component": [f"{prefix}_{band}_bspline{i}" for i in range(n_bspline_basis)]},
                ))
                n_regressor_per_type[f'{prefix}_{band}'] = n_band_regressor
        else:
            type_regressor_names = [name for name in all_regressor_names
                                     if name.startswith(f'{prefix}_delay')]
            n_type_regressor = len(type_regressor_names)
            delay_idx = np.arange(n_type_regressor)
            bspline_basis = BSplines(delay_idx, df=[n_bspline_basis], degree=[bspline_degree],
                                      include_intercept=True).basis  # (n_type_regressor, n_bspline_basis)
            basis_blocks.append(xr.DataArray(
                bspline_basis,
                dims=("regressor", "component"),
                coords={"regressor": type_regressor_names,
                        "component": [f"{prefix}_bspline{i}" for i in range(n_bspline_basis)]},
            ))
            n_regressor_per_type[prefix] = n_type_regressor
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

    #%% get GLM fitting results for each subject from shank Jun 02 2025
    print(f"Start cont_EEG GLM fitting ({subject})")
    results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    # extract HRF (delay-regressor betas) per parcel and per trial type, then expand the
    # low-rank bspline coefficients back to full per-delay resolution via the same basis
    betas_all = results.sm.params.copy()
    betas_eeg_per_type = dict()
    for prefix in trial_type_prefixes:
        eeg_reg = [p for p in betas_all.regressor.values if p.startswith(f'{prefix}_') and 'bspline' in p]
        betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
        type_basis_da = basis_da.sel(component=eeg_reg)
        betas_eeg = xr.dot(betas_bspline, type_basis_da, dims="component")
        type_regressor_names = [name for name in all_regressor_names
                                     if name.startswith(f'{prefix}_delay')]
        betas_eeg = betas_eeg.sel(regressor=type_regressor_names)
        betas_eeg_per_type[prefix] = betas_eeg

    #%% visual check fit results and HRF
    if is_plot:
        plt_type = 'mnt'
        parcel_names = [p for p in betas_eeg_per_type[plt_type].parcel.values if not p.startswith('Background+FreeSurfer')]
        select_network = select_parcel.split('_')[0]
        net_parcels = [p for p in parcel_names if p.split('_')[0] == select_network]

        vis_betas = results.sm.params.copy()
        eeg_reg = [p for p in vis_betas.regressor.values if 'bspline' in p]
        vis_betas = vis_betas.sel(regressor=eeg_reg)
        y_hat = xr.dot(dm_all.common, vis_betas, dims='regressor')
        #
        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=False)
        y_true = Y_all.sel(parcel=select_parcel).values.flatten()
        y_hat_vals = y_hat.sel(parcel=select_parcel).values.flatten()

        axs[0].plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=2)
        axs[0].plot(y_hat.time.values, y_hat_vals,'b',
                label='y_hat (EEG)', alpha=0.5)
        axs[0].set_ylabel(f'HbO concentration')
        axs[0].set_title(f'Parcel activities estimation ({select_parcel})')
        axs[0].legend()
        axs[0].grid()
        t_betas = np.arange(0,len_delay,1/fnirs_sfreq)
        for prefix in trial_type_prefixes:
            betas_eeg = betas_eeg_per_type[prefix]
            axs[1].plot(t_betas, betas_eeg.sel(parcel=select_parcel).values.flatten(), label=f'HRF {prefix} ({select_parcel})')
            axs[1].plot(t_betas, betas_eeg.sel(parcel=net_parcels).mean('parcel').values.flatten(), '--', label=f'HRF {prefix} ({select_network})')
        axs[1].set_title(f'HRF estimation using Cz EEG, city_correct vs mnt_correct')
        axs[1].legend()
        axs[1].grid()

    #%% save betas for later visualization
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg_per_type'] = betas_eeg_per_type
        betas_dict['basis_da'] = basis_da
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)
