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
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% find subjects with fNIRS and enough EEG epochs
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')
_MIN_EPOCHS = 500

_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf'))
    if re.search(r'sub-(\d+)', f)
}

_gradcpt_fifs = sorted(glob.glob(os.path.join(_eeg_deriv, 'sub-*', 'preprocessed_EEG_and_events', '*task-gradCPT*preproc_eeg.fif')))
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
eeg_reg_type = 'cont_EEG_cz_mnt_correct_incorrect'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = True # If True, load the highpassed (Hp) shared Y_all
is_plot = False # If True, generate visualization plots
select_chromo='HbO'
# select_parcel='DorsAttnA_ParOcc_1_RH'
select_parcel='DefaultA_PFCd_1_LH'
len_delay = 15 # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)
is_fit_irrr = True # If True, also fit iRRR on the same Y_all and dm_all (one predictor block per trial type)
irrr_lam1 = 1.0 # iRRR nuclear-norm penalty (Y is scaled to unit std before fitting, so this is scale-free)

# trial types with their own delayed-EEG regressor set, zeroed outside that trial type's trials
# (onset -> onset + duration). Same pipeline as cont_EEG_cz_3-stage_bspline-test in
# run_model_cont_EEG_fNIRS.py otherwise, and fit on the same shared Y_all.
# The delays are built first, then the delay-matrix rows (time points) outside the trials are zeroed,
# then projected onto the B-splines -- only time points within a trial are modeled, each by the EEG
# of the preceding len_delay seconds.
trial_type_selectors = {
    'mnt_correct': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] == 0),
    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),
}
trial_type_prefixes = list(trial_type_selectors.keys())

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
    save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')
    betas_save_path = get_betas_path(save_prefix)
    stats_save_path = get_stats_path(save_prefix)
    dm_all_save_path = get_dm_all_path(save_prefix)
    irrr_save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}')
    irrr_betas_save_path = get_betas_path(irrr_save_prefix)
    irrr_stats_save_path = get_stats_path(irrr_save_prefix)
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% load Y_all (e.g. sub-730_parcel_Y_all_truncated_to_trials_Hp.pkl.gz; drift and GSR already
    # OLS-regressed out, runs concatenated in gradcpt1/2/3 order)
    Y_all_load_path = get_shared_Y_all_path(data_save_path, subject, hp_flag)
    if not os.path.exists(Y_all_load_path):
        print(f"{subject}: {Y_all_load_path} not found, skipping.")
        continue
    with gzip.open(Y_all_load_path, 'rb') as f:
        Y_all = pickle.load(f)
    Y_all = Y_all.sel(chromo=[select_chromo])

    # run boundaries in Y_all: the 'samples' coord keeps each run's original fNIRS sample index
    samples = Y_all.samples.values
    seg_starts = np.r_[0, np.where(np.diff(samples) != 1)[0] + 1]
    seg_stops = np.r_[seg_starts[1:], len(samples)]
    assert len(seg_starts) == 3, f"{subject}: expected 3 runs in Y_all, found {len(seg_starts)}"
    fnirs_sfreq = 1 / np.median(np.diff(Y_all.time.values))

    #%% load stims
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
    with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        all_stims = pickle.load(f)['stims']

    #%% get continous EEG
    eeg_der_dir = os.path.join(project_path, "derivatives", "eeg")
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    # check if Cz exists
    cz_removed = any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]]
                    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3'])
    if cz_removed:
        print(f"sub-{subj_id}: Cz was removed in at least one run, skipping subject.")
        continue

    # match each fNIRS run in all_stims to its EEG run (gradcpt1/2/3) via first stim onset in events.tsv
    eeg_ev_files = {
        run_key: os.path.join(eeg_der_dir, subject, 'preprocessed_EEG_and_events', f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv")
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

    #%% Synchronize EEG and fNIRS, and mask Cz EEG per trial type
    # EEG is cropped exactly as in run_model_cont_EEG_fNIRS.py: from len_delay before the first event
    # (so the first fNIRS sample already has a full delay history) to len_delay after the last event
    eeg_list = []
    eeg_masks = {tt: [] for tt in trial_type_prefixes}  # per run, bool over the cropped EEG samples
    n_trials_per_type = {tt: 0 for tt in trial_type_prefixes}
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        nirs_ev_df = nirs_ev_dfs[matched_nirs_file[run_key]]
        t_offset = matched_t_offset[run_key]

        nirs_t_start = nirs_ev_df['onset'].values[0]
        nirs_t_stop = (nirs_ev_df['onset']).values[-1]+len_delay # second
        eeg_t_start = nirs_t_start - t_offset - len_delay # second
        eeg_t_stop = nirs_t_stop - t_offset

        EEG = single_subj_EEG_dict[run_key].copy()
        EEG_raw = single_subj_EEG_dict[run_key].copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))
        eeg_list.append(EEG_raw)

        # cropped EEG sample times in fNIRS time (same frame as the stim onsets)
        eeg_crop_start = EEG_raw.first_time - EEG.first_time
        sample_nirs_t = EEG_raw.times + eeg_crop_start + t_offset

        stim = all_stims[run_key_to_run_idx[run_key]]
        for tt, selector in trial_type_selectors.items():
            trials_df = stim[selector(stim)]
            n_trials_per_type[tt] += len(trials_df)
            run_mask = np.zeros(len(sample_nirs_t), dtype=bool)
            for onset, duration in zip(trials_df['onset'].values, trials_df['duration'].values):
                run_mask |= (sample_nirs_t >= onset) & (sample_nirs_t < onset + duration)
            eeg_masks[tt].append(run_mask)

    for tt in trial_type_prefixes:
        print(f"{subject} {tt}: {n_trials_per_type[tt]} trials")

    #%% create EEG regressors: per trial type, build the delayed Cz EEG regressors, mask them to that
    # trial type's trials (zero the delay-matrix rows outside them) and
    # project the delay axis onto the low-rank B-spline basis (at EEG rate). The projection is linear
    # along the delay axis, so zeroing delay-matrix rows before or after it gives the same result.
    eeg_sfreq = eeg_list[0].info['sfreq']
    n_regressor = np.round(len_delay*eeg_sfreq).astype(int)
    delay_names = [f"delay{d_i}" for d_i in range(n_regressor)]
    bspline_basis = BSplines(np.arange(n_regressor), df=[n_bspline_basis], degree=[bspline_degree],
                              include_intercept=True).basis  # (n_regressor, n_bspline_basis)
    basis_da = xr.DataArray(
        bspline_basis,
        dims=("regressor", "component"),
        coords={"regressor": delay_names,
                "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
    )

    eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]
    regressor_names = [f'{tt}_bspline{i}' for tt in trial_type_prefixes for i in range(n_bspline_basis)]
    eeg_regressors = []
    for run_i, (seg_start, seg_stop) in enumerate(zip(seg_starts, seg_stops)):
        # (n_regressor_total x n_eeg_samples), trial types stacked along the regressor axis
        # delay-matrix rows are EEG samples n_regressor..end (leading n_regressor samples removed)
        eeg_dm = model.get_cont_EEG_bspline_regressor(eeg_reg_value_list[run_i], bspline_basis)
        run_dm = np.concatenate([
            eeg_dm * eeg_masks[tt][run_i][n_regressor:, None]
            for tt in trial_type_prefixes], axis=1).T

        #%% Downsample EEG DM to fNIRS sampling rate and match this run's samples in Y_all
        run_dm = mne.filter.resample(run_dm, up=fnirs_sfreq, down=eeg_sfreq, npad='auto', axis=-1)
        n_fnirs_samples = seg_stop - seg_start
        if run_dm.shape[-1] < n_fnirs_samples:
            raise ValueError(f"{subject} run {run_i}: EEG DM has {run_dm.shape[-1]} samples, Y_all has {n_fnirs_samples}.")
        run_dm = run_dm[:, :n_fnirs_samples]
        eeg_regressors.append(xr.DataArray(
            run_dm.T[:, :, None],
            dims=("time", "regressor", "chromo"),
            coords={"time": Y_all.time.values[seg_start:seg_stop], "regressor": regressor_names, "chromo": [select_chromo]},
        ))

    dm_all = glm.design_matrix.DesignMatrix(common=xr.concat(eeg_regressors, dim='time').fillna(0), channel_wise=[])

    #%% get GLM fitting results
    print(f"Start cont_EEG GLM fitting ({subject})")
    results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    # extract HRF (delay-regressor betas) per parcel and per trial type, then expand the
    # low-rank bspline coefficients back to full per-delay resolution via the same basis
    betas_all = results.sm.params.copy()
    betas_eeg_per_type = dict()
    for tt in trial_type_prefixes:
        eeg_reg = [p for p in betas_all.regressor.values if p.startswith(f'{tt}_bspline')]
        betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
        betas_bspline = betas_bspline.assign_coords(component=[c[len(f'{tt}_'):] for c in eeg_reg])
        betas_eeg_per_type[tt] = xr.dot(betas_bspline, basis_da, dims="component")

    #%% f test and contrast t test (per trial type)
    stats_dict = dict()
    for tt in trial_type_prefixes:
        # test if EEG can explain more variance
        param_names = [name for name in betas_all.regressor.values if name.startswith(f'{tt}_bspline')]
        # Create hypothesis strings
        hypotheses = [f'{name} = 0' for name in param_names]
        # Run F-test
        stats_dict[f'f_test_full_noEEG_{tt}'] = results.sm.f_test(hypotheses)

        # test if EEG betas sums to 0
        hypotheses = '+'.join(param_names)+' = 0'
        # Run t-test
        stats_dict[f't_test_0_eeg_{tt}'] = results.sm.t_test(hypotheses)
    stats_dict['Y_all_path'] = Y_all_load_path

    #%% iRRR: fit all parcels jointly, Y (time x parcel) ~ sum_k X_k B_k with one predictor block X_k
    # (time x bspline) per trial type, and a nuclear-norm penalty on each (bspline x parcel) B_k so the
    # HRFs across parcels share a low-rank structure within each trial type
    if is_fit_irrr:
        print(f"Start cont_EEG iRRR fitting ({subject})")
        Y_np = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel').values
        X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
        irrr_reg_per_type = {tt: [r for r in X_da.regressor.values if r.startswith(f'{tt}_bspline')] for tt in trial_type_prefixes}
        X_np_list = [X_da.sel(regressor=irrr_reg_per_type[tt]).values for tt in trial_type_prefixes]
        Y_scale = np.nanstd(Y_np)
        n_time, n_parcel = Y_np.shape
        irrr_weight = []
        for X_np in X_np_list:
            X_c = X_np - X_np.mean(0, keepdims=True)
            irrr_weight.append(np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / n_time)
        C, mu, A, _, _, irrr_details = irrr_normal(Y_np / Y_scale, X_np_list, irrr_lam1,
                                                   {'varyrho': True, 'Tol': 0.01, 'fig': False,
                                                    'weight': irrr_weight},
                                                   return_details=True)
        C = C * Y_scale  # back to original Y units
        mu = mu * Y_scale
        irrr_betas_all = xr.DataArray(
            C.T[:, None, :],
            dims=('parcel', 'chromo', 'regressor'),
            coords={'parcel': Y_all.parcel.values, 'chromo': [select_chromo],
                    'regressor': np.concatenate([irrr_reg_per_type[tt] for tt in trial_type_prefixes])},
        )
        irrr_stats_dict = {'irrr_lam1': irrr_lam1, 'irrr_weight': irrr_weight, 'Y_scale': Y_scale,
                           'intercept': mu, 'details': irrr_details, 'Y_all_path': Y_all_load_path}
        irrr_betas_eeg_per_type = dict()
        for tt, A_k in zip(trial_type_prefixes, A):
            irrr_stats_dict[f'rank_{tt}'] = matrix_rank(A_k)
            irrr_stats_dict[f'singular_values_{tt}'] = svdvals(A_k * Y_scale)
            # expand the low-rank bspline coefficients back to full per-delay resolution
            betas_bspline = irrr_betas_all.sel(regressor=irrr_reg_per_type[tt]).rename({"regressor": "component"})
            betas_bspline = betas_bspline.assign_coords(component=[c[len(f'{tt}_'):] for c in irrr_reg_per_type[tt]])
            irrr_betas_eeg_per_type[tt] = xr.dot(betas_bspline, basis_da, dims="component")
        print(f"iRRR fit: " + ', '.join(f"rank(B_{tt}) = {irrr_stats_dict[f'rank_{tt}']}" for tt in trial_type_prefixes)
              + f" (of {n_bspline_basis})")

    #%% visual check fit results and HRF
    if is_plot:
        parcel_names = [p for p in betas_eeg_per_type[trial_type_prefixes[0]].parcel.values if not p.startswith('Background+FreeSurfer')]
        select_network = select_parcel.split('_')[0]
        net_parcels = [p for p in parcel_names if p.split('_')[0] == select_network]

        y_hat = xr.dot(dm_all.common, betas_all, dims='regressor')
        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=False)
        axs[0].plot(Y_all.time.values, Y_all.sel(parcel=select_parcel).values.flatten(), label='Y (true)', color='k', linewidth=2)
        axs[0].plot(y_hat.time.values, y_hat.sel(parcel=select_parcel).values.flatten(), 'b', label='y_hat (EEG)', alpha=0.5)
        axs[0].set_ylabel(f'HbO concentration')
        axs[0].set_title(f'Parcel activities estimation ({select_parcel})')
        axs[0].legend()
        axs[0].grid()
        for tt in trial_type_prefixes:
            betas_eeg = betas_eeg_per_type[tt]
            t_betas = np.linspace(0, len_delay, len(betas_eeg.regressor.values))
            axs[1].plot(t_betas, betas_eeg.sel(parcel=select_parcel).values.flatten(), label=f'HRF {tt} ({select_parcel})')
            axs[1].plot(t_betas, betas_eeg.sel(parcel=net_parcels).mean('parcel').values.flatten(), '--', label=f'HRF {tt} ({select_network})')
        axs[1].set_title(f'HRF estimation using Cz EEG, mnt_correct vs mnt_incorrect')
        axs[1].legend()
        axs[1].grid()

    #%% save betas for later visualization (Y_all is not re-saved; stats_dict points to the shared copy)
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg_per_type'] = betas_eeg_per_type
        betas_dict['basis_da'] = basis_da
        os.makedirs(os.path.dirname(betas_save_path), exist_ok=True)
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)

        os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
        with open(stats_save_path, 'wb') as f:
            pickle.dump(stats_dict, f)

        os.makedirs(os.path.dirname(dm_all_save_path), exist_ok=True)
        with gzip.open(dm_all_save_path, 'wb') as f:
            pickle.dump(dm_all, f)

        # iRRR results (same Y_all and dm_all as AR-IRLS, so neither is re-saved)
        if is_fit_irrr:
            irrr_stats_dict['dm_all_path'] = dm_all_save_path
            irrr_betas_dict = dict()
            irrr_betas_dict['betas'] = irrr_betas_all
            irrr_betas_dict['betas_eeg_per_type'] = irrr_betas_eeg_per_type
            irrr_betas_dict['basis_da'] = basis_da
            os.makedirs(os.path.dirname(irrr_betas_save_path), exist_ok=True)
            with open(irrr_betas_save_path, 'wb') as f:
                pickle.dump(irrr_betas_dict, f)

            os.makedirs(os.path.dirname(irrr_stats_save_path), exist_ok=True)
            with open(irrr_stats_save_path, 'wb') as f:
                pickle.dump(irrr_stats_dict, f)
