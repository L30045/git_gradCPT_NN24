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
import cedalion.io
import cedalion.models.glm as glm
from cedalion.sigproc import frequency
from statsmodels.gam.smooth_basis import BSplines
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, filtfilt, windows
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = cedalion.io.load_Adot(Adot_path + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
mask_medial = Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH',
    'Background+FreeSurfer_Defined_Medial_Wall_RH'
])
Adot_brain = Adot_brain.sel(vertex=~mask_medial)

intensity = np.log10(Adot_brain[:, :, 1].sum('channel'))
sensitivity_mask = (intensity > -2).drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values)
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex')
Adot_parcel = Adot_parcel.assign_coords(
    {'is_brain': ('parcel', np.ones(len(Adot_parcel.parcel), dtype=bool))}
)
sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (429 of 601)

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
t_pre = 2 # EEG lead (sec): each fNIRS sample also sees the EEG up to t_pre seconds after it
t_post = 18 # EEG lag (sec): each fNIRS sample sees the EEG up to t_post seconds before it
len_delay = t_pre + t_post # length of the delay window [-t_pre, t_post) (sec)
eeg_reg_type = f'cont_EEG_cz_mnt_correct_incorrect_pre{t_pre}post{t_post}'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save Y_all, DM and GLM results
is_hp_fNIRS = True # If True, highpass fNIRS by fnirs_hp_cutoff
fnirs_hp_cutoff = 0.01 # fNIRS highpass cutoff (Hz)
is_plot = False # If True, generate visualization plots
select_chromo='HbO'
# select_parcel='DorsAttnA_ParOcc_1_RH'
select_parcel='DefaultA_PFCd_1_LH'
USE_GSR=True
cfg_GLM['do_GSR']=USE_GSR
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)

# fixAR-IRLS / AR-iRRR: spectral whitening of Y and X, then IRLS per parcel / iRRR across parcels
is_scale_Y = True # If True, scale whitened Y to unit std before the IRLS / iRRR fits (betas are scaled back)
max_jobs = -1 # parallel jobs across parcels (fixAR-IRLS)
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685) # same robust norm as model.my_ar_irls_GLM
# AR-iRRR nuclear-norm penalty: the smallest lam1 at which the largest rank(B_k) across trial types is
# irrr_target_rank. cand_irrr_lam1 (scale-free only when is_scale_Y; otherwise in units of Y) brackets it,
# then n_bisect bisection steps on log lam1 refine it
irrr_target_rank = 2
if is_scale_Y:
    cand_irrr_lam1 = 10 ** np.arange(-3, 2.01, 0.5)
else:
    cand_irrr_lam1 = 10 ** np.arange(-11, -8, 0.25)
n_bisect = 8
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf')

# trial types with their own delayed-EEG regressor set, zeroed outside that trial type's trials
# (onset -> onset + duration). Preprocessing of Y is the same as cont_EEG_cz_3-stage_bspline-test in
# run_model_cont_EEG_fNIRS.py (sensitive parcels, HbO, highpass, per-run window, drift/GSR
# OLS-regressed out), but with fnirs_hp_cutoff and the [-t_pre, t_post) delay window.
# The Cz EEG outside that trial type's trials is zeroed first, then the delays are built, projected onto
# the B-splines and downsampled -- every fNIRS sample is modeled, by the within-trial EEG from t_post
# seconds before to t_pre seconds after it.
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
    hp_flag = f'Hp{fnirs_hp_cutoff}' if is_hp_fNIRS else 'noHp'
    save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')
    betas_save_path = get_betas_path(save_prefix)
    stats_save_path = get_stats_path(save_prefix)
    Y_all_save_path = get_own_Y_all_path(save_prefix)
    dm_all_save_path = get_dm_all_path(save_prefix)
    fixar_save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_fixAR-IRLS{get_scale_tag(is_scale_Y)}_{hp_flag}')
    irrr_save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_iRRR_AR{get_scale_tag(is_scale_Y)}_{hp_flag}')
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% RUN PREPROCESSING
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

    print('LOADING PREPROCESSED CHANNEL DATA')
    with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        results = pickle.load(f)
    all_stims = results['stims']

    print('LOADING IMAGE SPACE RESULTS')
    folder =  os.path.join(der_dir, subject)
    filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'
    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)
    all_runs = image_results['parcel_ts']

    all_runs_tmp = []
    for run in all_runs:
        if F_MIN > 0:
            run.time.attrs['units'] = units.s
            run_filt = frequency.freq_filter(run, F_MIN*units.Hz, F_MAX*units.Hz)
            all_runs_tmp.append(run_filt)
        else:
            all_runs_tmp.append(run)
    all_runs = all_runs_tmp

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel = run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp.copy()

    # mask out parcels with low forward-model sensitivity (601 -> 429 parcels)
    all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels)) for run in all_runs]

    if select_chromo is not None:
        all_runs = [x.sel(chromo=[select_chromo]) for x in all_runs]

    ori_all_runs = all_runs.copy()

    #%% get continous EEG
    eeg_der_dir = os.path.join(project_path, "derivatives", "eeg")
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    # check if Cz exists
    cz_removed = any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]]
                    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3'])
    if cz_removed:
        print(f"sub-{subj_id}: Cz was removed in at least one run, skipping subject.")
        continue

    # match each fNIRS run in all_runs to its EEG run (gradcpt1/2/3) via first stim onset in events.tsv
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
    # fNIRS window per run: t_pre before the first event to t_post after the last event (fNIRS time).
    # EEG is cropped to [window start - t_post, window end + t_pre] so every fNIRS sample has its full
    # [-t_pre, t_post) delay history; if the EEG recording does not cover that, the fNIRS window is
    # shrunk to what the EEG supports (instead of misaligning EEG and fNIRS).
    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

    eeg_list = []
    all_runs_truncated = []
    eeg_masks = {tt: [] for tt in trial_type_prefixes}  # per run, bool over the cropped EEG samples
    n_trials_per_type = {tt: 0 for tt in trial_type_prefixes}
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_idx = run_key_to_run_idx[run_key]
        fnirs_run = ori_all_runs[run_idx].copy()

        # highpass fNIRS to remove drift
        if is_hp_fNIRS:
            fnirs_units = fnirs_run.pint.units
            sos = butter(4, fnirs_hp_cutoff, btype='highpass', fs=fnirs_sfreq, output='sos')
            fnirs_run = xr.apply_ufunc(
                sosfiltfilt, sos, fnirs_run.pint.dequantify(),
                input_core_dims=[[], ['time']],
                output_core_dims=[['time']],
                exclude_dims={'time'},
            ).transpose(*fnirs_run.dims).pint.quantify(fnirs_units)
            fnirs_run = fnirs_run.assign_coords({'time': ori_all_runs[run_idx].time})

        # EEG <-> fNIRS clock offset for this run (nirs_time = eeg_time + t_offset)
        nirs_ev_df = nirs_ev_dfs[matched_nirs_file[run_key]]
        t_offset = matched_t_offset[run_key]
        EEG = single_subj_EEG_dict[run_key]

        # fNIRS window (fNIRS time), clamped to the fNIRS run and to the span the EEG can cover
        nirs_t_start = max(nirs_ev_df['onset'].values[0] - t_pre,
                           fnirs_run.time.values[0],
                           EEG.times[0] + t_offset + t_post)
        nirs_t_stop = min(nirs_ev_df['onset'].values[-1] + t_post,
                          fnirs_run.time.values[-1],
                          EEG.times[-1] + t_offset - t_pre)
        fnirs_run = fnirs_run.sel(time=slice(nirs_t_start, nirs_t_stop))
        nirs_t_start = fnirs_run.time.values[0]  # snap to the first kept fNIRS sample

        eeg_t_start = nirs_t_start - t_offset - t_post # second
        eeg_t_stop = fnirs_run.time.values[-1] - t_offset + t_pre # second
        EEG_raw = EEG.copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))
        eeg_list.append(EEG_raw)

        # trial mask over the cropped EEG samples, in fNIRS time (same frame as the stim onsets)
        eeg_crop_start = EEG_raw.first_time - EEG.first_time
        sample_nirs_t = EEG_raw.times + eeg_crop_start + t_offset
        stim = all_stims[run_idx]
        for tt, selector in trial_type_selectors.items():
            trials_df = stim[selector(stim)]
            n_trials_per_type[tt] += len(trials_df)
            run_mask = np.zeros(len(sample_nirs_t), dtype=bool)
            for onset, duration in zip(trials_df['onset'].values, trials_df['duration'].values):
                run_mask |= (sample_nirs_t >= onset) & (sample_nirs_t < onset + duration)
            eeg_masks[tt].append(run_mask)

        # reset fnirs_run.time to 0
        fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
        all_runs_truncated.append(fnirs_run)

    all_runs = all_runs_truncated

    for tt in trial_type_prefixes:
        print(f"{subject} {tt}: {n_trials_per_type[tt]} trials")

    #%% 3-stage regression: OLS-regress out drift, then OLS-regress out GSR
    # (computed from the drift-residualized signal), leaving only the EEG delay
    # regressors for the final fits below
    if cfg_GLM['do_drift_legendre']:
        drift_dms = model.get_drift_legendre_regressors(all_runs, cfg_GLM)
    elif cfg_GLM['do_drift']:
        drift_dms = model.get_drift_regressors(all_runs, cfg_GLM)
    else:
        drift_dms = None

    if drift_dms is not None:
        resid_runs = []
        for run, drift_dm in zip(all_runs, drift_dms):
            drift_results = glm.fit(run, drift_dm, noise_model='ols')
            drift_fit = glm.predict(run, drift_results.sm.params, drift_dm)
            drift_fit = drift_fit.pint.dequantify().pint.quantify('molar')
            resid_runs.append((run - drift_fit).transpose(*run.dims))
        all_runs = resid_runs

    if USE_GSR:
        gsr_dms = model.get_global_mean_regressor(all_runs)
        resid_runs = []
        for run, gsr_dm in zip(all_runs, gsr_dms):
            gsr_results = glm.fit(run, gsr_dm, noise_model='ols')
            gsr_fit = glm.predict(run, gsr_results.sm.params, gsr_dm)
            gsr_fit = gsr_fit.pint.dequantify().pint.quantify('molar')
            resid_runs.append((run - gsr_fit).transpose(*run.dims))
        all_runs = resid_runs

    #%% create EEG regressors: per trial type, zero the Cz EEG outside that trial type's trials, build the
    # delayed regressors from the masked EEG and project the delay axis onto the low-rank B-spline basis
    # (at EEG rate), then downsample to the fNIRS rate. Every fNIRS sample is kept, so each trial's EEG
    # contributes to the fNIRS from t_pre before to t_post after it (an event-locked HRF over
    # [-t_pre, t_post) weighted by the within-trial EEG).
    # Row j of the EEG-rate delay matrix uses EEG samples j .. j + n_regressor - 1 (from t_post before
    # to t_pre after the row's fNIRS time); delay index k of the basis is delay k/eeg_sfreq - t_pre.
    eeg_sfreq = eeg_list[0].info['sfreq']
    n_regressor = np.round(len_delay*eeg_sfreq).astype(int)
    delay_names = [f"delay{d_i}" for d_i in range(n_regressor)]
    delay_times = np.arange(n_regressor) / eeg_sfreq - t_pre  # sec, -t_pre .. t_post
    bspline_basis = BSplines(np.arange(n_regressor), df=[n_bspline_basis], degree=[bspline_degree],
                              include_intercept=True).basis  # (n_regressor, n_bspline_basis)
    basis_da = xr.DataArray(
        bspline_basis,
        dims=("regressor", "component"),
        coords={"regressor": delay_names,
                "delay_time": ("regressor", delay_times),
                "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
    )

    eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]
    regressor_names = [f'{tt}_bspline{i}' for tt in trial_type_prefixes for i in range(n_bspline_basis)]
    eeg_regressors = []
    for run_i, fnirs_run in enumerate(all_runs):
        # (n_regressor_total x n_rows) at EEG rate, trial types stacked along the regressor axis
        run_dm = np.concatenate([
            model.get_cont_EEG_bspline_regressor(eeg_reg_value_list[run_i] * eeg_masks[tt][run_i], bspline_basis)
            for tt in trial_type_prefixes], axis=1).T

        #%% Downsample EEG DM to fNIRS sampling rate, then enforce an exact sample-count
        # match with the corresponding (already-truncated) fNIRS run and adopt its time coordinate.
        run_dm = mne.filter.resample(run_dm, up=fnirs_sfreq, down=eeg_sfreq, npad='auto', axis=-1)
        n_fnirs_samples = len(fnirs_run.time)
        n_dm_samples = run_dm.shape[-1]
        if n_dm_samples > n_fnirs_samples:
            run_dm = run_dm[:, :n_fnirs_samples]
        elif n_dm_samples < n_fnirs_samples:
            all_runs[run_i] = fnirs_run.isel(time=slice(0, n_dm_samples))

        eeg_regressors.append(glm.design_matrix.DesignMatrix(
            common=xr.DataArray(
                run_dm.T[:, :, None],
                dims=("time", "regressor", "chromo"),
                coords={"time": all_runs[run_i].time.values, "regressor": regressor_names, "chromo": [select_chromo]},
            ),
            channel_wise=[]))

    #%% concatenate all runs and dms
    Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, eeg_regressors)
    dm_all.common = dm_all.common.fillna(0)
    dm_all.common = dm_all.common.sel(chromo=[select_chromo])

    def expand_bspline_betas(betas_all):
        """per trial type: expand the low-rank bspline coefficients back to full per-delay resolution"""
        betas_eeg_per_type = dict()
        for tt in trial_type_prefixes:
            eeg_reg = [p for p in betas_all.regressor.values if p.startswith(f'{tt}_bspline')]
            betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
            betas_bspline = betas_bspline.assign_coords(component=[c[len(f'{tt}_'):] for c in eeg_reg])
            betas_eeg_per_type[tt] = xr.dot(betas_bspline, basis_da, dims="component")
        return betas_eeg_per_type

    #%% AR-IRLS (Cedalion)
    print(f"Start cont_EEG AR-IRLS fitting ({subject})")
    results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    betas_all = results.sm.params.copy()
    betas_eeg_per_type = expand_bspline_betas(betas_all)

    # f test and contrast t test (per trial type)
    stats_dict = dict()
    for tt in trial_type_prefixes:
        # test if EEG can explain more variance
        param_names = [name for name in betas_all.regressor.values if name.startswith(f'{tt}_bspline')]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict[f'f_test_full_noEEG_{tt}'] = results.sm.f_test(hypotheses)

        # test if EEG betas sums to 0
        hypotheses = '+'.join(param_names)+' = 0'
        stats_dict[f't_test_0_eeg_{tt}'] = results.sm.t_test(hypotheses)
    stats_dict['Y_all_path'] = Y_all_save_path
    stats_dict['dm_all_path'] = dm_all_save_path

    #%% spectral whitening of Y and X (shared by fixAR-IRLS and AR-iRRR)
    # the residual spectrum is estimated from the OLS residuals of Y_partial ~ 1 + X (Y_all is already
    # drift- and GSR-residualized), averaged across parcels, and its 1/sqrt(power) filter is applied
    # to both Y and X. Runs are padded and filtered separately.
    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    X_ols = np.column_stack([np.ones(len(X_da.time)), X_da.values])
    B_ols = np.linalg.lstsq(X_ols, Y_da.values, rcond=None)[0]
    samples = Y_da.samples.values  # per-run sample index; resets mark run starts in Y_all
    run_bounds = np.where(np.diff(samples) <= 0)[0] + 1
    noise_model = model.NoiseModel(method='spectral')
    noise_model.fit(model.split_runs(Y_da.values - X_ols @ B_ols, samples))
    Y_white = np.vstack(noise_model.whiten(model.split_runs(Y_da.values, samples)))
    X_white = np.vstack(noise_model.whiten(model.split_runs(X_da.values, samples)))
    print(f"spectral whitening: {len(run_bounds) + 1} runs, w_pad = {noise_model.w_pad}")
    # the whitened series are zero-mean (DC bin of the kernel is 0), so no intercept is added
    Y_scale = np.nanstd(Y_white) if is_scale_Y else 1.0
    parcels = Y_da.parcel.values
    whiten_stats = {'Y_scale': Y_scale, 'run_bounds': run_bounds,
                    'whiten_kernel_fft': noise_model.W_fft, 'acf_kernel': noise_model.acf_kernel,
                    'Y_all_path': Y_all_save_path, 'dm_all_path': dm_all_save_path}

    #%% fixAR-IRLS: IRLS fit per parcel on the whitened data (no further AR whitening)
    print(f"Start cont_EEG fixAR-IRLS fitting ({subject})")
    x_df = pd.DataFrame(X_white, columns=X_da.regressor.values)
    rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
        delayed(model.irls_fit)(pd.Series(Y_white[:, p_i] / Y_scale), x_df, M=irls_norm)
        for p_i in tqdm(range(len(parcels)))
    )
    fixar_betas_all = xr.DataArray(
        np.stack([res.params.values for res in rlm_results])[:, None, :] * Y_scale,
        dims=('parcel', 'chromo', 'regressor'),
        coords={'parcel': parcels, 'chromo': [select_chromo],
                'regressor': X_da.regressor.values},
    )
    fixar_betas_eeg_per_type = expand_bspline_betas(fixar_betas_all)

    # f test / contrast t test per parcel and trial type (statistics are in units of Y / Y_scale)
    fixar_stats_dict = dict(whiten_stats)
    for tt in trial_type_prefixes:
        param_names = [name for name in X_da.regressor.values if name.startswith(f'{tt}_bspline')]
        # test if EEG can explain more variance
        f_hypotheses = [f'{name} = 0' for name in param_names]
        fixar_stats_dict[f'f_test_full_noEEG_{tt}'] = {p: res.f_test(f_hypotheses) for p, res in zip(parcels, rlm_results)}
        # test if EEG betas sums to 0
        t_hypotheses = '+'.join(param_names) + ' = 0'
        fixar_stats_dict[f't_test_0_eeg_{tt}'] = {p: res.t_test(t_hypotheses) for p, res in zip(parcels, rlm_results)}

    #%% AR-iRRR: fit all parcels jointly on the whitened data, Y (time x parcel) ~ sum_k X_k B_k with one
    # predictor block X_k (time x bspline) per trial type, and a nuclear-norm penalty on each
    # (bspline x parcel) B_k so the HRFs across parcels share a low-rank structure within each trial type
    print(f"Start cont_EEG AR-iRRR fitting ({subject})")
    irrr_reg_per_type = {tt: [r for r in X_da.regressor.values if r.startswith(f'{tt}_bspline')] for tt in trial_type_prefixes}
    irrr_col_idx = [[list(X_da.regressor.values).index(r) for r in irrr_reg_per_type[tt]] for tt in trial_type_prefixes]
    n_time, n_parcel = Y_white.shape
    Y_s = Y_white / Y_scale

    def get_irrr_weight(X_list):
        # theoretical iRRR weight per block: max(svd(X_c)) * (sqrt(q) + sqrt(rank(X_c))) / n, X_c column-centered
        weights = []
        for X in X_list:
            X_c = X - X.mean(0, keepdims=True)
            weights.append(np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / X.shape[0])
        return weights

    # select irrr_lam1 so the fit has rank irrr_target_rank: the largest rank(B_k) across trial types
    # decreases as lam1 grows, so take the smallest lam1 (least shrinkage) at which it is <= irrr_target_rank.
    # cand_irrr_lam1 brackets the transition, then n_bisect bisection steps (on log lam1) refine it.
    X_list = [X_white[:, idx] for idx in irrr_col_idx]
    irrr_weight = get_irrr_weight(X_list)
    irrr_fits = dict()  # lam1 -> (C, mu, A, details)
    def fit_irrr(lam1):
        if lam1 not in irrr_fits:
            C_l, mu_l, A_l, _, _, details_l = irrr_normal(Y_s, X_list, lam1,
                                                          {'varyrho': True, 'Tol': 0.01, 'fig': False,
                                                           'weight': irrr_weight},
                                                          return_details=True)
            irrr_fits[lam1] = (C_l, mu_l, A_l, details_l)
        return irrr_fits[lam1]
    def block_ranks(lam1):
        return [matrix_rank(A_k) for A_k in fit_irrr(lam1)[2]]

    grid_max_rank = np.array([max(block_ranks(lam1)) for lam1 in tqdm(cand_irrr_lam1, desc='iRRR lam1 grid')])
    above = np.where(grid_max_rank > irrr_target_rank)[0]
    below = np.where(grid_max_rank <= irrr_target_rank)[0]
    if len(below) == 0:
        print(f"WARNING: rank > {irrr_target_rank} for every lam1 in cand_irrr_lam1; using the largest lam1.")
        irrr_lam1 = cand_irrr_lam1[-1]
    elif len(above) == 0:
        print(f"WARNING: rank <= {irrr_target_rank} for every lam1 in cand_irrr_lam1; using the smallest lam1.")
        irrr_lam1 = cand_irrr_lam1[0]
    else:
        lo = cand_irrr_lam1[above[-1]]  # rank > target
        hi = cand_irrr_lam1[below[below > above[-1]][0]] if np.any(below > above[-1]) else cand_irrr_lam1[below[-1]]
        for _ in range(n_bisect):
            mid = np.sqrt(lo * hi)
            if max(block_ranks(mid)) <= irrr_target_rank:
                hi = mid
            else:
                lo = mid
        irrr_lam1 = hi
    C, mu, A, irrr_details = fit_irrr(irrr_lam1)
    if max(block_ranks(irrr_lam1)) != irrr_target_rank:
        print(f"WARNING: no lam1 gives rank {irrr_target_rank} (rank jumps across it); using lam1 = {irrr_lam1:.4g} "
              f"with rank {max(block_ranks(irrr_lam1))}.")

    # rank vs lam1 over every fit (grid + bisection)
    lam1_path = np.array(sorted(irrr_fits))
    rank_path = np.array([block_ranks(lam1) for lam1 in lam1_path])  # (n_lam1, n_trial_type)
    subj_plot_dir = os.path.join(plot_dir, subject, os.path.basename(irrr_save_prefix)[len(subject) + 1:])
    os.makedirs(subj_plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for tt_i, tt in enumerate(trial_type_prefixes):
        ax.step(lam1_path, rank_path[:, tt_i], where='post', marker='o', ms=3, label=f'rank(B_{tt})')
    ax.axhline(irrr_target_rank, color='k', ls=':', lw=1)
    ax.axvline(irrr_lam1, color='r', ls='--', label=f'selected lam1 = {irrr_lam1:.3g}')
    ax.set_xscale('log')
    ax.set_xlabel('irrr_lam1')
    ax.set_ylabel('rank')
    ax.set_title(f'{subject} iRRR lam1 selection (target rank {irrr_target_rank})')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(subj_plot_dir, f'{subject}_iRRR_lam1_vs_rank.png'), dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)

    # fit with the selected irrr_lam1
    C = C * Y_scale  # back to original Y units
    mu = mu * Y_scale
    irrr_betas_all = xr.DataArray(
        C.T[:, None, :],
        dims=('parcel', 'chromo', 'regressor'),
        coords={'parcel': parcels, 'chromo': [select_chromo],
                'regressor': np.concatenate([irrr_reg_per_type[tt] for tt in trial_type_prefixes])},
    )
    irrr_betas_eeg_per_type = expand_bspline_betas(irrr_betas_all)
    irrr_stats_dict = dict(whiten_stats)
    irrr_stats_dict.update({'irrr_lam1': irrr_lam1, 'irrr_weight': irrr_weight,
                            'irrr_target_rank': irrr_target_rank,
                            'lam1_path': lam1_path, 'rank_path': rank_path,
                            'intercept': mu, 'details': irrr_details})
    for tt, A_k in zip(trial_type_prefixes, A):
        irrr_stats_dict[f'rank_{tt}'] = matrix_rank(A_k)
        irrr_stats_dict[f'singular_values_{tt}'] = svdvals(A_k * Y_scale)
    print(f"AR-iRRR fit (lam1 = {irrr_lam1:.4g}): "
          + ', '.join(f"rank(B_{tt}) = {irrr_stats_dict[f'rank_{tt}']}" for tt in trial_type_prefixes)
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
            axs[1].plot(delay_times, betas_eeg.sel(parcel=select_parcel).values.flatten(), label=f'HRF {tt} ({select_parcel})')
            axs[1].plot(delay_times, betas_eeg.sel(parcel=net_parcels).mean('parcel').values.flatten(), '--', label=f'HRF {tt} ({select_network})')
        axs[1].set_xlabel('Delay (s)')
        axs[1].set_title(f'HRF estimation using Cz EEG (AR-IRLS), mnt_correct vs mnt_incorrect')
        axs[1].legend()
        axs[1].grid()

    #%% save Y_all, dm_all and the betas / stats of each fit
    if is_save:
        os.makedirs(os.path.dirname(Y_all_save_path), exist_ok=True)
        with gzip.open(Y_all_save_path, 'wb') as f:
            pickle.dump(Y_all, f)

        os.makedirs(os.path.dirname(dm_all_save_path), exist_ok=True)
        with gzip.open(dm_all_save_path, 'wb') as f:
            pickle.dump(dm_all, f)

        # fixAR-IRLS / AR-iRRR use the same Y_all and dm_all as AR-IRLS, so neither is re-saved
        irrr_extra = {'irrr_lam1': irrr_lam1}  # lam1 selected for rank irrr_target_rank
        for prefix, betas, betas_per_type, stats, extra in [
                (save_prefix, betas_all, betas_eeg_per_type, stats_dict, {}),
                (fixar_save_prefix, fixar_betas_all, fixar_betas_eeg_per_type, fixar_stats_dict, {}),
                (irrr_save_prefix, irrr_betas_all, irrr_betas_eeg_per_type, irrr_stats_dict, irrr_extra)]:
            betas_dict = {'betas': betas, 'betas_eeg_per_type': betas_per_type,
                          'basis_da': basis_da, 't_pre': t_pre, 't_post': t_post, **extra}
            os.makedirs(os.path.dirname(get_betas_path(prefix)), exist_ok=True)
            with open(get_betas_path(prefix), 'wb') as f:
                pickle.dump(betas_dict, f)

            os.makedirs(os.path.dirname(get_stats_path(prefix)), exist_ok=True)
            with open(get_stats_path(prefix), 'wb') as f:
                pickle.dump(stats, f)
