#%% load library
import numpy as np
import pickle
import gzip
import glob
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
from scipy.signal import butter, sosfiltfilt

#%% select model type
model_type = 'onlyStim' # 'full', 'onlyStim', 'onlyEEG' -> key of the saved DM (see below)
is_overwrite = False # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
select_chromo = 'HbO'
USE_GSR = True
DO_3STAGE_REGRESSION = True # Same as run_model_cont_EEG_fNIRS.py: OLS-regress out per-run drift, then GSR,
                            # before the AR-IRLS fit. Required: the DM below has no drift / GSR columns.
cfg_GLM['do_GSR'] = USE_GSR
len_delay = 15 # Delay time in HRF (sec); fNIRS window ends len_delay after the last event (same as cont EEG pipeline)
l_cutoff = 0.02 # highpass cutoff (Hz)

#%% design matrix
# Built by build_dm_gaussian-kernel_mnt-correct_EEG-informed.py and saved to
# <data_save_path>/dm/gaussian-kernel_mnt-correct_EEG-informed_Cz-P3-ptp-Woody.pkl (get_gauss_eeg_dm_path):
# 20 Gaussian kernels (-2 to 18 s) for mnt-correct trials, 'HRF mnt-correct-stim XX' (unmodulated) and
# 'HRF mnt-correct-eeg XX' (scaled by the single-trial Cz Woody P3 peak amplitude), on the rows of the
# shared Y_all. Keys: 'full' (eeg then stim), 'onlyStim', 'onlyEEG'. The kernel settings (cfg_HRF_basis)
# are read from the same file.
EEG_AMP_TAG = 'woodyN2P3ptp'         # added to the saved file names of full / onlyEEG models
MODEL_NAME = 'gaussian-kernel_mnt-correct'  # in the saved file names

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
# only needed when Y_all has to be rebuilt, so compute lazily
_sensitive_parcels = None
def get_sensitive_parcels():
    global _sensitive_parcels
    if _sensitive_parcels is None:
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
        _sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (429 of 601)
    return _sensitive_parcels

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
assert not any(f'sub-{x}' in excluded_subj for x in subj_id_array), 'excluded subjects left in subj_id_array'

#%% fNIRS preprocessing (same as run_model_cont_EEG_fNIRS.py)
def preprocess_fnirs(subject):
    """Rebuild the concatenated, preprocessed parcel time series (Y_all) the same way as
    run_model_cont_EEG_fNIRS.py: sensitive parcels, HbO, highpass, per-run event window
    (first event -> last event + len_delay, clamped to the fNIRS run and to the span whose full
    len_delay EEG history is recorded), then OLS-regress out drift and GSR.
    Only used when the cont EEG pipeline's shared Y_all file does not exist.
    """
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

    print('LOADING PREPROCESSED CHANNEL DATA')
    with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        results = pickle.load(f)
    all_stims = results['stims']

    print('LOADING IMAGE SPACE RESULTS')
    filepath = os.path.join(der_dir, subject, f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl')
    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)
    all_runs = image_results['parcel_ts']

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]
    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel=run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp

    # mask out parcels with low forward-model sensitivity (601 -> 429 parcels)
    sensitive_parcels = get_sensitive_parcels()
    all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels)) for run in all_runs]

    if select_chromo is not None:
        all_runs = [x.sel(chromo=[select_chromo]) for x in all_runs]

    # match each fNIRS run (all_runs order) to gradcpt1/2/3 via the first stim onset in events.tsv
    nirs_ev_dfs = dict()
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_num = f"{run_key[-1]:0>2}"
        nirs_ev_dfs[run_key] = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                                        f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
    run_key_to_run_idx = dict()
    for run_key, nirs_df in nirs_ev_dfs.items():
        for r_i, stim in enumerate(all_stims):
            if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_df['onset'].values[0], atol=0.01):
                run_key_to_run_idx[run_key] = r_i
                break

    # EEG <-> fNIRS clock offset per run (nirs_time = eeg_time + t_offset) and the EEG time span, from the
    # same preprocessed .fif / events.tsv the cont EEG pipeline loads (utils.eeg_preproc_subj_level)
    eeg_preproc_dir = os.path.join(project_path, 'derivatives', 'eeg', subject, 'preprocessed_EEG_and_events')
    eeg_t_span = dict()
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_num = f"{run_key[-1]:0>2}"
        eeg_ev_df = pd.read_csv(os.path.join(eeg_preproc_dir, f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
        eeg_raw = mne.io.read_raw_fif(os.path.join(eeg_preproc_dir, f"{subject}_task-gradCPT_run-{run_num}_preproc_eeg.fif"),
                                      preload=False, verbose=False)
        t_offset = nirs_ev_dfs[run_key]['onset'].values[0] - eeg_ev_df['onset'].values[0]
        eeg_t_span[run_key] = (eeg_raw.times[0] + t_offset, eeg_raw.times[-1] + t_offset)

    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

    all_runs_truncated = []
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_idx = run_key_to_run_idx[run_key]
        fnirs_run = all_runs[run_idx].copy()

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

        # window from the first event onset to len_delay after the last event onset, in fNIRS time, clamped
        # to the fNIRS run and to the span whose full len_delay EEG history is recorded (as in the cont pipeline)
        nirs_ev_df = nirs_ev_dfs[run_key]
        nirs_t_start = max(nirs_ev_df['onset'].values[0], fnirs_run.time.values[0],
                           eeg_t_span[run_key][0] + len_delay)
        nirs_t_stop = min(nirs_ev_df['onset'].values[-1] + len_delay, fnirs_run.time.values[-1],
                          eeg_t_span[run_key][1])
        fnirs_run = fnirs_run.sel(time=slice(nirs_t_start, nirs_t_stop))

        # reset fnirs_run.time to 0
        fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
        all_runs_truncated.append(fnirs_run)
    all_runs = all_runs_truncated

    # 3-stage regression: OLS-regress out drift, then OLS-regress out GSR
    if DO_3STAGE_REGRESSION:
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

    # concatenate runs (gradcpt1, 2, 3 order)
    Y_all, _, _ = model.concatenate_runs(all_runs, list(nirs_ev_dfs.values()))
    return Y_all

#%% main
for subj_id in tqdm(subj_id_array):
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
    use_eeg_amp = model_type.startswith('full') or model_type.startswith('onlyEEG')
    model_tag = f'{model_type}-{EEG_AMP_TAG}' if use_eeg_amp else model_type
    save_prefix = os.path.join(data_save_path, f'{subject}_event-based_onParcel_{MODEL_NAME}_{model_tag}_{NOISE_MODEL}_{hp_flag}')
    betas_save_path = get_betas_path(save_prefix)
    stats_save_path = get_stats_path(save_prefix)
    dm_all_save_path = get_dm_all_path(save_prefix)
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    dm_path = get_gauss_eeg_dm_path(data_save_path)
    if not os.path.exists(dm_path):
        print(f"{subject}: {os.path.basename(dm_path)} not found (run build_dm_gaussian-kernel_mnt-correct_EEG-informed.py), skipping.")
        continue
    with open(dm_path, 'rb') as f:
        dm_saved = pickle.load(f)
    dm_key = next((k for k in ['full', 'onlyStim', 'onlyEEG'] if model_type.startswith(k)), None)
    if dm_key not in dm_saved['dm']:
        print(f"{subject}: no '{dm_key}' DM in {os.path.basename(dm_path)} (EEG features missing?), skipping.")
        continue
    if not DO_3STAGE_REGRESSION:
        raise ValueError("the saved DM has no drift / GSR columns; set DO_3STAGE_REGRESSION = True")
    dm_all = dm_saved['dm'][dm_key]
    cfg_HRF_basis = {k: v * units.s for k, v in dm_saved['cfg_HRF_basis'].items()}
    del dm_saved

    #%% load Y_all from the cont EEG pipeline if available; otherwise rebuild it
    cont_Y_all_path = get_shared_Y_all_path(data_save_path, subject, hp_flag)
    if os.path.exists(cont_Y_all_path):
        print(f"LOADING Y_all FROM {os.path.basename(cont_Y_all_path)}")
        with gzip.open(cont_Y_all_path, 'rb') as f:
            Y_all = pickle.load(f)
    else:
        print(f"{os.path.basename(cont_Y_all_path)} not found, RUNNING fNIRS PREPROCESSING")
        Y_all = preprocess_fnirs(subject)
    Y_all = Y_all.sel(chromo=[select_chromo])

    #%% check that the DM rows are Y_all's samples (per-run fNIRS sample index, resets at run starts)
    if not np.array_equal(dm_all.common.samples.values, Y_all.samples.values):
        raise ValueError(f"{subject}: DM rows do not match Y_all samples; rebuild the DM.")
    dm_all.common = dm_all.common.drop_vars('samples').assign_coords(time=Y_all.time.values)
    dm_all.common = dm_all.common.fillna(0)

    #%% select HbO to fasten training process
    dm_all.common = dm_all.common.sel(chromo=[select_chromo])

    #%% get GLM fitting results
    print(f"Start EEG-informed GLM fitting ({subject})")
    glm_results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    betas_all = glm_results.sm.params.copy()
    cov_params = glm_results.sm.cov_params()

    #%% f test
    stats_dict = dict()
    if model_type.startswith('full'):
        # full vs stim
        param_names = [name for name in betas_all.regressor.values if 'eeg' in name]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict['f_test_full_stim'] = glm_results.sm.f_test(hypotheses)
        # full vs basis
        param_names = [name for name in betas_all.regressor.values if ('eeg' in name) or ('stim' in name)]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict['f_test_full_basis'] = glm_results.sm.f_test(hypotheses)
        # full vs eeg
        param_names = [name for name in betas_all.regressor.values if 'stim' in name]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict['f_test_full_eeg'] = glm_results.sm.f_test(hypotheses)
    elif model_type.startswith('onlyStim'):
        param_names = [name for name in betas_all.regressor.values if 'stim' in name]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict['f_test_stim_basis'] = glm_results.sm.f_test(hypotheses)
    elif model_type.startswith('onlyEEG'):
        param_names = [name for name in betas_all.regressor.values if 'eeg' in name]
        hypotheses = [f'{name} = 0' for name in param_names]
        stats_dict['f_test_eeg_basis'] = glm_results.sm.f_test(hypotheses)

    #%% contrast t test
    if model_type.startswith('full'):
        param_names = [name for name in betas_all.regressor.values if 'eeg' in name]
        stats_dict['t_test_0_eeg'] = glm_results.sm.t_test('+'.join(param_names)+' = 0')
        param_names = [name for name in betas_all.regressor.values if ('eeg' in name) or ('stim' in name)]
        stats_dict['t_test_0_eeg_stim'] = glm_results.sm.t_test('+'.join(param_names)+' = 0')
        param_names = [name for name in betas_all.regressor.values if 'stim' in name]
        stats_dict['t_test_0_stim'] = glm_results.sm.t_test('+'.join(param_names)+' = 0')
    elif model_type.startswith('onlyStim'):
        param_names = [name for name in betas_all.regressor.values if 'stim' in name]
        stats_dict['t_test_0_stim'] = glm_results.sm.t_test('+'.join(param_names)+' = 0')
    elif model_type.startswith('onlyEEG'):
        param_names = [name for name in betas_all.regressor.values if 'eeg' in name]
        stats_dict['t_test_0_eeg'] = glm_results.sm.t_test('+'.join(param_names)+' = 0')

    #%% expand HRF betas to HRF time courses via the Gaussian kernel basis
    betas_dict = dict()
    betas_dict['betas'] = betas_all
    betas_dict['cov_params'] = cov_params
    if not model_type.startswith('basis'):
        trial_type_list = ['mnt-correct']
        run_unit = Y_all.pint.units
        basis_hrf = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'], cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])(Y_all)
        n_hrf_reg = int(betas_all.regressor.str.startswith('HRF mnt-correct-').sum()) // (2 if model_type.startswith('full') else 1)
        if basis_hrf.sizes['component'] != n_hrf_reg:
            raise ValueError(f"cfg_HRF_basis gives {basis_hrf.sizes['component']} kernels but the DM has {n_hrf_reg} per trial type.")
        if model_type.startswith('full'):
            # 'HRF mnt-correct' matches both the -eeg and -stim regressors (eeg first, then stim)
            basis_hrf = xr.concat([basis_hrf, basis_hrf], dim='component')

        hrf_mse_list = []
        hrf_estimate_list = []
        for trial_type in trial_type_list:
            betas_hrf = betas_all.sel(regressor=betas_all.regressor.str.startswith(f"HRF {trial_type}"))
            hrf_estimate = model.estimate_HRF_from_beta(betas_hrf, basis_hrf)

            cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                                     regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}"))
            hrf_mse = model.estimate_HRF_cov(cov_hrf, basis_hrf)

            hrf_estimate_list.append(hrf_estimate.expand_dims({'trial_type': [trial_type]}))
            hrf_mse_list.append(hrf_mse.expand_dims({'trial_type': [trial_type]}))

        hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type').pint.quantify(run_unit)
        hrf_mse = xr.concat(hrf_mse_list, dim='trial_type').pint.quantify(run_unit**2)

        # set universal time so that all hrfs have the same time base
        fs = model.frequency.sampling_rate(Y_all).to('Hz')
        before_samples = int(np.ceil((cfg_HRF_basis['t_pre'] * fs).magnitude))
        after_samples = int(np.ceil((cfg_HRF_basis['t_post'] * fs).magnitude))
        dT = np.round(1 / fs, 3)  # millisecond precision
        reltime = np.linspace(-before_samples * dT, after_samples * dT, len(hrf_estimate.time))

        hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
        hrf_estimate.time.attrs['units'] = 'second'
        hrf_mse = hrf_mse.assign_coords({'time': reltime})
        hrf_mse.time.attrs['units'] = 'second'

        betas_dict['hrf_estimate'] = hrf_estimate
        betas_dict['hrf_mse'] = hrf_mse
        betas_dict['basis_da'] = basis_hrf

    #%% save betas for later visualization
    if is_save:
        os.makedirs(os.path.dirname(betas_save_path), exist_ok=True)
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)

        os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)

        with open(stats_save_path, 'wb') as f:
            pickle.dump(stats_dict, f)

        # save the DM truncated to Y_all's time points (Y_all itself is the cont EEG pipeline's
        # parcel_Y_all_truncated_to_trials_{hp_flag}.pkl.gz, so it is not saved again here)
        os.makedirs(os.path.dirname(dm_all_save_path), exist_ok=True)
        with gzip.open(dm_all_save_path, 'wb') as f:
            pickle.dump(dm_all, f)

print("All trainings completed.")
