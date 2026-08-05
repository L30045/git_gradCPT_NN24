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


#%% find subjects with fNIRS and enough EEG epochs
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')
_MIN_EPOCHS = 500

_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(_project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf'))
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
model_type='cont_EEG_cz'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
select_chromo='HbO'
USE_GSR=True
cfg_GLM['do_GSR']=USE_GSR
len_delay = 12 # Delay time in HRF (sec)

#%% main 
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    #%% RUN PREPROCESSING
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

    print('LOADING PREPROCESSED CHANNEL DATA')
    with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}.pkl'), 'rb') as f:
        results = pickle.load(f)

    # all_runs = results['runs']
    all_chs_pruned = results['chs_pruned']
    all_stims = results['stims']
    geo3d = results['geo3d']

    print('LOADING IMAGE SPACE RESULTS')
    folder =  os.path.join(der_dir, subject)
    filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}.pkl'

    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)

    if weight_flag == 'aca':
        all_runs = image_results['parcel_ts_aca']
        vv = image_results['vertex_aca']
    elif weight_flag == 'post': 
        all_runs = image_results['parcel_ts_post']
        vv = image_results['vertex_mse']
    else: 
        all_runs = image_results['parcel_ts_none']
        vv = image_results['vertex_mse']

    n_runs = len(vv)
    vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
    vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

    if cfg_GLM['do_GSR']: 
        aca_lst = image_results['vertex_aca']
        aca_p_lst = []
        for aca in aca_lst:
            aca_p = aca.groupby('parcel').sum('vertex') / aca.groupby('parcel').count()**2
            aca_p = aca_p.sel(parcel = aca_p.parcel != 'scalp')
            aca_p_lst.append(aca_p)

        cfg_GLM['GSR_weight'] = aca_p_lst


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


    #%% get continous EEG
    eeg_der_dir = os.path.join(project_path, "derivatives", "eeg")
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    # check if Cz exists
    cz_removed = any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]]
                    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3'])
    if cz_removed:
        raise ValueError(f"sub-{subj_id}: Cz was removed in at least one run, skipping subject.")

    # match each fNIRS run in all_runs to its EEG run (gradcpt1/2/3) via first stim onset in events.tsv
    eeg_ev_files = {
        run_key: os.path.join(eeg_der_dir, subject, f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv")
        for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']
    }
    eeg_ev_dfs = {run_key: pd.read_csv(f, sep='\t') for run_key, f in eeg_ev_files.items()}

    nirs_ev_files = sorted(glob.glob(os.path.join(project_path, subject, 'nirs', f"{subject}_task-gradCPT_run-*_events.tsv")))
    nirs_ev_dfs = {f: pd.read_csv(f, sep='\t') for f in nirs_ev_files}

    matched_nirs_file = dict()
    for run_key, ev_df in eeg_ev_dfs.items():
        eeg_onset0 = ev_df['onset'].values[0]
        for nirs_file, nirs_df in nirs_ev_dfs.items():
            if len(nirs_df) == len(ev_df) and np.allclose(nirs_df['onset'].values - nirs_df['onset'].values[0],
                                                            ev_df['onset'].values - eeg_onset0, atol=0.05):
                matched_nirs_file[run_key] = nirs_file
                break

    run_key_to_run_idx = dict()
    for run_key, nirs_file in matched_nirs_file.items():
        nirs_onset0 = nirs_ev_dfs[nirs_file]['onset'].values[0]
        for r_i, stim in enumerate(all_stims):
            if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onset0, atol=0.05):
                run_key_to_run_idx[run_key] = r_i
                break

    #%% lowpass and resample EEG
    # fNIRS sampling rate (all_runs' time coordinate is in seconds)
    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

    eeg_list = []
    all_runs_truncated = []
    run_time_windows = dict()  # run_key -> (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)
    for run_key in ['gradcpt1', 'gradcpt2', 'gradcpt3']:
        run_idx = run_key_to_run_idx[run_key]
        fnirs_run = all_runs[run_idx]

        # EEG <-> fNIRS clock offset for this run (nirs_time = eeg_time + t_offset)
        eeg_ev_df = eeg_ev_dfs[run_key]
        nirs_ev_df = nirs_ev_dfs[matched_nirs_file[run_key]]
        eeg_onset0 = eeg_ev_df['onset'].values[0]
        nirs_onset0 = nirs_ev_df['onset'].values[0]
        t_offset = nirs_onset0 - eeg_onset0

        # window from the first event onset to the last event's end (onset + duration), in fNIRS time
        nirs_t_start = nirs_ev_df['onset'].values[0]
        # nirs_t_stop = (nirs_ev_df['onset'] + nirs_ev_df['duration']).values[-1]
        nirs_t_stop = (nirs_ev_df['onset']).values[-1]+len_delay # second
        eeg_t_start = nirs_t_start - t_offset
        eeg_t_stop = nirs_t_stop - t_offset

        EEG = single_subj_EEG_dict[run_key].copy()
        EEG_raw = single_subj_EEG_dict[run_key].copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))

        # lowpass EEG to fNIRS sampling rate/2, with -3dB cutoff at h_freq (tight transition band)
        cutoff = fnirs_sfreq / 2
        EEG_filter = EEG.filter(l_freq=None, h_freq=cutoff, h_trans_bandwidth=0.25, picks='cz').copy()

        # truncate EEG to the shared event window (clamped to the recording's own bounds)
        EEG_filter.crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG_filter.times[-1]))

        # truncate fNIRS to the same shared event window
        fnirs_run = fnirs_run.sel(time=slice(max(nirs_t_start, fnirs_run.time.values[0]),
                                            min(nirs_t_stop, fnirs_run.time.values[-1])))
        n_fnirs_samples = len(fnirs_run.time)
        

        # downsample EEG to match the number of sample points in fNIRS
        EEG_resample = EEG_filter.copy()
        EEG_resample.resample(fnirs_sfreq, npad='auto')

        # enforce exact sample-count match with the truncated fNIRS run
        if EEG_resample.n_times > n_fnirs_samples:
            EEG_resample.crop(tmax=EEG_resample.times[n_fnirs_samples - 1])
        elif EEG_resample.n_times < n_fnirs_samples:
            fnirs_run = fnirs_run.isel(time=slice(0, EEG_resample.n_times))
            n_fnirs_samples = EEG_resample.n_times

        # truncate fNIRS so the delay at the beginning of the recording is removed
        fnirs_run = fnirs_run.isel(time=slice(np.round(len_delay*fnirs_sfreq).astype(int), n_fnirs_samples))

        # reset fnirs_run.time to 0
        fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])

        # remember this run's time window so it can be reapplied to an all-parcel copy later
        run_time_windows[run_key] = (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)

        # append data
        all_runs_truncated.append(fnirs_run)
        eeg_list.append(EEG_resample)

    all_runs = all_runs_truncated


    #%% create EEG DM
    # extract EEG signal for creating DesignMatrix
    eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]
    # create EEG regressors
    eeg_regressors = model.get_cont_EEG_regressor(eeg_reg_value_list, fnirs_sfreq, delay=len_delay)
    # concatenate all runs and dms
    Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, eeg_regressors)
    # Add GSR
    if USE_GSR:
        gs_regressors = model.get_global_mean_regressor(all_runs, weights=cfg_GLM['GSR_weight'])
        _, gs_dm, _ = model.concatenate_runs_dms(all_runs, gs_regressors)
        # Merge gsr and eeg_regressors
        dm_all = model.combine_dm(dm_all, gs_dm)
    # select HbO to fasten training process
    dm_all.common = dm_all.common.sel(chromo=[select_chromo])

    #%% get GLM fitting results for each subject from shank Jun 02 2025
    print(f"Start cont_EEG GLM fitting ({subject})")
    results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model']) 
    # extract HRF (delay-regressor betas) per parcel
    betas = results.sm.params
    eeg_reg = [p for p in betas.regressor.values if p.startswith('delay')]
    betas = betas.sel(regressor=eeg_reg)

    # save betas for later visualization
    if is_save:
        betas_save_path = os.path.join(data_save_path, f'{subject}_{model_type}_betas.pkl')
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas, f)
