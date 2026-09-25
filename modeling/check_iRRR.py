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
from scipy.signal import butter, sosfiltfilt, filtfilt, windows

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

#%% select model type
# eeg_reg_type = 'cont_EEG_allBandPower-bandpass'
eeg_reg_type = 'cont_EEG_cz_3-stage_iRRR'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
is_plot = False # If True, generate visualization plots
select_chromo='HbO'
# select_parcel='DorsAttnA_ParOcc_1_RH'
select_parcel='DefaultA_PFCd_1_LH'
USE_GSR=True
DO_3STAGE_REGRESSION = True # If True: (1) OLS-regress out per-run drift, (2) OLS-regress out GSR
                              # (computed from the drift-residualized signal), (3) AR-IRLS-fit only the
                              # EEG regressors on the twice-residualized signal. Overrides is_GSR_then_Others
                              # and skips adding drift/GSR back into dm_all later, since they're already removed.
cfg_GLM['do_GSR']=USE_GSR
len_delay = 15 # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)

#%% main 
subj_id = 723
subject = f"sub-{subj_id}"
print(f"Start processing {subject}")
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

# check if betas.pkl exist already. If yes, skip this subject.
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
betas_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')
stats_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_stats.pkl')
Y_all_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_Y_all.pkl.gz')
dm_all_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_dm_all.pkl.gz')
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

# RUN HRF ESTIMATION
L = 20  # <-- set this appropriately
W = windows.gaussian(L, std=L/6) / 2  
                    
if SPLIT_VTC:
    possible_trial_types = ['mnt-correct-in', 'mnt-correct-out', 'mnt-incorrect', 'city-incorrect']
else:
    possible_trial_types = ['mnt-correct', 'mnt-incorrect']    

trial_presence_list = []
stims_pruned_list = []

all_runs_tmp = []
for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    # city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
    # city_trials['trial_type'] = 'city-incorrect'
    
    if SPLIT_VTC:
        VTC = stim['VTC'].to_numpy()
        VTC = filtfilt(W, sum(W), VTC)
        median = np.median(VTC)

        in_zone = np.where(VTC <= median)[0]
        out_zone = np.where(VTC > median)[0]
        mnt_trials.loc[
                        (mnt_trials['trial_type'] == 'mnt-correct') & 
                        (mnt_trials.index.isin(in_zone)),
                        'trial_type'
                    ] = 'mnt-correct-in'

        mnt_trials.loc[
                        (mnt_trials['trial_type'] == 'mnt-correct') & 
                        (mnt_trials.index.isin(out_zone)),
                        'trial_type'
                    ] = 'mnt-correct-out'


    # Combine the filtered trials
    # stims_pruned = pd.concat(, ignore_index=True)
    # run.stim = stims_pruned
    if F_MIN > 0: 
        # TODO HP the timeseries data 
        run.time.attrs['units'] = units.s
        run_filt = frequency.freq_filter(run, 
                                    F_MIN*units.Hz, 
                                    F_MAX*units.Hz)
        all_runs_tmp.append(run_filt)
    else:
        all_runs_tmp.append(run)

    stims_pruned_list.append(mnt_trials)

# run_ts_list = [image_results['parcel_ts_weights']]
all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

all_runs_tmp = []
for run in all_runs:
    run.time.attrs['units'] = units.s
    run = run.sel(parcel = run.parcel != 'scalp')
    all_runs_tmp.append(run)
all_runs = all_runs_tmp.copy()

# mask out parcels with low forward-model sensitivity (601 -> 429 parcels)
all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels)) for run in all_runs]

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

#%% Synchronize EEG and fNIRS
# fNIRS sampling rate (all_runs' time coordinate is in seconds)
fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()
# get highpass filter frequency
# l_cutoff = np.round(1/len_delay,decimals=2)
l_cutoff = 0.02

eeg_list = []
eeg_raw_list = []
all_runs_truncated = []
fnirs_raw_list = []
run_time_windows = dict()  # run_key -> (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)
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
    eeg_t_start = nirs_t_start - t_offset - len_delay # second. extract len_delay of data prior to nirs_t_start so we don't have to reject first len_delay of fNIRS.
    eeg_t_stop = nirs_t_stop - t_offset

    # if eeg_t_start<0:
    #     raise ValueError(f"eeg_t_start <0 : {subject}")
    # if eeg_t_stop>EEG.times[-1]:
    #     raise ValueError(f"eeg_t_stop >EEG.times : {subject}")

    EEG = single_subj_EEG_dict[run_key].copy()
    EEG_raw = single_subj_EEG_dict[run_key].copy().crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG.times[-1]))

    # truncate fNIRS to the same shared event window
    fnirs_run = fnirs_run.sel(time=slice(max(nirs_t_start, fnirs_run.time.values[0]),
                                        min(nirs_t_stop, fnirs_run.time.values[-1])))
    n_fnirs_samples = len(fnirs_run.time)
    fnirs_run_raw = fnirs_run_raw.sel(time=slice(max(nirs_t_start, fnirs_run_raw.time.values[0]),
                                        min(nirs_t_stop, fnirs_run_raw.time.values[-1])))

    # reset fnirs_run.time to 0
    fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
    fnirs_run_raw = fnirs_run_raw.assign_coords(time=fnirs_run_raw.time.values - fnirs_run_raw.time.values[0])

    # remember this run's time window so it can be reapplied to an all-parcel copy later
    run_time_windows[run_key] = (run_idx, nirs_t_start, nirs_t_stop, n_fnirs_samples)

    # append data
    fnirs_raw_list.append(fnirs_run_raw)
    all_runs_truncated.append(fnirs_run)
    eeg_list.append(EEG_raw)
    eeg_raw_list.append(EEG_raw)

all_runs = all_runs_truncated

#%% 3-stage regression: OLS-regress out drift, then OLS-regress out GSR
# (computed from the drift-residualized signal), leaving only the EEG delay
# regressors for the final AR-IRLS fit later in the script
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

#%% Extract EEG values for DM
# extract EEG signal for creating DesignMatrix
eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]

# create EEG regressors
eeg_regressors = model.get_cont_EEG_regressor(eeg_reg_value_list, eeg_list[0].info['sfreq'], delay=len_delay)

#%% Low-rank representation of Delay using BSpline
# spline basis evaluated at each delay tap (not at the regressor's data values),
# so the FIR delay curve is constrained to a smooth, low-rank subspace
all_regressor_names = eeg_regressors[0].common.regressor.values
n_regressor = len(all_regressor_names)
delay_idx = np.arange(n_regressor)
bspline_basis = BSplines(delay_idx, df=[n_bspline_basis], degree=[bspline_degree],
                            include_intercept=True).basis  # (n_regressor, n_bspline_basis)
basis_da = xr.DataArray(
    bspline_basis,
    dims=("regressor", "component"),
    coords={"regressor": all_regressor_names,
            "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
)
# project the full-rank delay design matrix onto the low-rank spline basis for each eeg_regressors
for eeg_i, eeg_dm in enumerate(eeg_regressors):
    eeg_regressors[eeg_i].common = xr.dot(eeg_dm.common, basis_da, dims="regressor").rename({"component": "regressor"})

#%% Downsample EEG DM to fNIRS sampling rate
# eeg_regressors[i].common is still at EEG's native sampling rate (time = sample index).
# Resample along time from eeg_sfreq to fnirs_sfreq, then enforce an exact sample-count
# match with the corresponding (already-truncated) fNIRS run and adopt its time coordinate.
eeg_sfreq = eeg_list[0].info['sfreq']
for eeg_i, (eeg_dm, fnirs_run) in enumerate(zip(eeg_regressors, all_runs)):
    n_fnirs_samples = len(fnirs_run.time)
    # mne.filter.resample only preserves the order of non-resampled axes when
    # resampling along the last axis, so move 'time' there before resampling.
    dm_common = eeg_dm.common.transpose('regressor', 'chromo', 'time')
    dm_resampled = mne.filter.resample(dm_common.values, up=fnirs_sfreq, down=eeg_sfreq,
                                        npad='auto', axis=-1)

    # enforce exact sample-count match with the truncated fNIRS run
    n_dm_samples = dm_resampled.shape[-1]
    if n_dm_samples > n_fnirs_samples:
        dm_resampled = dm_resampled[..., :n_fnirs_samples]
        n_dm_samples = n_fnirs_samples
    elif n_dm_samples < n_fnirs_samples:
        all_runs[eeg_i] = fnirs_run.isel(time=slice(0, n_dm_samples))

    eeg_regressors[eeg_i].common = xr.DataArray(
        dm_resampled,
        dims=dm_common.dims,
        coords={**{k: v for k, v in dm_common.coords.items() if k != 'time'},
                'time': all_runs[eeg_i].time.values},
    ).transpose('time', 'regressor', 'chromo')

#%% concatenate all runs and dms
Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, eeg_regressors)

dm_all.common = dm_all.common.fillna(0)

# select HbO to match all_runs 
dm_all.common = dm_all.common.sel(chromo=[select_chromo])

#%% get GLM fitting results for each subject from shank Jun 02 2025
print(f"Start cont_EEG GLM fitting ({subject})")
results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
# extract HRF (delay-regressor betas) per parcel, then expand the low-rank
# bspline coefficients back to full per-delay resolution via the same basis
betas_all = results.sm.params.copy()
eeg_reg = [p for p in betas_all.regressor.values if 'bspline' in p]
betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])


#%% save betas for later visualization
if is_save:
    betas_dict = dict()
    betas_dict['betas'] = betas_all
    betas_dict['betas_eeg'] = betas_eeg
    betas_dict['betas_bspline'] = betas_bspline
    betas_dict['basis_da'] = basis_da
    with open(betas_save_path, 'wb') as f:
        pickle.dump(betas_dict, f)

    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats_dict, f)

    # save Y_true and design matrix in separate files (used by vis_EV_on_surface.py
    # to compute Y_hat = dm_all.common @ betas and explained variance per parcel)
    with gzip.open(Y_all_save_path, 'wb') as f:
        pickle.dump(Y_all, f)

    with gzip.open(dm_all_save_path, 'wb') as f:
        pickle.dump(dm_all, f)
