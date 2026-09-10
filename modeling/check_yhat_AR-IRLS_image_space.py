#%% Compare y_hat from CORRECT_CODE_runGLM_on_image.py (event-based HRF GLM)
# against y_hat from run_model_cont_EEG_fNIRS.py (continuous-EEG GLM),
# for the same subject and parcel. Both fits include GSR and drift (Legendre)
# regressors via the shared cfg_GLM from params_setting.py.
import os
import gzip
import pickle
import sys
import glob
import re

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cedalion
from cedalion import units, nirs
from cedalion.sigproc import frequency
import cedalion.models.glm as glm
from scipy.signal import filtfilt, windows

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf

from params_setting import *

import warnings
warnings.filterwarnings('ignore')

#%% subject / parcel selection (shared between both fits)
subject = 'sub-723'
select_parcel = 'SalVentAttnA_FrMed_5_LH'
select_chromo = 'HbO'
eeg_reg_type = 'cont_EEG_cz_add_15s'
is_hp_fNIRS = False
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
len_delay = 15  # must match run_model_cont_EEG_fNIRS.py's len_delay, used for the crop window

assert USE_GSR and cfg_GLM['do_GSR'], "USE_GSR must be True to compare GSR-included fits"
assert cfg_GLM['do_drift_legendre'] or cfg_GLM['do_drift'], "drift regressors must be enabled"

#%% ---- Fit 1: CORRECT_CODE_runGLM_on_image.py (event-based HRF GLM on MNT trials) ----
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

print('LOADING PREPROCESSED CHANNEL DATA')
with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    results_chs = pickle.load(f)

all_chs_pruned = results_chs['chs_pruned']
all_stims = results_chs['stims']
geo3d = results_chs['geo3d']

print('LOADING IMAGE SPACE RESULTS')
folder = os.path.join(der_dir, subject)
filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'

with open(filepath, 'rb') as f:
    image_results = pickle.load(f)

all_runs = image_results['parcel_ts']

L = 20
W = windows.gaussian(L, std=L/6) / 2

if SPLIT_VTC:
    possible_trial_types = ['mnt-correct-in', 'mnt-correct-out', 'mnt-incorrect', 'city-incorrect']
else:
    possible_trial_types = ['mnt-correct', 'mnt-incorrect']

stims_pruned_list = []
all_runs_tmp = []
for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    if SPLIT_VTC:
        VTC = stim['VTC'].to_numpy()
        VTC = filtfilt(W, sum(W), VTC)
        median = np.median(VTC)
        in_zone = np.where(VTC <= median)[0]
        out_zone = np.where(VTC > median)[0]
        mnt_trials.loc[
            (mnt_trials['trial_type'] == 'mnt-correct') & (mnt_trials.index.isin(in_zone)),
            'trial_type'
        ] = 'mnt-correct-in'
        mnt_trials.loc[
            (mnt_trials['trial_type'] == 'mnt-correct') & (mnt_trials.index.isin(out_zone)),
            'trial_type'
        ] = 'mnt-correct-out'

    if F_MIN > 0:
        run.time.attrs['units'] = units.s
        run_filt = frequency.freq_filter(run, F_MIN*units.Hz, F_MAX*units.Hz)
        all_runs_tmp.append(run_filt)
    else:
        all_runs_tmp.append(run)

    stims_pruned_list.append(mnt_trials)

all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs_tmp]

all_runs_tmp = []
for run in all_runs:
    run.time.attrs['units'] = units.s
    run = run.sel(parcel=run.parcel != 'scalp')
    all_runs_tmp.append(run)
all_runs = all_runs_tmp.copy()

# keep an all-parcel copy (chromo-selected only) for computing GSR from all parcels,
# separate from the single-parcel copy used as the GLM fit target
all_runs_allparcel = [x.sel(chromo=[select_chromo]) for x in all_runs]
all_runs = [x.sel(parcel=[select_parcel], chromo=[select_chromo]) for x in all_runs]

# reorder all_runs to match the event order (gradcpt1, gradcpt2, gradcpt3)
nirs_ev_files = sorted(glob.glob(os.path.join(root_dir, subject, 'nirs', f"{subject}_task-gradCPT_run-*_events.tsv")))
nirs_ev_dfs = {f: pd.read_csv(f, sep='\t') for f in nirs_ev_files}

run_key_to_run_idx = dict()
run_key_to_nirs_df = dict()
for run_num, (nirs_file, nirs_df) in enumerate(nirs_ev_dfs.items(), start=1):
    nirs_onset0 = nirs_df['onset'].values[0]
    for r_i, stim in enumerate(all_stims):
        if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onset0, atol=0.01):
            run_key_to_run_idx[f'gradcpt{run_num}'] = r_i
            run_key_to_nirs_df[f'gradcpt{run_num}'] = nirs_df
            break

assert len(run_key_to_run_idx) == len(all_runs), "could not match all runs to a gradcpt run key"
run_keys = [f'gradcpt{i}' for i in range(1, len(all_runs) + 1)]
reorder_idx = [run_key_to_run_idx[k] for k in run_keys]
all_runs = [all_runs[i] for i in reorder_idx]
all_runs_allparcel = [all_runs_allparcel[i] for i in reorder_idx]

# crop each run to [first_stim_onset, last_stim_onset + len_delay], the same window
# run_model_cont_EEG_fNIRS.py crops fNIRS to (its exact final length also depends on
# EEG resampling, which this script does not replicate, so lengths may be off by ~1
# sample per run), then reset each run's time to start at 0
cropped_runs = []
cropped_runs_allparcel = []
for run_key, run, run_ap in zip(run_keys, all_runs, all_runs_allparcel):
    nirs_df = run_key_to_nirs_df[run_key]
    nirs_t_start = nirs_df['onset'].values[0]
    nirs_t_stop = nirs_df['onset'].values[-1] + len_delay

    run_c = run.sel(time=slice(max(nirs_t_start, run.time.values[0]),
                                min(nirs_t_stop, run.time.values[-1])))
    run_c = run_c.assign_coords(time=run_c.time.values - run_c.time.values[0])
    run_c.time.attrs['units'] = units.s
    cropped_runs.append(run_c)

    run_ap_c = run_ap.sel(time=slice(max(nirs_t_start, run_ap.time.values[0]),
                                      min(nirs_t_stop, run_ap.time.values[-1])))
    run_ap_c = run_ap_c.assign_coords(time=run_ap_c.time.values - run_ap_c.time.values[0])
    run_ap_c.time.attrs['units'] = units.s
    cropped_runs_allparcel.append(run_ap_c)

all_runs = cropped_runs
all_runs_allparcel = cropped_runs_allparcel
stims_pruned_list = [stims_pruned_list[i] for i in reorder_idx]

# compute GSR from all parcels (not just select_parcel), time-shifted the same way
# pf.concatenate_runs shifts time, so the regressor lines up with Y_all's time axis
gsr_per_run = pf.get_global_mean_regressor(all_runs_allparcel)
CURRENT_OFFSET = 0
gsr_shifted = []
for run_ap, gsr in zip(all_runs_allparcel, gsr_per_run):
    time = run_ap.time.values
    new_time = time + CURRENT_OFFSET
    gsr_common = gsr.common.assign_coords(time=new_time)
    gsr_shifted.append(gsr_common)
    CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])
gsr_allparcel = xr.concat(gsr_shifted, dim='time')

cfg_GLM_noGSR = dict(cfg_GLM)
cfg_GLM_noGSR['do_GSR'] = False

print(f"Start event-based HRF GLM fitting ({subject}, do_GSR=True (all-parcel GS), "
      f"do_drift_legendre={cfg_GLM['do_drift_legendre']})")
glm_results_event, hrf_estimate, hrf_mse, dms_event = pf.GLM(
    all_runs, cfg_GLM_noGSR, geo3d, all_chs_pruned, stims_pruned_list, regressors=gsr_allparcel)
Y_all_event, stim_df, runs_updated = pf.concatenate_runs(all_runs, stims_pruned_list)

betas_event = glm_results_event.sm.params
y_hat_event = (dms_event.common * betas_event).sum('regressor')
y_hat_event = y_hat_event.transpose('chromo', 'parcel', 'time')

y_true_event = Y_all_event.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
y_hat_event_vals = y_hat_event.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
t_event = Y_all_event.time.values

#%% ---- Fit 2: run_model_cont_EEG_fNIRS.py (continuous-EEG GLM) ----
# load Y_all + dm_all saved by run_model_cont_EEG_fNIRS.py for this subject, then
# retrain the GLM on just select_parcel (rather than reusing the all-parcel saved betas)
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
base = os.path.join(eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')

Y_all_path = base + '_Y_all.pkl.gz'
dm_all_path = base + '_dm_all.pkl.gz'

for p in (Y_all_path, dm_all_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Rerun run_model_cont_EEG_fNIRS.py with is_overwrite=True for {subject}.")

with gzip.open(Y_all_path, 'rb') as f:
    Y_all_cont = pickle.load(f)  # dims: chromo, parcel, time

with gzip.open(dm_all_path, 'rb') as f:
    dm_all_cont = pickle.load(f)  # .common dims: time, chromo, regressor

Y_all_cont_parcel = Y_all_cont.sel(parcel=[select_parcel])
glm_results_cont = glm.fit(Y_all_cont_parcel, dm_all_cont, noise_model=cfg_GLM['noise_model'])
betas_cont = glm_results_cont.sm.params  # dims: parcel, chromo, regressor

y_hat_cont = xr.dot(dm_all_cont.common, betas_cont, dims='regressor')  # dims: time, chromo, parcel

y_true_cont = Y_all_cont.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
y_hat_cont_vals = y_hat_cont.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
t_cont = Y_all_cont.time.values

#%% ---- visualize both fits together ----
fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=False)

axs[0].plot(t_event, y_true_event, label='Y (true, event-based)', color='k', linewidth=2)
axs[0].plot(t_event, y_hat_event_vals, 'b', label='y_hat (event-based HRF GLM)', alpha=0.6)
axs[0].plot(t_cont, y_hat_cont_vals, 'r', label='y_hat (continuous EEG GLM)', alpha=0.6)
axs[0].set_ylabel('HbO concentration')
axs[0].set_title(f'{subject}: y_hat comparison ({select_parcel}), GSR=all-parcel, '
                  f'drift_legendre={cfg_GLM["do_drift_legendre"]}')
axs[0].legend()
axs[0].grid()

axs[1].plot(t_event, y_true_event - y_hat_event_vals, 'b', label='Resid (event-based)', alpha=0.7)
axs[1].plot(t_cont, y_true_cont - y_hat_cont_vals, 'r', label='Resid (continuous EEG)', alpha=0.7)
axs[1].set_xlabel('Time (s)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# %%
