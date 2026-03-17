#%% load library
import numpy as np
import scipy as sp
import pickle
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from neon_to_bids_gradCPT_add_nodding import parseNeon_to_bids
from pupil_labs import neon_recording as nr
import re
from cedalion import io
from utils_eyetracking import preprocess_pupil, get_pupil_epoch, plot_pupil_epoch
import warnings

#%% cross subjects epoch analysis
dirs = os.listdir(project_path)
subject_list = [d for d in dirs if 'sub' in d] # and d not in excluded]

epoch_length = 1.6 # sec (fade-in + fade-out)
baseline_length = 0.2 # sec (previous event's fade-out phase)
pupil_dict = {}
for subj in sorted(subject_list):
    subj_id = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    if not os.path.isdir(subj_nirs) or not os.path.isdir(subj_neon_dir):
        continue
    snirf_files = sorted([f for f in os.listdir(subj_nirs) if f.endswith('.snirf')])
    neon_dirs_subj = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])
    pupil_dict[subj] = {}
    for run_id in range(1, 4):
        snirf_name = f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_nirs.snirf"
        physio_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
        event_file  = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if snirf_name not in snirf_files or not os.path.isfile(physio_file) or not os.path.isfile(event_file):
            continue
        neon_idx = snirf_files.index(snirf_name)
        neon_data = pd.read_csv(physio_file, sep='\t')
        rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[neon_idx]))
        t_neon_s, pupil_d_s = preprocess_pupil(neon_data, rec)
        sfreq_neon = np.round(np.median(1 / np.diff(t_neon_s)))
        print(f"{subj} run-{run_id:02d}: sfreq_neon = {sfreq_neon} Hz")
        events_df_s = pd.read_csv(event_file, sep='\t')
        mnt_correct_idx_s   = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']==0)
        mnt_incorrect_idx_s = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']!=0)
        city_correct_idx_s  = (events_df_s['trial_type']=='city') & (events_df_s['response_code']!=0)
        city_incorrect_idx_s= (events_df_s['trial_type']=='city') & (events_df_s['response_code']==0)
        vtc_s = utils.smoothing_VTC_gaussian_array(events_df_s['VTC'].values, L=20)
        win_epoch    = int(np.round(sfreq_neon * epoch_length))
        win_baseline = int(np.round(sfreq_neon * baseline_length))
        t_epoch_run  = np.linspace(-baseline_length, epoch_length, win_baseline + win_epoch)
        pupil_dict[subj][f'run-{run_id:02d}'] = {
            'mnt_correct':        get_pupil_epoch(events_df_s, mnt_correct_idx_s,   t_neon_s, pupil_d_s),
            'mnt_incorrect':      get_pupil_epoch(events_df_s, mnt_incorrect_idx_s, t_neon_s, pupil_d_s),
            'city_correct':       get_pupil_epoch(events_df_s, city_correct_idx_s,  t_neon_s, pupil_d_s),
            'city_incorrect':     get_pupil_epoch(events_df_s, city_incorrect_idx_s,t_neon_s, pupil_d_s),
            'mnt_correct_vtc':    vtc_s[mnt_correct_idx_s.values],
            'mnt_incorrect_vtc':  vtc_s[mnt_incorrect_idx_s.values],
            'city_correct_vtc':   vtc_s[city_correct_idx_s.values],
            'city_incorrect_vtc': vtc_s[city_incorrect_idx_s.values],
            't_epoch':            t_epoch_run,
            'sfreq_neon':         sfreq_neon,
        }
pupil_dict = {subj: runs for subj, runs in pupil_dict.items() if runs}

#%% Plot Cross-subject results
epoch_length = 1.6 # sec (fade-in + fade-out)
baseline_length = 0.2 # sec (previous event's fade-out phase)
conditions = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']

# For each subject, pool epochs across runs then compute the per-subject mean time course
subj_mean = {cond: [] for cond in conditions}
for subj, runs in pupil_dict.items():
    for cond in conditions:
        # concatenate all trials across runs with per-epoch baseline correction
        all_epochs = []
        for run_data in runs.values():
            if run_data['sfreq_neon'] != 125:
                continue
            for ep in run_data[cond]:
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                all_epochs.append(ep - baseline_val)
        if len(all_epochs) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                epochs_arr = np.array(all_epochs)
                subj_mean[cond].append(np.nanmean(epochs_arr, axis=0))
        except Warning as w:
            print(f"Skipping {subj} {cond}: {w}")
            continue

# Plot cross-subject mean ± SEM
t_epoch = np.linspace(-baseline_length, epoch_length,
                      len(next(v for v in subj_mean.values() if v)[0]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(0, color='k', linestyle='--', label='Onset')
for cond, label in [
    ('mnt_correct',   'mnt — Correct'),
    ('mnt_incorrect', 'mnt — Incorrect'),
    ('city_correct',  'city — Correct'),
    ('city_incorrect','city — Incorrect'),
]:
    arr = np.array(subj_mean[cond])  # shape: (n_subjects, n_samples)
    if len(arr) == 0:
        continue
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0) / np.sqrt(len(arr))
    line, = ax.plot(t_epoch, mean, label=label)
    ax.fill_between(t_epoch, mean - sem, mean + sem, alpha=0.3,
                    color=line.get_color(), label='_nolegend_')
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title('Pupil epoch (cross-subject)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pupil diameter (mm)')
ax.legend()
ax.grid()
plt.tight_layout()

#%% Add VTC as another dimension
# Median split per subject: solid line = high VTC, dashed = low VTC
conditions = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']

# Per subject: compute subject-level VTC median across all runs, then split
subj_mean_vtc = {cond: {'high': [], 'low': []} for cond in conditions}
for subj, runs in pupil_dict.items():
    for cond in conditions:
        # Pool all VTC values within this subject to get the subject-level median
        subj_vtc_all = []
        for run_data in runs.values():
            if run_data['sfreq_neon'] != 125:
                continue
            subj_vtc_all.extend(run_data[f'{cond}_vtc'].tolist())
        if not subj_vtc_all:
            continue
        subj_median = np.median(subj_vtc_all)

        # Split trials by subject-level median
        high_epochs, low_epochs = [], []
        for run_data in runs.values():
            if run_data['sfreq_neon'] != 125:
                continue
            for ep, vtc_val in zip(run_data[cond], run_data[f'{cond}_vtc']):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                ep_corr = ep - baseline_val
                if vtc_val >= subj_median:
                    high_epochs.append(ep_corr)
                else:
                    low_epochs.append(ep_corr)
        if high_epochs:
            subj_mean_vtc[cond]['high'].append(np.nanmean(high_epochs, axis=0))
        if low_epochs:
            subj_mean_vtc[cond]['low'].append(np.nanmean(low_epochs, axis=0))

t_epoch = np.linspace(-baseline_length, epoch_length,
                      len(next(v for v in subj_mean.values() if v)[0]))
colors = {'mnt_correct': 'tab:blue', 'mnt_incorrect': 'tab:orange',
          'city_correct': 'tab:green', 'city_incorrect': 'tab:red'}
linestyles = {'high': '-', 'low': '--'}

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Onset')
for cond in conditions:
    for split, ls in linestyles.items():
        arr = np.array(subj_mean_vtc[cond][split])
        if len(arr) == 0:
            continue
        mean = arr.mean(axis=0)
        sem  = arr.std(axis=0) / np.sqrt(len(arr))
        label = f"{cond.replace('_', ' ')} ({split} VTC)"
        line, = ax.plot(t_epoch, mean, color=colors[cond], linestyle=ls, label=label)
        ax.fill_between(t_epoch, mean - sem, mean + sem, alpha=0.2,
                        color=colors[cond], label='_nolegend_')
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title('Pupil epoch by VTC median split (cross-subject)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pupil diameter (mm)')
ax.legend(fontsize=8, ncol=2)
ax.grid()
plt.tight_layout()

#%%


#%% Coherence


