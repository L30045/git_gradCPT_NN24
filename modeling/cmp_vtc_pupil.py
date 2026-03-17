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

#%% load pupil size
# subj_id_array = [670, 695,721,723,726, 730]
subj_id = 726
run_id = 3

# file path
subj_neon = os.path.join(project_path, "sourcedata", "raw", f"sub-{subj_id}", "eye_tracking")
subj_snirf = os.path.join(project_path, f"sub-{subj_id}", "nirs")
snirf_files = sorted([f for f in os.listdir(subj_snirf) if f.endswith(".snirf")])
neon_dirs   = sorted([d for d in os.listdir(subj_neon) if re.match(r"\d{4}-", d)])
neon_idx    = snirf_files.index(f"sub-{subj_id}_task-gradCPT_run-0{run_id}_nirs.snirf")

# load neon
filename = os.path.join(project_path, f"sub-{subj_id}","nirs", f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
neon_data = pd.read_csv(filename, sep='\t')

# preprocess pupil size
rec = nr.open(os.path.join(subj_neon, neon_dirs[neon_idx]))
t_neon, pupil_d = preprocess_pupil(neon_data, rec)

                    
# smooth pupil_d using same smoothing for VTC
sfreq_neon = np.round(np.median(1/np.diff(t_neon)))
len_smooth = 20*0.8 # sec (number of trial *  length of trial) When smoothing VTC, we used 20 trials.
win_smooth = (len_smooth*sfreq_neon).astype(int)
# smooth pupil size
pupil_d_smoothed = utils.smoothing_VTC_gaussian_array(pupil_d,L=win_smooth)

# load SNIRF event file
event_file = os.path.join(project_path,f"sub-{subj_id}","nirs",
                        f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
events_df = pd.read_csv(event_file,sep='\t')
t_vtc = events_df['onset'].values
vtc_smoothed = utils.smoothing_VTC_gaussian_array(events_df['VTC'],L=20)
# get pupil size at t_vtc
pupil_d_ev = np.array([pupil_d_smoothed[np.searchsorted(t_neon, t, side='right')] for t in t_vtc])
# normalized both vtc_smoothed and pupil_d_ev
vtc_smoothed_norm = (vtc_smoothed - vtc_smoothed.mean()) / vtc_smoothed.std()
pupil_d_ev_norm = (pupil_d_ev - pupil_d_ev.mean()) / pupil_d_ev.std()

# calculate the correlation between vtc_smoothed_norm and pupil_d_ev_norm
corr, pval = sp.stats.pearsonr(vtc_smoothed_norm, pupil_d_ev_norm)

# visualize
plt.figure()
plt.plot(t_vtc, vtc_smoothed_norm, label='VTC')
plt.plot(t_vtc, pupil_d_ev_norm, label='Pupil size')
plt.legend()
plt.grid()
plt.title(f"sub-{subj_id}, run-{run_id:02d} (r={corr:.2f}, p={pval:.3f})")
plt.xlabel('Time')
plt.ylabel('Norm')


#%% compare filter/ unfilter pupil size
plt.figure()
plt.plot(t_neon[:10000], pupil_d_ori[:10000], label='Pupil size (with blink)')
plt.plot(t_neon[:10000], pupil_d[:10000], label='Pupil size (without blink)')
plt.plot(t_neon[:10000], pupil_d_smoothed[:10000], label='Pupil size (filtered)')
plt.legend()
plt.grid()
plt.xlabel('Time')
plt.ylabel('mm')

#%% plot full runs from one subject
subj_id = 670
p_list = []

for i in range(3):
    run_id = i +1
    # file path
    subj_neon = os.path.join(project_path, "sourcedata", "raw", f"sub-{subj_id}", "eye_tracking")
    subj_snirf = os.path.join(project_path, f"sub-{subj_id}", "nirs")
    snirf_files = sorted([f for f in os.listdir(subj_snirf) if f.endswith(".snirf")])
    neon_dirs   = sorted([d for d in os.listdir(subj_neon) if re.match(r"\d{4}-", d)])
    neon_idx    = snirf_files.index(f"sub-{subj_id}_task-gradCPT_run-0{run_id}_nirs.snirf")

    # load neon
    filename = os.path.join(project_path, f"sub-{subj_id}","nirs", f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
    neon_data = pd.read_csv(filename, sep='\t')
    t_neon = neon_data['timestamps']
    pupil_d = (neon_data['eyeleft_pupilDiameter']+neon_data['eyeright_pupilDiameter'])/2
    # smooth pupil_d using same smoothing for VTC
    sfreq_neon = np.round(np.median(1/np.diff(t_neon)))
    len_smooth = 20*0.8 # sec (number of trial *  length of trial) When smoothing VTC, we used 20 trials.
    win_smooth = (len_smooth*sfreq_neon).astype(int)
    # smooth pupil size
    pupil_d_smoothed = utils.smoothing_VTC_gaussian_array(pupil_d,L=win_smooth)
    p_list.append(pupil_d_smoothed)
    
# visualize
plt.figure()
plt.plot(np.concatenate(p_list), label='Pupil size')
plt.legend()
plt.grid()
plt.title(f"sub-{subj_id}, run-{run_id:02d} ")
plt.xlabel('Time')
plt.ylabel('Norm')

#%% Epoch pupil size
epoch_length = 1.6 # sec (fade-in + fade-out)
baseline_length = 0.2 # sec (previous event's fade-out phase)
# get event type index
mnt_correct_idx = (events_df['trial_type']=='mnt')&(events_df["response_code"]==0)
mnt_incorrect_idx = (events_df['trial_type']=='mnt')&(events_df["response_code"]!=0)
city_correct_idx =  (events_df['trial_type']=='city')&(events_df["response_code"]!=0)
city_incorrect_idx =  (events_df['trial_type']=='city')&(events_df["response_code"]==0)
# for each event index
ev_idx = mnt_correct_idx


fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(0, color='k', linestyle='--', label='Onset')
for ev_idx, label in [
    (mnt_correct_idx,   'mnt — Correct'),
    (mnt_incorrect_idx, 'mnt — Incorrect'),
    (city_correct_idx,  'city — Correct'),
    (city_incorrect_idx,'city — Incorrect'),
]:
    plot_pupil_epoch(ax, get_pupil_epoch(events_df, ev_idx, t_neon, pupil_d, baseline_length, epoch_length), baseline_length, epoch_length, label)
# add a single SEM proxy entry
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title('Pupil epoch')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pupil diameter (mm)')
ax.legend()
ax.grid()
plt.tight_layout()

#%% cross subjects epoch analysis
dirs = os.listdir(project_path)
subject_list = [d for d in dirs if 'sub' in d] # and d not in excluded]

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
        win_epoch    = int(np.round(sfreq_neon * epoch_length))
        win_baseline = int(np.round(sfreq_neon * baseline_length))
        t_epoch_run  = np.linspace(-baseline_length, epoch_length, win_baseline + win_epoch)
        pupil_dict[subj][f'run-{run_id:02d}'] = {
            'mnt_correct':    get_pupil_epoch(events_df_s, mnt_correct_idx_s,   t_neon_s, pupil_d_s),
            'mnt_incorrect':  get_pupil_epoch(events_df_s, mnt_incorrect_idx_s, t_neon_s, pupil_d_s),
            'city_correct':   get_pupil_epoch(events_df_s, city_correct_idx_s,  t_neon_s, pupil_d_s),
            'city_incorrect': get_pupil_epoch(events_df_s, city_incorrect_idx_s,t_neon_s, pupil_d_s),
            't_epoch':        t_epoch_run,
            'sfreq_neon':     sfreq_neon,
        }
pupil_dict = {subj: runs for subj, runs in pupil_dict.items() if runs}

#%% Cross-subject epoch analysis
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

#%% Check file availability for each subject
file_check_rows = []
for subj in sorted(subject_list):
    subj_id = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    has_dirs = os.path.isdir(subj_nirs) and os.path.isdir(subj_neon_dir)
    if not has_dirs:
        for run_id in range(1, 4):
            file_check_rows.append({'subject': subj, 'run': f'run-{run_id:02d}',
                                    'snirf': False, 'physio': False, 'events': False})
        continue
    snirf_files = sorted([f for f in os.listdir(subj_nirs) if f.endswith('.snirf')])
    for run_id in range(1, 4):
        snirf_name  = f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_nirs.snirf"
        physio_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
        event_file  = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        file_check_rows.append({
            'subject': subj,
            'run':     f'run-{run_id:02d}',
            'snirf':   snirf_name in snirf_files,
            'physio':  os.path.isfile(physio_file),
            'events':  os.path.isfile(event_file),
        })
file_check_df = pd.DataFrame(file_check_rows).set_index(['subject', 'run'])
print(file_check_df)

#%%
#TODO: check why physio file is missing

#%% Coherence


