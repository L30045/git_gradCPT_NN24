#%% load library
import numpy as np
import scipy as sp
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
# import utils
from params_setting import *
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from neon_to_bids_gradCPT_add_nodding import parseNeon_to_bids
from pupil_labs import neon_recording as nr
import re
from utils_eyetracking import preprocess_pupil, get_pupil_epoch, plot_pupil_epoch
import warnings

#%% functions
def smoothing_VTC_gaussian_array(vtc, sigma=None, alpha=2.5, L=None, radius=None, truncate=4, savepath=None):
    """
    Smooth VTC using gaussian_filter1d.
    ---------------------------------------------------
    Input:
        vtc_dict: VTC Dict.
        sigma: stddev used in gaussian filter.
        alpha: default parameter for determining sigma in Matlab gausswin function
        L: designed window size. use this parameter to determine radius. If None, define window size by sigma and truncate, or radius.
        radius: radius of gaussian filter. Ignored if L is given.
        truncate: number of stddev away from center will be truncated
        savepath: where to save smoothed VTC
    Output:
        smooth_vtc_dict: smoothed VTC Dict.
    """
    # define  from L and alpha if sigma is not given
    if not sigma:
        if not L:
            sigma = 12
        else:
            sigma = (L-1)/(2*alpha) # default formula for stddev in Matlab gausswin function
    # define radius if L is given
    if L:
        radius = np.ceil((L-1)/2).astype(int) # radius of gaussian filter
    # smooth VTC
    smooth_vtc = sp.ndimage.gaussian_filter1d(vtc, sigma=sigma, radius=radius, truncate=4)
    return smooth_vtc

def smooth_t(x, t, smooth_win=1):
    """sliding window average along a signal (preserves length via 'same' mode)."""
    dt  = t[1] - t[0]
    win = max(1, int(round(smooth_win / dt)))
    return np.convolve(x, np.ones(win) / win, mode='same')

def diff_epoch(ep, t_ep):
    """Time derivative of a pupil epoch via np.gradient (mm/s)."""
    ep = np.array(ep, dtype=float)
    if np.all(np.isnan(ep)):
        return ep
    return np.gradient(ep, t_ep)

def detrend_epoch(ep):
    """Detrend linearly, NaN-safe. Returns float array."""
    ep = np.array(ep, dtype=float)
    if np.all(np.isnan(ep)):
        return ep
    mask = ~np.isnan(ep)
    ep[mask] = sp.signal.detrend(ep[mask])
    return ep

def detrend_run(pupil_d):
    """Quadratic detrend of a full-run pupil signal, NaN-aware.
    Fits a least-squares parabola through valid samples and subtracts it."""
    x = np.array(pupil_d, dtype=float)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < 3:
        return x
    t = np.arange(len(x), dtype=float)
    coef = np.polyfit(t[valid], x[valid], 2)
    return x - np.polyval(coef, t)

def collect_rt_epochs(pupil_dict, epoch_key, vtc_key, rt_pre, rt_post):
    epochs, vtcs = [], []
    for subj, runs in pupil_dict.items():
        for run_data in runs.values():
            for ep, vtc_val in zip(run_data[epoch_key], run_data[vtc_key]):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-rt_pre, rt_post, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                epochs.append(ep - baseline_val)
                vtcs.append(vtc_val)
    return epochs, vtcs

def collect_rt_epochs_dt(pupil_dict, epoch_key, vtc_key, rt_pre, rt_post):
    epochs, vtcs = [], []
    for subj, runs in pupil_dict.items():
        for run_data in runs.values():
            for ep, vtc_val in zip(run_data[epoch_key], run_data[vtc_key]):
                ep = detrend_epoch(ep)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-rt_pre, rt_post, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                epochs.append(ep - baseline_val)
                vtcs.append(vtc_val)
    return epochs, vtcs

def sort_and_smooth(epochs, vtcs, vis_smooth):
    epochs = np.array(epochs)
    vtcs   = np.array(vtcs)
    sort_idx = np.argsort(vtcs)
    epochs   = epochs[sort_idx]
    vtcs     = vtcs[sort_idx]
    win = max(1, int(epochs.shape[0] / 100 * vis_smooth))
    epochs = np.apply_along_axis(
        lambda col: np.convolve(col, np.ones(win) / win, mode='same'),
        axis=0, arr=epochs)
    return epochs, vtcs


#%% cross subjects epoch analysis
dirs = os.listdir(project_path)
subject_list = [d for d in dirs if 'sub' in d] # and d not in excluded]

epoch_length = 12 # sec (fade-in + fade-out)
baseline_length = 1 # sec (previous event's fade-out phase)
rt_pre, rt_post = 5, 12  # sec before/after RT
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
        if not os.path.isfile(physio_file):
            physio_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
        event_file  = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if snirf_name not in snirf_files or not os.path.isfile(physio_file) or not os.path.isfile(event_file):
            continue
        neon_idx = snirf_files.index(snirf_name)
        neon_data = pd.read_csv(physio_file, sep='\t')
        # check if the data is recorded by Neon
        if neon_dirs_subj:
            rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[neon_idx]))
        else:
            rec = None
        t_neon_s, pupil_d_s = preprocess_pupil(neon_data, rec=rec)
        # check if subject missing too many data
        if t_neon_s is None:
            print(f"Missing too many data. Skip sub-{subj}")
            continue
        sfreq_neon = np.round(np.median(1 / np.diff(t_neon_s)))
        print(f"{subj} run-{run_id:02d}: sfreq_neon = {sfreq_neon} Hz")
        events_df_s = pd.read_csv(event_file, sep='\t')
        mnt_correct_idx_s   = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']==0)
        mnt_incorrect_idx_s = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']!=0)
        city_correct_idx_s  = (events_df_s['trial_type']=='city') & (events_df_s['response_code']!=0)
        city_incorrect_idx_s= (events_df_s['trial_type']=='city') & (events_df_s['response_code']==0)
        vtc_s = smoothing_VTC_gaussian_array(events_df_s['VTC'].values, L=20)
        win_epoch    = int(np.round(sfreq_neon * epoch_length))
        win_baseline = int(np.round(sfreq_neon * baseline_length))
        t_epoch_run  = np.linspace(-baseline_length, epoch_length, win_baseline + win_epoch)
        # RT-locked epoch for mnt_incorrect: shift onset by reaction_time
        events_df_rt_mnt = events_df_s[mnt_incorrect_idx_s].copy()
        if len(events_df_rt_mnt) > 0:
            events_df_rt_mnt['onset'] = events_df_rt_mnt['onset'] + events_df_rt_mnt['reaction_time']
            rt_mnt_all_idx = pd.Series([True] * len(events_df_rt_mnt), index=events_df_rt_mnt.index)
        else:
            rt_mnt_all_idx = []
        # RT-locked epoch for city_correct: shift onset by reaction_time
        events_df_rt_city_correct = events_df_s[city_correct_idx_s].copy()
        if len(events_df_rt_city_correct) > 0:
            events_df_rt_city_correct['onset'] = events_df_rt_city_correct['onset'] + events_df_rt_city_correct['reaction_time']
            rt_city_correct_all_idx = pd.Series([True] * len(events_df_rt_city_correct), index=events_df_rt_city_correct.index)
        else:
            rt_city_correct_all_idx = []
        # RT-locked epoch for city_incorrect: shift onset by reaction_time
        events_df_rt_city_incorrect = events_df_s[city_incorrect_idx_s].copy()
        if len(events_df_rt_city_incorrect) > 0:
            events_df_rt_city_incorrect['onset'] = events_df_rt_city_incorrect['onset'] + events_df_rt_city_incorrect['reaction_time']
            rt_city_incorrect_all_idx = pd.Series([True] * len(events_df_rt_city_incorrect), index=events_df_rt_city_incorrect.index)
        else:
            rt_city_incorrect_all_idx = []
        pupil_dict[subj][f'run-{run_id:02d}'] = {
            'mnt_correct':                  get_pupil_epoch(events_df_s, mnt_correct_idx_s,   t_neon_s, pupil_d_s, epoch_length=epoch_length),
            'mnt_incorrect':                get_pupil_epoch(events_df_s, mnt_incorrect_idx_s, t_neon_s, pupil_d_s, epoch_length=epoch_length),
            'city_correct':                 get_pupil_epoch(events_df_s, city_correct_idx_s,  t_neon_s, pupil_d_s, epoch_length=epoch_length),
            'city_incorrect':               get_pupil_epoch(events_df_s, city_incorrect_idx_s,t_neon_s, pupil_d_s, epoch_length=epoch_length),
            'mnt_incorrect_rt_epoch':       get_pupil_epoch(events_df_rt_mnt, rt_mnt_all_idx, t_neon_s, pupil_d_s,
                                                            baseline_length=rt_pre, epoch_length=rt_post)
                                            if len(events_df_rt_mnt) > 0 else [],
            'city_correct_rt_epoch':        get_pupil_epoch(events_df_rt_city_correct, rt_city_correct_all_idx, t_neon_s, pupil_d_s,
                                                            baseline_length=rt_pre, epoch_length=rt_post)
                                            if len(events_df_rt_city_correct) > 0 else [],
            'city_incorrect_rt_epoch':      get_pupil_epoch(events_df_rt_city_incorrect, rt_city_incorrect_all_idx, t_neon_s, pupil_d_s,
                                                            baseline_length=rt_pre, epoch_length=rt_post)
                                            if len(events_df_rt_city_incorrect) > 0 else [],
            'mnt_correct_vtc':              vtc_s[mnt_correct_idx_s.values],
            'mnt_incorrect_vtc':            vtc_s[mnt_incorrect_idx_s.values],
            'city_correct_vtc':             vtc_s[city_correct_idx_s.values],
            'city_incorrect_vtc':           vtc_s[city_incorrect_idx_s.values],
            'mnt_incorrect_rt':             events_df_s[mnt_incorrect_idx_s]['reaction_time'].values,
            'city_correct_rt':              events_df_s[city_correct_idx_s]['reaction_time'].values,
            'city_incorrect_rt':            events_df_s[city_incorrect_idx_s]['reaction_time'].values,
            # Raw signal + RT-shifted onsets stored for re-epoching at analysis time
            't_neon':                       t_neon_s,
            'pupil_d':                      pupil_d_s,
            'mnt_incorrect_rt_onsets':      events_df_rt_mnt['onset'].values
                                            if len(events_df_rt_mnt) > 0 else np.array([]),
            'city_correct_rt_onsets':       events_df_rt_city_correct['onset'].values
                                            if len(events_df_rt_city_correct) > 0 else np.array([]),
            't_epoch':                      t_epoch_run,
            'sfreq_neon':                   sfreq_neon,
        }
pupil_dict = {subj: runs for subj, runs in pupil_dict.items() if runs}

#%% Plot Cross-subject results
conditions = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']

# For each subject, pool epochs across runs then compute the per-subject mean time course
subj_mean = {cond: [] for cond in conditions}
for subj, runs in pupil_dict.items():
    for cond in conditions:
        # concatenate all trials across runs with per-epoch baseline correction
        all_epochs = []
        for run_data in runs.values():
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
            subj_vtc_all.extend(run_data[f'{cond}_vtc'].tolist())
        if not subj_vtc_all:
            continue
        subj_median = np.median(subj_vtc_all)

        # Split trials by subject-level median
        high_epochs, low_epochs = [], []
        for run_data in runs.values():
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
colors = {
    'mnt_correct':   {'high': 'royalblue',   'low': 'lightskyblue'},
    'mnt_incorrect': {'high': 'darkorange',  'low': 'gold'},
    'city_correct':  {'high': 'forestgreen', 'low': 'limegreen'},
    'city_incorrect':{'high': 'crimson',     'low': 'lightcoral'},
}
linestyles = {'high': '-', 'low': '--'}

n_subjects = len(pupil_dict)

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
        c = colors[cond][split]
        line, = ax.plot(t_epoch, mean, color=c, linestyle=ls, label=label)
        ax.fill_between(t_epoch, mean - sem, mean + sem, alpha=0.2,
                        color=c, label='_nolegend_')
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title(f'Pupil epoch by VTC median split (cross-subject, N={n_subjects})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pupil diameter (mm)')
ax.legend(fontsize=8, ncol=2)
ax.grid()
plt.tight_layout()


#%% RT-locked: VTC median split per condition
rt_vtc_specs = [
    ('mnt_incorrect_rt_epoch', 'mnt_incorrect_vtc',
     {'high': 'darkorange', 'low': 'gold'}, 'mnt Incorrect'),
    ('city_correct_rt_epoch',  'city_correct_vtc',
     {'high': 'forestgreen', 'low': 'limegreen'}, 'city Correct'),
]
n_subjects = len(pupil_dict)


for epoch_key, vtc_key, colors_rt, cond_title in rt_vtc_specs:
    subj_mean_rt_vtc = {'high': [], 'low': []}

    for subj, runs in pupil_dict.items():
        # Subject-level VTC median from this condition
        subj_vtc_all = []
        for run_data in runs.values():
            subj_vtc_all.extend(run_data[vtc_key].tolist())
        if not subj_vtc_all:
            continue
        subj_median = np.median(subj_vtc_all)

        high_epochs, low_epochs = [], []
        for run_data in runs.values():
            for ep, vtc_val in zip(run_data[epoch_key], run_data[vtc_key]):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-rt_pre, rt_post, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                ep_corr = ep - baseline_val
                if vtc_val >= subj_median:
                    high_epochs.append(ep_corr)
                else:
                    low_epochs.append(ep_corr)
        if high_epochs:
            subj_mean_rt_vtc['high'].append(np.nanmean(high_epochs, axis=0))
        if low_epochs:
            subj_mean_rt_vtc['low'].append(np.nanmean(low_epochs, axis=0))

    t_rt_vtc = np.linspace(-rt_pre, rt_post,
                           len(next(v for v in subj_mean_rt_vtc.values() if v)[0]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Onset')
    for split, ls in [('high', '-'), ('low', '--')]:
        arr = np.array(subj_mean_rt_vtc[split])
        if len(arr) == 0:
            continue
        mean = arr.mean(axis=0)
        sem  = arr.std(axis=0) / np.sqrt(len(arr))
        c = colors_rt[split]
        ax.plot(t_rt_vtc, mean, color=c, linestyle=ls, label=f'{split} VTC')
        ax.fill_between(t_rt_vtc, mean - sem, mean + sem, alpha=0.25,
                        color=c, label='_nolegend_')
    ax.fill_between([], [], alpha=0.25, color='gray', label='SEM')
    ax.set_title(f'RT-locked pupil by VTC — {cond_title} (cross-subject, N={n_subjects})')
    ax.set_xlabel('Time relative to RT (s)')
    ax.set_ylabel('Pupil diameter (mm)')
    ax.legend(fontsize=8)
    ax.grid()
    plt.tight_layout()


#%% Epoch raster sorted by VTC
# Pool all epochs + VTC values across subjects/runs for each condition
vis_smooth = 10 # percentage of smooth
cond_labels = {
    'mnt_correct':   'mnt — Correct',
    'mnt_incorrect': 'mnt — Incorrect',
    'city_correct':  'city — Correct',
    'city_incorrect':'city — Incorrect',
}
t_epoch_raster = np.linspace(-baseline_length, epoch_length,
                             len(next(v for v in subj_mean.values() if v)[0]))

# First pass: collect all smoothed epochs to get a shared color range
all_sorted_data = {}
for cond in cond_labels:
    all_epochs_cond, all_vtc_cond, all_rt_cond = [], [], []
    rt_key = f'{cond}_rt' if f'{cond}_rt' in next(iter(next(iter(pupil_dict.values())).values())) else None
    for subj, runs in pupil_dict.items():
        for run_data in runs.values():
            rt_arr = run_data[rt_key] if rt_key else np.full(len(run_data[cond]), np.nan)
            for ep, vtc_val, rt_val in zip(run_data[cond], run_data[f'{cond}_vtc'], rt_arr):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
                baseline_val = np.nanmean(ep[t_ep < 0])
                all_epochs_cond.append(ep - baseline_val)
                all_vtc_cond.append(vtc_val)
                all_rt_cond.append(rt_val)
    if not all_epochs_cond:
        continue
    all_epochs_cond = np.array(all_epochs_cond)
    all_vtc_cond    = np.array(all_vtc_cond)
    all_rt_cond     = np.array(all_rt_cond)
    sort_idx        = np.argsort(all_vtc_cond)
    sorted_epochs   = all_epochs_cond[sort_idx]
    win             = max(1, int(sorted_epochs.shape[0] / 100 * vis_smooth))
    sorted_epochs   = np.apply_along_axis(
        lambda col: np.convolve(col, np.ones(win) / win, mode='same'),
        axis=0, arr=sorted_epochs)
    all_sorted_data[cond] = {'epochs': sorted_epochs, 'vtc': all_vtc_cond[sort_idx],
                             'rt': all_rt_cond[sort_idx]}

global_vmax = np.nanmax(
    np.abs(np.concatenate([d['epochs'].ravel() for d in all_sorted_data.values()])))

# Second pass: plot with shared color range
for cond, title in cond_labels.items():
    if cond not in all_sorted_data:
        continue
    sorted_epochs = all_sorted_data[cond]['epochs']
    sorted_vtc    = all_sorted_data[cond]['vtc']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={'width_ratios': [1, 8]})
    # Left panel: VTC bar
    axes[0].barh(np.arange(len(sorted_vtc)), sorted_vtc, color='steelblue', height=1.0)
    axes[0].set_ylim(-0.5, len(sorted_vtc) - 0.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('VTC')
    axes[0].set_ylabel('Trial (sorted)')
    # Right panel: epoch heatmap
    im = axes[1].imshow(sorted_epochs, aspect='auto', origin='upper',
                        extent=[t_epoch_raster[0], t_epoch_raster[-1],
                                len(sorted_vtc), 0],
                        cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1)
    # Overlay reaction time for mnt_incorrect
    rt = all_sorted_data[cond]['rt']
    if not np.all(np.isnan(rt)):
        axes[1].scatter(rt, np.arange(len(rt)) + 0.5,
                        color='k', s=4, marker='|', linewidths=0.8,
                        label='Reaction time', zorder=3)
        axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], label='Pupil diameter (mm)')
    fig.suptitle(f'Epoch raster sorted by VTC — {title}')
    plt.tight_layout()

#%% RT-locked raster for mnt_incorrect and city_correct (shared color scale)
rt_pre, rt_post = 0.5, 1.0
t_epoch_rt = np.linspace(-rt_pre, rt_post,
                         int(np.round(125 * (rt_pre + rt_post))))  # assumes sfreq=125

# Collect and process both conditions
rt_data = {}
for epoch_key, vtc_key, label in [
    ('mnt_incorrect_rt_epoch', 'mnt_incorrect_vtc', 'mnt Incorrect'),
    ('city_correct_rt_epoch',  'city_correct_vtc',  'city Correct'),
]:
    raw_epochs, raw_vtcs = collect_rt_epochs(pupil_dict, epoch_key, vtc_key, rt_pre, rt_post)
    if raw_epochs:
        s_epochs, s_vtcs = sort_and_smooth(raw_epochs, raw_vtcs, vis_smooth)
        rt_data[label] = {'epochs': s_epochs, 'vtcs': s_vtcs}

# Shared color scale across both conditions
if rt_data:
    shared_vmax = max(np.nanmax(np.abs(d['epochs'])) for d in rt_data.values())

    for label, d in rt_data.items():
        s_epochs, s_vtcs = d['epochs'], d['vtcs']
        t_axis = np.linspace(-rt_pre, rt_post, s_epochs.shape[1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                                 gridspec_kw={'width_ratios': [1, 8]})
        axes[0].barh(np.arange(len(s_vtcs)), s_vtcs, color='steelblue', height=1.0)
        axes[0].set_ylim(-0.5, len(s_vtcs) - 0.5)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('VTC')
        axes[0].set_ylabel('Trial (sorted by VTC)')
        im = axes[1].imshow(s_epochs, aspect='auto', origin='upper',
                            extent=[t_axis[0], t_axis[-1], len(s_vtcs), 0],
                            cmap='RdBu_r', vmin=-shared_vmax, vmax=shared_vmax)
        axes[1].axvline(0, color='k', linestyle='--', linewidth=1, label='Reaction time')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].set_xlabel('Time relative to RT (s)')
        axes[1].set_yticks([])
        plt.colorbar(im, ax=axes[1], label='Pupil diameter (mm)')
        fig.suptitle(f'Pupil epoch — {label} (RT-locked, sorted by VTC)')
        plt.tight_layout()

#%% Detrend after epoching
# # Repeat entire analysis with per-epoch linear detrending before baseline correction

# # --- Cross-subject mean (detrended) ---
# subj_mean_dt = {cond: [] for cond in conditions}
# for subj, runs in pupil_dict.items():
#     for cond in conditions:
#         all_epochs = []
#         for run_data in runs.values():
#             if run_data['sfreq_neon'] != 125:
#                 continue
#             for ep in run_data[cond]:
#                 ep = detrend_epoch(ep)
#                 if np.all(np.isnan(ep)):
#                     continue
#                 t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
#                 baseline_val = np.nanmean(ep[t_ep < 0])
#                 all_epochs.append(ep - baseline_val)
#         if len(all_epochs) == 0:
#             continue
#         try:
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('error')
#                 subj_mean_dt[cond].append(np.nanmean(np.array(all_epochs), axis=0))
#         except Warning as w:
#             print(f"Skipping {subj} {cond}: {w}")

# t_epoch_dt = np.linspace(-baseline_length, epoch_length,
#                          len(next(v for v in subj_mean_dt.values() if v)[0]))
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.axvline(0, color='k', linestyle='--', label='Onset')
# for cond, label in [
#     ('mnt_correct',   'mnt — Correct'),
#     ('mnt_incorrect', 'mnt — Incorrect'),
#     ('city_correct',  'city — Correct'),
#     ('city_incorrect','city — Incorrect'),
# ]:
#     arr = np.array(subj_mean_dt[cond])
#     if len(arr) == 0:
#         continue
#     mean = arr.mean(axis=0)
#     sem  = arr.std(axis=0) / np.sqrt(len(arr))
#     line, = ax.plot(t_epoch_dt, mean, label=label)
#     ax.fill_between(t_epoch_dt, mean - sem, mean + sem, alpha=0.3,
#                     color=line.get_color(), label='_nolegend_')
# ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
# ax.set_title('Pupil epoch — detrended (cross-subject)')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Pupil diameter (mm)')
# ax.legend()
# ax.grid()
# plt.tight_layout()

# # --- VTC median split (detrended) ---
# subj_mean_vtc_dt = {cond: {'high': [], 'low': []} for cond in conditions}
# for subj, runs in pupil_dict.items():
#     for cond in conditions:
#         subj_vtc_all = []
#         for run_data in runs.values():
#             if run_data['sfreq_neon'] != 125:
#                 continue
#             subj_vtc_all.extend(run_data[f'{cond}_vtc'].tolist())
#         if not subj_vtc_all:
#             continue
#         subj_median = np.median(subj_vtc_all)
#         high_epochs, low_epochs = [], []
#         for run_data in runs.values():
#             if run_data['sfreq_neon'] != 125:
#                 continue
#             for ep, vtc_val in zip(run_data[cond], run_data[f'{cond}_vtc']):
#                 ep = detrend_epoch(ep)
#                 if np.all(np.isnan(ep)):
#                     continue
#                 t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
#                 baseline_val = np.nanmean(ep[t_ep < 0])
#                 ep_corr = ep - baseline_val
#                 if vtc_val >= subj_median:
#                     high_epochs.append(ep_corr)
#                 else:
#                     low_epochs.append(ep_corr)
#         if high_epochs:
#             subj_mean_vtc_dt[cond]['high'].append(np.nanmean(high_epochs, axis=0))
#         if low_epochs:
#             subj_mean_vtc_dt[cond]['low'].append(np.nanmean(low_epochs, axis=0))

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Onset')
# for cond in conditions:
#     for split, ls in linestyles.items():
#         arr = np.array(subj_mean_vtc_dt[cond][split])
#         if len(arr) == 0:
#             continue
#         mean = arr.mean(axis=0)
#         sem  = arr.std(axis=0) / np.sqrt(len(arr))
#         label = f"{cond.replace('_', ' ')} ({split} VTC)"
#         c = colors[cond][split]
#         line, = ax.plot(t_epoch_dt, mean, color=c, linestyle=ls, label=label)
#         ax.fill_between(t_epoch_dt, mean - sem, mean + sem, alpha=0.2,
#                         color=c, label='_nolegend_')
# ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
# ax.set_title(f'Pupil epoch by VTC median split — detrended (cross-subject, N={n_subjects})')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Pupil diameter (mm)')
# ax.legend(fontsize=8, ncol=2)
# ax.grid()
# plt.tight_layout()

# # --- Epoch raster sorted by VTC (detrended) ---
# all_sorted_data_dt = {}
# for cond in cond_labels:
#     all_epochs_cond, all_vtc_cond, all_rt_cond = [], [], []
#     rt_key = f'{cond}_rt' if f'{cond}_rt' in next(iter(next(iter(pupil_dict.values())).values())) else None
#     for subj, runs in pupil_dict.items():
#         for run_data in runs.values():
#             if run_data['sfreq_neon'] != 125:
#                 continue
#             rt_arr = run_data[rt_key] if rt_key else np.full(len(run_data[cond]), np.nan)
#             for ep, vtc_val, rt_val in zip(run_data[cond], run_data[f'{cond}_vtc'], rt_arr):
#                 ep = detrend_epoch(ep)
#                 if np.all(np.isnan(ep)):
#                     continue
#                 t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
#                 baseline_val = np.nanmean(ep[t_ep < 0])
#                 all_epochs_cond.append(ep - baseline_val)
#                 all_vtc_cond.append(vtc_val)
#                 all_rt_cond.append(rt_val)
#     if not all_epochs_cond:
#         continue
#     all_epochs_cond = np.array(all_epochs_cond)
#     all_vtc_cond    = np.array(all_vtc_cond)
#     all_rt_cond     = np.array(all_rt_cond)
#     sort_idx        = np.argsort(all_vtc_cond)
#     sorted_epochs   = all_epochs_cond[sort_idx]
#     win             = max(1, int(sorted_epochs.shape[0] / 100 * vis_smooth))
#     sorted_epochs   = np.apply_along_axis(
#         lambda col: np.convolve(col, np.ones(win) / win, mode='same'),
#         axis=0, arr=sorted_epochs)
#     all_sorted_data_dt[cond] = {'epochs': sorted_epochs, 'vtc': all_vtc_cond[sort_idx],
#                                 'rt': all_rt_cond[sort_idx]}

# global_vmax_dt = np.nanmax(
#     np.abs(np.concatenate([d['epochs'].ravel() for d in all_sorted_data_dt.values()])))

# for cond, title in cond_labels.items():
#     if cond not in all_sorted_data_dt:
#         continue
#     sorted_epochs = all_sorted_data_dt[cond]['epochs']
#     sorted_vtc    = all_sorted_data_dt[cond]['vtc']
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6),
#                              gridspec_kw={'width_ratios': [1, 8]})
#     axes[0].barh(np.arange(len(sorted_vtc)), sorted_vtc, color='steelblue', height=1.0)
#     axes[0].set_ylim(-0.5, len(sorted_vtc) - 0.5)
#     axes[0].invert_yaxis()
#     axes[0].set_xlabel('VTC')
#     axes[0].set_ylabel('Trial (sorted)')
#     im = axes[1].imshow(sorted_epochs, aspect='auto', origin='upper',
#                         extent=[t_epoch_raster[0], t_epoch_raster[-1], len(sorted_vtc), 0],
#                         cmap='RdBu_r', vmin=-global_vmax_dt, vmax=global_vmax_dt)
#     axes[1].axvline(0, color='k', linestyle='--', linewidth=1)
#     rt = all_sorted_data_dt[cond]['rt']
#     if not np.all(np.isnan(rt)):
#         axes[1].scatter(rt, np.arange(len(rt)) + 0.5,
#                         color='k', s=4, marker='|', linewidths=0.8,
#                         label='Reaction time', zorder=3)
#         axes[1].legend(loc='upper right', fontsize=8)
#     axes[1].set_xlabel('Time (s)')
#     axes[1].set_yticks([])
#     plt.colorbar(im, ax=axes[1], label='Pupil diameter (mm)')
#     fig.suptitle(f'Epoch raster sorted by VTC — detrended — {title}')
#     plt.tight_layout()

# # --- RT-locked rasters (detrended, shared color scale) ---
# rt_data_dt = {}
# for epoch_key, vtc_key, label in [
#     ('mnt_incorrect_rt_epoch', 'mnt_incorrect_vtc', 'mnt Incorrect'),
#     ('city_correct_rt_epoch',  'city_correct_vtc',  'city Correct'),
# ]:
#     raw_epochs, raw_vtcs = collect_rt_epochs_dt(pupil_dict, epoch_key, vtc_key, rt_pre, rt_post)
#     if raw_epochs:
#         s_epochs, s_vtcs = sort_and_smooth(raw_epochs, raw_vtcs, vis_smooth)
#         rt_data_dt[label] = {'epochs': s_epochs, 'vtcs': s_vtcs}

# if rt_data_dt:
#     shared_vmax_dt = max(np.nanmax(np.abs(d['epochs'])) for d in rt_data_dt.values())
#     for label, d in rt_data_dt.items():
#         s_epochs, s_vtcs = d['epochs'], d['vtcs']
#         t_axis = np.linspace(-rt_pre, rt_post, s_epochs.shape[1])
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6),
#                                  gridspec_kw={'width_ratios': [1, 8]})
#         axes[0].barh(np.arange(len(s_vtcs)), s_vtcs, color='steelblue', height=1.0)
#         axes[0].set_ylim(-0.5, len(s_vtcs) - 0.5)
#         axes[0].invert_yaxis()
#         axes[0].set_xlabel('VTC')
#         axes[0].set_ylabel('Trial (sorted by VTC)')
#         im = axes[1].imshow(s_epochs, aspect='auto', origin='upper',
#                             extent=[t_axis[0], t_axis[-1], len(s_vtcs), 0],
#                             cmap='RdBu_r', vmin=-shared_vmax_dt, vmax=shared_vmax_dt)
#         axes[1].axvline(0, color='k', linestyle='--', linewidth=1, label='Reaction time')
#         axes[1].legend(loc='upper right', fontsize=8)
#         axes[1].set_xlabel('Time relative to RT (s)')
#         axes[1].set_yticks([])
#         plt.colorbar(im, ax=axes[1], label='Pupil diameter (mm)')
#         fig.suptitle(f'Pupil epoch — {label} (RT-locked, detrended, sorted by VTC)')
#         plt.tight_layout()

#%% Pupil size derivative
# Rerun all analyses using dPupil/dt (np.gradient, mm/s) instead of raw pupil size.
conditions_stim = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']
smooth_win = 0.5
# --- Cross-subject mean (derivative) ---
subj_mean_dpp = {cond: [] for cond in conditions_stim}
for subj, runs in pupil_dict.items():
    for cond in conditions_stim:
        all_epochs = []
        for run_data in runs.values():
            for ep in run_data[cond]:
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
                dep = diff_epoch(ep, t_ep)
                bl  = np.nanmean(dep[t_ep < 0])
                all_epochs.append(dep - bl)
        if not all_epochs:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                subj_mean_dpp[cond].append(np.nanmean(np.array(all_epochs), axis=0))
        except Warning as w:
            print(f"Skipping {subj} {cond}: {w}")

t_epoch_dpp = np.linspace(-baseline_length, epoch_length,
                          len(next(v for v in subj_mean_dpp.values() if v)[0]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(0, color='k', linestyle='--', label='Onset')
for cond, label in [
    ('mnt_correct',   'mnt — Correct'),
    ('mnt_incorrect', 'mnt — Incorrect'),
    ('city_correct',  'city — Correct'),
    ('city_incorrect','city — Incorrect'),
]:
    arr = np.array(subj_mean_dpp[cond])
    if len(arr) == 0:
        continue
    mean = smooth_t(arr.mean(axis=0), t_epoch_dpp, smooth_win=smooth_win)
    sem  = smooth_t(arr.std(axis=0) / np.sqrt(len(arr)), t_epoch_dpp, smooth_win=smooth_win)
    line, = ax.plot(t_epoch_dpp, mean, label=label)
    ax.fill_between(t_epoch_dpp, mean - sem, mean + sem, alpha=0.3,
                    color=line.get_color(), label='_nolegend_')
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title('Pupil size derivative (cross-subject)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('dPupil/dt (mm/s)')
ax.legend()
ax.grid()
plt.tight_layout()

# --- VTC median split (derivative) ---
subj_mean_vtc_dpp = {cond: {'high': [], 'low': []} for cond in conditions_stim}
for subj, runs in pupil_dict.items():
    for cond in conditions_stim:
        subj_vtc_all = []
        for run_data in runs.values():
            subj_vtc_all.extend(run_data[f'{cond}_vtc'].tolist())
        if not subj_vtc_all:
            continue
        subj_median = np.median(subj_vtc_all)
        high_epochs, low_epochs = [], []
        for run_data in runs.values():
            for ep, vtc_val in zip(run_data[cond], run_data[f'{cond}_vtc']):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-baseline_length, epoch_length, len(ep))
                dep = diff_epoch(ep, t_ep)
                bl  = np.nanmean(dep[t_ep < 0])
                ep_corr = dep - bl
                if vtc_val >= subj_median:
                    high_epochs.append(ep_corr)
                else:
                    low_epochs.append(ep_corr)
        if high_epochs:
            subj_mean_vtc_dpp[cond]['high'].append(np.nanmean(high_epochs, axis=0))
        if low_epochs:
            subj_mean_vtc_dpp[cond]['low'].append(np.nanmean(low_epochs, axis=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Onset')
for cond in conditions_stim:
    for split, ls in linestyles.items():
        arr = np.array(subj_mean_vtc_dpp[cond][split])
        if len(arr) == 0:
            continue
        mean = smooth_t(arr.mean(axis=0), t_epoch_dpp, smooth_win=smooth_win)
        sem  = smooth_t(arr.std(axis=0) / np.sqrt(len(arr)), t_epoch_dpp, smooth_win=smooth_win)
        label = f"{cond.replace('_', ' ')} ({split} VTC)"
        c = colors[cond][split]
        ax.plot(t_epoch_dpp, mean, color=c, linestyle=ls, label=label)
        ax.fill_between(t_epoch_dpp, mean - sem, mean + sem, alpha=0.2,
                        color=c, label='_nolegend_')
ax.fill_between([], [], alpha=0.3, color='gray', label='SEM')
ax.set_title(f'Pupil size derivative by VTC median split (cross-subject, N={n_subjects})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('dPupil/dt (mm/s)')
ax.legend(fontsize=8, ncol=2)
ax.grid()
plt.tight_layout()

# --- RT-locked cross-subject mean (derivative) ---
rt_pre_data  = 5
rt_post_data = 12

rt_cond_specs_dpp = [
    ('mnt_incorrect_rt_onsets', 'mnt Incorrect'),
    ('city_correct_rt_onsets',  'city Correct'),
]
for onsets_key, cond_label in rt_cond_specs_dpp:
    subj_means_dpp = []
    for subj, runs in pupil_dict.items():
        epochs = []
        for run_data in runs.values():
            rt_onsets = run_data[onsets_key]
            if len(rt_onsets) == 0:
                continue
            events_rt = pd.DataFrame({'onset': rt_onsets})
            idx_rt    = pd.Series([True] * len(events_rt))
            eps = get_pupil_epoch(events_rt, idx_rt,
                                  run_data['t_neon'], run_data['pupil_d'],
                                  baseline_length=rt_pre_data,
                                  epoch_length=rt_post_data)
            for ep in eps:
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-rt_pre_data, rt_post_data, len(ep))
                dep = diff_epoch(ep, t_ep)
                bl  = np.nanmean(dep[t_ep < 0])
                epochs.append(dep - bl)
        if epochs:
            subj_means_dpp.append(np.nanmean(epochs, axis=0))
    if not subj_means_dpp:
        continue
    t_rt_dpp = np.linspace(-rt_pre_data, rt_post_data, len(subj_means_dpp[0]))
    arr  = np.array(subj_means_dpp)
    mean = smooth_t(arr.mean(axis=0), t_rt_dpp, smooth_win=smooth_win)
    sem  = smooth_t(arr.std(axis=0) / np.sqrt(len(arr)), t_rt_dpp, smooth_win=smooth_win)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='RT (time 0)')
    ax.plot(t_rt_dpp, mean, color='steelblue')
    ax.fill_between(t_rt_dpp, mean - sem, mean + sem, alpha=0.25,
                    color='steelblue', label='SEM')
    ax.set_xlabel('Time relative to RT (s)')
    ax.set_ylabel('dPupil/dt (mm/s)')
    ax.set_title(f'RT-locked pupil derivative — {cond_label} (cross-subject, N={n_subjects})')
    ax.legend(fontsize=8)
    ax.grid()
    plt.tight_layout()

# --- RT-locked VTC split (derivative) ---
rt_vtc_specs_dpp = [
    ('mnt_incorrect_rt_onsets', 'mnt_incorrect_vtc',
     {'high': 'darkorange', 'low': 'gold'}, 'mnt Incorrect'),
    ('city_correct_rt_onsets',  'city_correct_vtc',
     {'high': 'forestgreen', 'low': 'limegreen'}, 'city Correct'),
]
for onsets_key, vtc_key, colors_rt, cond_title in rt_vtc_specs_dpp:
    subj_mean_rt_vtc_dpp = {'high': [], 'low': []}
    for subj, runs in pupil_dict.items():
        subj_vtc_all = []
        for run_data in runs.values():
            subj_vtc_all.extend(run_data[vtc_key].tolist())
        if not subj_vtc_all:
            continue
        subj_median = np.median(subj_vtc_all)
        high_epochs, low_epochs = [], []
        for run_data in runs.values():
            rt_onsets = run_data[onsets_key]
            if len(rt_onsets) == 0:
                continue
            events_rt = pd.DataFrame({'onset': rt_onsets})
            idx_rt    = pd.Series([True] * len(events_rt))
            eps = get_pupil_epoch(events_rt, idx_rt,
                                  run_data['t_neon'], run_data['pupil_d'],
                                  baseline_length=rt_pre_data,
                                  epoch_length=rt_post_data)
            for ep, vtc_val in zip(eps, run_data[vtc_key]):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                t_ep = np.linspace(-rt_pre_data, rt_post_data, len(ep))
                dep = diff_epoch(ep, t_ep)
                bl  = np.nanmean(dep[t_ep < 0])
                ep_corr = dep - bl
                if vtc_val >= subj_median:
                    high_epochs.append(ep_corr)
                else:
                    low_epochs.append(ep_corr)
        if high_epochs:
            subj_mean_rt_vtc_dpp['high'].append(np.nanmean(high_epochs, axis=0))
        if low_epochs:
            subj_mean_rt_vtc_dpp['low'].append(np.nanmean(low_epochs, axis=0))
    if not any(subj_mean_rt_vtc_dpp.values()):
        continue
    t_rt_vtc_dpp = np.linspace(-rt_pre_data, rt_post_data,
                               len(next(v for v in subj_mean_rt_vtc_dpp.values() if v)[0]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='RT (time 0)')
    for split, ls in [('high', '-'), ('low', '--')]:
        arr = np.array(subj_mean_rt_vtc_dpp[split])
        if len(arr) == 0:
            continue
        mean = smooth_t(arr.mean(axis=0), t_rt_vtc_dpp, smooth_win=smooth_win)
        sem  = smooth_t(arr.std(axis=0) / np.sqrt(len(arr)), t_rt_vtc_dpp, smooth_win=smooth_win)
        c = colors_rt[split]
        ax.plot(t_rt_vtc_dpp, mean, color=c, linestyle=ls, label=f'{split} VTC')
        ax.fill_between(t_rt_vtc_dpp, mean - sem, mean + sem, alpha=0.25,
                        color=c, label='_nolegend_')
    ax.fill_between([], [], alpha=0.25, color='gray', label='SEM')
    ax.set_title(f'RT-locked pupil derivative by VTC — {cond_title} (cross-subject, N={n_subjects})')
    ax.set_xlabel('Time relative to RT (s)')
    ax.set_ylabel('dPupil/dt (mm/s)')
    ax.legend(fontsize=8)
    ax.grid()
    plt.tight_layout()

#%% Correlation between pupil size and VTC
# Re-read event files to get trial onsets; detrend the full-run pupil signal from
# pupil_dict without modifying it.  Results are stored in corr_dict.

_cond_idx_fns = {
    'mnt_correct':   lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] == 0),
    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),
    'city_correct':  lambda df: (df['trial_type'] == 'city') & (df['response_code'] != 0),
    'city_incorrect':lambda df: (df['trial_type'] == 'city') & (df['response_code'] == 0),
}
_cond_labels_corr = {
    'mnt_correct':   'mnt — Correct',
    'mnt_incorrect': 'mnt — Incorrect',
    'city_correct':  'city — Correct',
    'city_incorrect':'city — Incorrect',
}

corr_dict = {}   # corr_dict[subj][run_key][cond] = {'r': float, 'p': float, 'n': int}

for subj, runs in pupil_dict.items():
    subj_id   = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    corr_dict[subj] = {}
    for run_key, run_data in runs.items():
        run_id     = int(run_key.replace('run-', ''))
        event_file = os.path.join(
            subj_nirs,
            f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if not os.path.isfile(event_file):
            continue
        events_df  = pd.read_csv(event_file, sep='\t')
        vtc_full   = smoothing_VTC_gaussian_array(events_df['VTC'].values, L=20)

        t_neon         = run_data['t_neon']
        pupil_detrend  = detrend_run(run_data['pupil_d'])

        corr_dict[subj][run_key] = {}
        for cond, idx_fn in _cond_idx_fns.items():
            idx     = idx_fn(events_df)
            onsets  = events_df[idx]['onset'].values
            vtcs    = vtc_full[idx.values]
            if len(onsets) == 0:
                continue
            trial_means = []
            for onset in onsets:
                mask = (t_neon >= onset) & (t_neon < onset + epoch_length)
                ep   = pupil_detrend[mask]
                trial_means.append(
                    np.nanmean(ep) if (len(ep) > 0 and not np.all(np.isnan(ep)))
                    else np.nan)
            trial_means = np.array(trial_means)
            valid       = ~np.isnan(trial_means) & ~np.isnan(vtcs)
            if valid.sum() > 2:
                r, p = sp.stats.pearsonr(trial_means[valid], vtcs[valid])
                corr_dict[subj][run_key][cond] = {'r': r, 'p': p, 'n': int(valid.sum())}

# --- Plot: per-run r values for each condition ---
conditions_corr = list(_cond_idx_fns.keys())
rs_by_cond = {cond: [] for cond in conditions_corr}
for subj_runs in corr_dict.values():
    for run_res in subj_runs.values():
        for cond in conditions_corr:
            if cond in run_res:
                rs_by_cond[cond].append(run_res[cond]['r'])

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(conditions_corr))
for i, cond in enumerate(conditions_corr):
    rs = np.array(rs_by_cond[cond])
    if len(rs) == 0:
        continue
    ax.scatter(np.full(len(rs), i) + np.random.uniform(-0.15, 0.15, len(rs)),
               rs, alpha=0.5, s=20, zorder=3)
    mean_r = rs.mean()
    sem_r  = rs.std() / np.sqrt(len(rs))
    ax.errorbar(i, mean_r, yerr=sem_r, fmt='D', color='k', capsize=5, zorder=4)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([_cond_labels_corr[c] for c in conditions_corr],
                   rotation=15, ha='right')
ax.set_ylabel('Pearson r (detrended pupil vs VTC)')
n_subj_corr = sum(1 for runs in corr_dict.values() if runs)
ax.set_title(f'Correlation: full-run detrended pupil vs VTC '
             f'(per run, N={n_subj_corr} subjects)')
ax.grid(axis='y')
plt.tight_layout()

# %%
