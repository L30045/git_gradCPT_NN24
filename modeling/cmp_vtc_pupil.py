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
from utils import plt_multitaper
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from neon_to_bids_gradCPT_add_nodding import parseNeon_to_bids
from pupil_labs import neon_recording as nr
import re
from utils_eyetracking import preprocess_pupil, get_pupil_epoch, plot_pupil_epoch, \
    _build_gaussian_basis, _convolve_onsets
import warnings
from spectral_connectivity import Multitaper, Connectivity
import mne
import copy

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

#%% Detrend sample
_demo_subj   = 'sub-721'
_demo_run    = 2
detrend_order = 2
f_lowpass=30
f_downsample=60
_demo_subj_id = _demo_subj.replace('sub-', '')
_demo_nirs_dir = os.path.join(project_path, _demo_subj, 'nirs')
_demo_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', _demo_subj, 'eye_tracking')

_physio_file = os.path.join(_demo_nirs_dir,
    f"{_demo_subj}_task-gradCPT_run-{_demo_run:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
if not os.path.isfile(_physio_file):
    _physio_file = os.path.join(_demo_nirs_dir,
        f"{_demo_subj}_task-gradCPT_run-{_demo_run:02d}_recording-eyetracking_physio.tsv")

_neon_dirs = sorted([d for d in os.listdir(_demo_neon_dir) if re.match(r'\d{4}-', d)])
_neon_data = pd.read_csv(_physio_file, sep='\t')
_rec = nr.open(os.path.join(_demo_neon_dir, _neon_dirs[_demo_run - 1])) if _neon_dirs else None
t_neon = _neon_data['timestamps']           # in fNIRS time
pupil_d = (_neon_data['eyeleft_pupilDiameter'] + _neon_data['eyeright_pupilDiameter']) / 2
# remove blink periods (t_blink_start/stop and t_neon_ori are in Neon time)
pupil_d = pupil_d.values.copy().astype(float)
if _rec:
    print(f"NeonRecording is presented. Remove blink from pupil data.")
    t_neon_arr = _neon_data['time'].values # in Neon time. no offset
    try:
        t_blink_start = _rec.blinks["start_time"]
        t_blink_stop  = _rec.blinks["stop_time"]
        for t_start, t_stop in zip(t_blink_start, t_blink_stop):
            mask = (t_neon_arr >= t_start) & (t_neon_arr <= t_stop)
            pupil_d[mask] = np.nan
    except KeyError:
        print(f"Warning: blink data unavailable for this recording, skipping blink removal.")
else:
    print("No recording data present. Linear interpret missing data.")
# linear interpolation over blink periods
valid = ~np.isnan(pupil_d)
# check if there is too many missing data. If yes, return NaN
if np.sum(valid)/len(valid) < 0.7:
    print("Missing more than 30\% of data. Ignore the subject.")
pupil_d = np.interp(t_neon, t_neon[valid], pupil_d[valid])
# polynomial detrend (NaN-safe polyfit)
_t_idx = np.arange(len(pupil_d), dtype=float)
_valid = np.where(~np.isnan(pupil_d))[0]
_coef  = np.polyfit(_t_idx[_valid], pupil_d[_valid], detrend_order)
pupil_ori = pupil_d.copy()
pupil_d = pupil_d - np.polyval(_coef, _t_idx)
pupil_detrend = np.polyval(_coef, _t_idx)
# estimate sampling frequency from timestamps
t_neon_arr = t_neon.values
fs = 1.0 / np.median(np.diff(t_neon_arr))
# lowpass filter
sos = sp.signal.butter(4, f_lowpass, btype='low', fs=fs, output='sos')
pupil_d = sp.signal.sosfiltfilt(sos, pupil_d)
pupil_ori = sp.signal.sosfiltfilt(sos, pupil_ori)
pupil_detrend = sp.signal.sosfiltfilt(sos, pupil_detrend)
# downsample
t_new = np.arange(t_neon_arr[0], t_neon_arr[-1], 1.0 / f_downsample)
pupil_d = np.interp(t_new, t_neon_arr, pupil_d)
pupil_ori = np.interp(t_new, t_neon_arr, pupil_ori)
pupil_detrend = np.interp(t_new, t_neon_arr, pupil_detrend)
t_neon = t_new
_t = t_neon
_x = pupil_ori

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(_t, _x,     color='steelblue', linewidth=0.6, label='Original')
axes[0].plot(_t, pupil_detrend, color='crimson',   linewidth=1.5, linestyle='--', label='Quadratic fit')
axes[0].set_ylabel('Pupil diameter (mm)')
axes[0].set_title(f'Detrend demo — {_demo_subj} run-{_demo_run:02d}: original + fitted polyline')
axes[0].legend(fontsize=9)
axes[0].grid()

axes[1].plot(_t, pupil_d, color='steelblue', linewidth=0.6, label='Detrended')
axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[1].set_ylabel('Pupil diameter (mm)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('After quadratic detrending')
axes[1].legend(fontsize=9)
axes[1].grid()
plt.tight_layout()

# GLM regression demo: remove phasic components and visualize
_event_file = os.path.join(_demo_nirs_dir,
    f"{_demo_subj}_task-gradCPT_run-{_demo_run:02d}_events.tsv")
_events_df = pd.read_csv(_event_file, sep='\t')

_basis, _t_hrf = _build_gaussian_basis(f_downsample,
                                        t_pre=2.0, t_post=10.0,
                                        t_delta=1.0, t_std=1.0)
_stim_conds = {
    'mnt_correct':    (_events_df['trial_type'] == 'mnt') & (_events_df['response_code'] == 0),
    'mnt_incorrect':  (_events_df['trial_type'] == 'mnt') & (_events_df['response_code'] != 0),
    'city_correct':   (_events_df['trial_type'] == 'city') & (_events_df['response_code'] != 0),
    'city_incorrect': (_events_df['trial_type'] == 'city') & (_events_df['response_code'] == 0),
}
_resp_conds = {
    'mnt_incorrect':  (_events_df['trial_type'] == 'mnt') & (_events_df['response_code'] != 0),
    'city_correct':   (_events_df['trial_type'] == 'city') & (_events_df['response_code'] != 0),
    'city_incorrect': (_events_df['trial_type'] == 'city') & (_events_df['response_code'] == 0),
}
_X_blocks = []
for _idx in _stim_conds.values():
    _onsets = _events_df[_idx]['onset'].values
    if len(_onsets) > 0:
        _X_blocks.append(_convolve_onsets(_onsets, t_neon, _basis, _t_hrf, f_downsample))
for _idx in _resp_conds.values():
    _sub = _events_df[_idx]
    _rt_onsets = (_sub['onset'] + _sub['reaction_time']).values
    if len(_rt_onsets) > 0:
        _X_blocks.append(_convolve_onsets(_rt_onsets, t_neon, _basis, _t_hrf, f_downsample))
_X = np.hstack(_X_blocks)
_X_int = np.hstack([_X, np.ones((len(pupil_d), 1))])
_beta, _, _, _ = np.linalg.lstsq(_X_int, pupil_d, rcond=None)
_pupil_phasic   = _X @ _beta[:-1]
_pupil_residual = pupil_d - _pupil_phasic

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(t_neon, pupil_d, color='steelblue', linewidth=0.6, label='Detrended')
axes[0].plot(t_neon, _pupil_phasic, color='crimson', linewidth=1.0, label='GLM fit (phasic)')
axes[0].set_ylabel('Pupil diameter (mm)')
axes[0].set_title(f'GLM regression demo — {_demo_subj} run-{_demo_run:02d}: detrended + phasic fit')
axes[0].legend(fontsize=9)
axes[0].grid()

axes[1].plot(t_neon, _pupil_phasic, color='crimson', linewidth=0.8)
axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[1].set_ylabel('Pupil diameter (mm)')
axes[1].set_title('Phasic component (GLM fit)')
axes[1].grid()

axes[2].plot(t_neon, _pupil_residual, color='steelblue', linewidth=0.6)
axes[2].axhline(0, color='k', linewidth=0.8, linestyle='--')
axes[2].set_ylabel('Pupil diameter (mm)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('After phasic regression')
axes[2].grid()
plt.tight_layout()

#%% cross subjects epoch analysis
dirs = os.listdir(project_path)
subject_list = [d for d in dirs if 'sub' in d] # and d not in excluded]

epoch_length = 12 # sec (fade-in + fade-out)
baseline_length = 12 # sec (previous event's fade-out phase)
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
        # load events
        events_df_s = pd.read_csv(event_file, sep='\t')
        mnt_correct_idx_s   = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']==0)
        mnt_incorrect_idx_s = (events_df_s['trial_type']=='mnt') & (events_df_s['response_code']!=0)
        city_correct_idx_s  = (events_df_s['trial_type']=='city') & (events_df_s['response_code']!=0)
        city_incorrect_idx_s= (events_df_s['trial_type']=='city') & (events_df_s['response_code']==0)
        vtc_s = smoothing_VTC_gaussian_array(events_df_s['VTC'].values, L=20)
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
        # preprocess pupil
        t_neon_s, pupil_d_s = preprocess_pupil(neon_data, rec=rec, detrend_order=2, is_rm_phasic=False, events_df=events_df_s)
        # check if subject missing too many data
        if t_neon_s is None:
            print(f"Missing too many data. Skip sub-{subj}")
            continue
        sfreq_neon = np.round(np.median(1 / np.diff(t_neon_s)))
        print(f"{subj} run-{run_id:02d}: sfreq_neon = {sfreq_neon} Hz")
        win_epoch    = int(np.round(sfreq_neon * epoch_length))
        win_baseline = int(np.round(sfreq_neon * baseline_length))
        t_epoch_run  = np.linspace(-baseline_length, epoch_length, win_baseline + win_epoch)
        pupil_dict[subj][f'run-{run_id:02d}'] = {
            'mnt_correct':                  get_pupil_epoch(events_df_s, mnt_correct_idx_s,   t_neon_s, pupil_d_s, baseline_length=baseline_length, epoch_length=epoch_length),
            'mnt_incorrect':                get_pupil_epoch(events_df_s, mnt_incorrect_idx_s, t_neon_s, pupil_d_s, baseline_length=baseline_length, epoch_length=epoch_length),
            'city_correct':                 get_pupil_epoch(events_df_s, city_correct_idx_s,  t_neon_s, pupil_d_s, baseline_length=baseline_length, epoch_length=epoch_length),
            'city_incorrect':               get_pupil_epoch(events_df_s, city_incorrect_idx_s,t_neon_s, pupil_d_s, baseline_length=baseline_length, epoch_length=epoch_length),
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
            'baseline_length':              baseline_length,
            'epoch_length':                 epoch_length,
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
                t_ep = np.linspace(-run_data['baseline_length'], run_data['epoch_length'], len(ep))
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
                t_ep = np.linspace(-run_data['baseline_length'], run_data['epoch_length'], len(ep))
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
                t_ep = np.linspace(-run_data['baseline_length'], run_data['epoch_length'], len(ep))
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
                t_ep = np.linspace(-run_data['baseline_length'], run_data['epoch_length'], len(ep))
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
                t_ep = np.linspace(-run_data['baseline_length'], run_data['epoch_length'], len(ep))
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


# %% Time-frequency analysis
time_halfbandwidth_product = 3
time_window_duration = 4  # sec
time_window_step = 1

conditions_tf = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']

# Per-subject VTC median across all runs and conditions (in-zone = VTC < median)
subj_vtc_median = {}
for _subj, _runs in pupil_dict.items():
    _all_vtc = []
    for _rd in _runs.values():
        for _cond in conditions_tf:
            _arr = _rd.get(f'{_cond}_vtc', np.array([]))
            _all_vtc.extend(np.asarray(_arr).tolist())
    subj_vtc_median[_subj] = np.median(_all_vtc) if _all_vtc else np.nan

power_dict = {}
for subj_tf, runs_tf in pupil_dict.items():
    power_dict[subj_tf] = {}
    vtc_threshold = subj_vtc_median[subj_tf]
    for run_key, run_data_tf in runs_tf.items():
        power_dict[subj_tf][run_key] = {}
        sfreq_tf = run_data_tf['sfreq_neon']
        for cond_tf in conditions_tf:
            epochs_tf = run_data_tf.get(cond_tf, [])
            vtc_tf = run_data_tf.get(f'{cond_tf}_vtc', np.full(len(epochs_tf), np.nan))
            valid_epochs = []
            is_in_zone = []
            for ep, vtc_v in zip(epochs_tf, vtc_tf):
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                ep = np.where(np.isnan(ep), 0.0, ep)
                valid_epochs.append(ep)
                is_in_zone.append(1 if vtc_v < vtc_threshold else 0)
            if not valid_epochs:
                continue

            is_in_zone_arr = np.array(is_in_zone)
            in_epochs  = [ep for ep, z in zip(valid_epochs, is_in_zone_arr) if z == 1]
            out_epochs = [ep for ep, z in zip(valid_epochs, is_in_zone_arr) if z == 0]

            def _run_tf(epoch_list):
                data = np.array(epoch_list)[:, np.newaxis, :]
                info = mne.create_info(ch_names=['pupil'], sfreq=sfreq_tf, ch_types=['misc'])
                ep_mne = mne.EpochsArray(data, info, tmin=-baseline_length, verbose=False)
                lp, mt, conn = plt_multitaper(
                    ep_mne,
                    time_halfbandwidth_product=time_halfbandwidth_product,
                    time_window_duration=time_window_duration,
                    time_window_step=time_window_step,
                    ratio_to='baseline',
                    vis_f_range=[0, 15],
                    is_plot=False)
                return lp, mt, conn, ep_mne

            # Stack into MNE EpochsArray: (n_epochs, 1, n_times)
            data_tf = np.array(valid_epochs)[:, np.newaxis, :]
            info_tf = mne.create_info(ch_names=['pupil'], sfreq=sfreq_tf, ch_types=['misc'])
            epochs_mne = mne.EpochsArray(data_tf, info_tf, tmin=-baseline_length, verbose=False)

            print(f"\n--- TF: {subj_tf} {run_key} {cond_tf} ({len(valid_epochs)} epochs) ---")
            (log_power, multitaper, connectivity) = plt_multitaper(
                epochs_mne,
                time_halfbandwidth_product=time_halfbandwidth_product,
                time_window_duration=time_window_duration,
                time_window_step=time_window_step,
                ratio_to='baseline',
                vis_f_range=[0, 15],
                is_plot=False)
            vis_f_range = [0, 15]
            vis_mask = (connectivity.frequencies >= vis_f_range[0]) & (connectivity.frequencies <= vis_f_range[1])
            vis_f = connectivity.frequencies[vis_mask]
            time_vector = epochs_mne.times
            multitaper_time = multitaper.time + time_vector[0]

            log_power_in  = _run_tf(in_epochs)[0]  if in_epochs  else None
            log_power_out = _run_tf(out_epochs)[0] if out_epochs else None

            power_dict[subj_tf][run_key][cond_tf] = {
                'log_power': log_power,
                'log_power_in': log_power_in,
                'log_power_out': log_power_out,
                'is_in_zone': is_in_zone_arr,
                'vis_f_range': vis_f_range,
                'vis_mask': vis_mask,
                'vis_f': vis_f,
                'time_vector': time_vector,
                'multitaper_time': multitaper_time,
            }

#%% visualize tf results across subjects by averageing 
def _vis_tf(plt_power, vis_mask, multitaper_time, vis_f, title='ERSP'):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    # Plot log power spectrogram
    extent = [multitaper_time[0], multitaper_time[-1], 0, np.sum(vis_mask)]
    vmax = np.abs(plt_power).max()
    im = ax1.imshow(plt_power[:,vis_mask].T, aspect='auto', origin='lower', cmap='RdBu_r', extent=extent, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax1, label='Log Power')
    ax1.set_ylabel('Frequency')
    ax1.set_yticks(np.arange(np.sum(vis_mask)))
    ax1.set_yticklabels(vis_f)
    ax1.set_title(title)
    ax1.axvline(0, color='white', linestyle='--', linewidth=1)
    plt.show()


# mnt_correct − city_correct: averaged across subjects and runs
# diff_list = []
# vis_mask_ref = vis_f_ref = multitaper_time_ref = None
# for subj, runs in power_dict.items():
#     for run_key, run in runs.items():
#         if 'mnt_correct' not in run or 'city_correct' not in run:
#             continue
#         lp_mnt  = run['mnt_correct']['log_power']
#         lp_city = run['city_correct']['log_power']
#         if lp_mnt.shape != lp_city.shape:
#             continue
#         diff_list.append(lp_mnt - lp_city)
#         if vis_mask_ref is None:
#             vis_mask_ref        = run['mnt_correct']['vis_mask']
#             vis_f_ref           = run['mnt_correct']['vis_f']
#             multitaper_time_ref = run['mnt_correct']['multitaper_time']
# if diff_list:
#     mean_diff = np.nanmean(diff_list, axis=0)
#     n_pairs = len(diff_list)
#     _vis_tf(mean_diff, vis_mask_ref, multitaper_time_ref, vis_f_ref,
#             title=f'mnt_correct − city_correct (N={n_pairs} subject-runs)')

# in-zone − out-of-zone for each condition: 3-panel plot per condition
for cond_tf in conditions_tf:
    # average runs within each subject → one diff per subject for valid t-test
    subj_diff_list = []
    vis_mask_ref = vis_f_ref = multitaper_time_ref = None
    for subj, runs in power_dict.items():
        run_diffs = []
        for run_key, run in runs.items():
            if cond_tf not in run:
                continue
            lp_in  = run[cond_tf]['log_power_in']
            lp_out = run[cond_tf]['log_power_out']
            if lp_in is None or lp_out is None:
                continue
            if lp_in.shape != lp_out.shape:
                continue
            run_diffs.append(lp_in - lp_out)
            if vis_mask_ref is None:
                vis_mask_ref        = run[cond_tf]['vis_mask']
                vis_f_ref           = run[cond_tf]['vis_f']
                multitaper_time_ref = run[cond_tf]['multitaper_time']
        if run_diffs:
            subj_diff_list.append(np.nanmean(run_diffs, axis=0))
    if not subj_diff_list or vis_mask_ref is None:
        continue

    n_subj = len(subj_diff_list)
    subj_diff_arr = np.array(subj_diff_list)           # (n_subj, n_time, n_freq)
    mean_diff     = np.nanmean(subj_diff_arr, axis=0)  # (n_time, n_freq)

    # one-sample t-test at each TF point (H0: mean diff = 0)
    t_stat, p_val = sp.stats.ttest_1samp(subj_diff_arr, popmean=0, axis=0)

    # restrict to visible frequency band
    mean_diff_vis = mean_diff[:, vis_mask_ref]          # (n_time, n_vis_freq)
    t_vis         = t_stat[:, vis_mask_ref]
    p_vis         = p_val[:, vis_mask_ref]

    # FDR correction (Benjamini-Hochberg) over all visible TF points
    reject_flat, p_flat = mne.stats.fdr_correction(p_vis.ravel(), alpha=0.05)
    sig = reject_flat.reshape(p_vis.shape)              # (n_time, n_vis_freq)
    mean_diff_sig = np.where(sig, mean_diff_vis, np.nan)



    # --- 3-panel figure ---
    extent    = [multitaper_time_ref[0], multitaper_time_ref[-1], 0, np.sum(vis_mask_ref)]
    n_vis_f   = int(np.sum(vis_mask_ref))
    ytick_idx = np.arange(0, n_vis_f, max(1, n_vis_f // 8))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    def _imshow_tf(ax, data, cmap='RdBu_r', label='', vmax=None):
        vmax = vmax or (np.nanmax(np.abs(data)) or 1)
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                       extent=extent, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels(np.round(vis_f_ref[ytick_idx], 1))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.axvline(0, color='white', linestyle='--', linewidth=1)
        return im

    _imshow_tf(axes[0], mean_diff_vis, label='Log Power diff')
    axes[0].set_title('in-zone − out-of-zone (mean)')

    p_vis_masked = np.where(p_vis <= 0.05, p_vis, np.nan)
    cmap_p = plt.cm.hot.copy()
    cmap_p.set_bad(color='white')
    im2 = axes[1].imshow(p_vis_masked.T, aspect='auto', origin='lower', cmap=cmap_p,
                         extent=extent, vmin=0, vmax=0.05)
    plt.colorbar(im2, ax=axes[1], label='p-value')
    axes[1].set_yticks(ytick_idx)
    axes[1].set_yticklabels(np.round(vis_f_ref[ytick_idx], 1))
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].axvline(0, color='white', linestyle='--', linewidth=1)
    axes[1].set_title(f'p-value ≤ 0.05 (N={n_subj} subjects)')

    _imshow_tf(axes[2], mean_diff_sig, label='Log Power diff')
    axes[2].set_title('Significant TF (FDR p<0.05)')

    fig.suptitle(f'in-zone − out-of-zone | {cond_tf}')
    plt.show()

    # p-value distribution plots
    fig2, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    p_vis_sorted = np.sort(p_vis.flatten())[::-1]
    ax_p1.bar(np.arange(len(p_vis_sorted)), p_vis_sorted, width=1.0, color='steelblue', linewidth=0)
    ax_p1.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax_p1.set_xlabel('TF point rank')
    ax_p1.set_ylabel('p-value (uncorrected)')
    ax_p1.set_title(f'Uncorrected p-values | {cond_tf}')
    ax_p1.legend()

    p_flat_sorted = np.sort(p_flat)[::-1]
    ax_p2.bar(np.arange(len(p_flat_sorted)), p_flat_sorted, width=1.0, color='steelblue', linewidth=0)
    ax_p2.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='p = 0.05')
    ax_p2.set_xlabel('TF point rank')
    ax_p2.set_ylabel('p-value (FDR corrected)')
    ax_p2.set_title(f'FDR-corrected p-values | {cond_tf}')
    ax_p2.legend()

    plt.show()

#%% Time-frequency analysis (Whole Run)
time_halfbandwidth_product = 3
time_window_duration = 4
time_window_step = 1

subj_wr    = next(iter(pupil_dict))
run_key_wr = next(iter(pupil_dict[subj_wr]))
run_data_wr = pupil_dict[subj_wr][run_key_wr]

pupil_run  = run_data_wr['pupil_d']
t_run      = run_data_wr['t_neon']
sfreq_run  = run_data_wr['sfreq_neon']
vtc_thr_wr = subj_vtc_median[subj_wr]

# reload events for onset times + VTC
subj_id_wr   = subj_wr.replace('sub-', '')
run_id_wr    = int(run_key_wr.replace('run-', ''))
event_file_wr = os.path.join(project_path, subj_wr, 'nirs',
    f"sub-{subj_id_wr}_task-gradCPT_run-{run_id_wr:02d}_events.tsv")
events_wr  = pd.read_csv(event_file_wr, sep='\t')
vtc_wr     = smoothing_VTC_gaussian_array(events_wr['VTC'].values, L=20)
onset_wr   = events_wr['onset'].values

# multitaper on the full run signal — shape (n_times, 1, 1)
run_mt_data = np.expand_dims(pupil_run[:, np.newaxis], axis=-1)
mt_run = Multitaper(
    run_mt_data,
    sampling_frequency=sfreq_run,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=time_window_duration,
    time_window_step=time_window_step,
    detrend_type='linear'
)
conn_run      = Connectivity.from_multitaper(mt_run, expectation_type="trials_tapers")
log_power_run = np.log10(np.squeeze(conn_run.power()))   # (n_time_windows, n_freqs)
mt_time_run   = mt_run.time + t_run[0]                   # shift to run start time

vis_mask_wr = (conn_run.frequencies >= 0) & (conn_run.frequencies <= 15)
vis_f_wr    = conn_run.frequencies[vis_mask_wr]
n_vis_wr    = int(np.sum(vis_mask_wr))
lp_vis_wr   = log_power_run[:, vis_mask_wr]
lp_vis_wr   = lp_vis_wr - np.mean(lp_vis_wr) # ratio to the mean power

fig, (ax_tf, ax_vtc) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

# subplot 1: TF spectrogram
extent_wr = [mt_time_run[0], mt_time_run[-1], 0, n_vis_wr]
vmax_wr   = np.nanmax(np.abs(lp_vis_wr))
im_wr = ax_tf.imshow(lp_vis_wr.T, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=extent_wr, vmin=-vmax_wr, vmax=vmax_wr)
plt.colorbar(im_wr, ax=ax_tf, label='Log Power')
ytick_wr = np.arange(0, n_vis_wr, max(1, n_vis_wr // 8))
ax_tf.set_yticks(ytick_wr)
ax_tf.set_yticklabels(np.round(vis_f_wr[ytick_wr], 1))
ax_tf.set_ylabel('Frequency (Hz)')
ax_tf.set_title(f'Time-Frequency Power — {subj_wr} {run_key_wr}')
ax_tf.grid(axis='x', color='black', linestyle='--', linewidth=0.5, alpha=0.7)

# subplot 2: VTC colored by in-zone / out-of-zone
ax_vtc.plot(onset_wr, vtc_wr, color='black', linewidth=0.8)
ax_vtc.fill_between(onset_wr, vtc_wr, vtc_thr_wr,
                    where=(vtc_wr >= vtc_thr_wr), color='red',  alpha=0.5, label='out-of-zone')
ax_vtc.fill_between(onset_wr, vtc_wr, vtc_thr_wr,
                    where=(vtc_wr <  vtc_thr_wr), color='blue', alpha=0.5, label='in-zone')
ax_vtc.axhline(vtc_thr_wr, color='gray', linestyle='--', linewidth=1.5, label='threshold')
ax_vtc.set_ylabel('VTC')
ax_vtc.set_xlabel('Time (s)')
ax_vtc.set_title('VTC vs Time')
ax_vtc.grid(axis='x', linestyle='--', color='black',linewidth=0.5, alpha=0.7)
ax_vtc.legend(fontsize=8)

plt.show()


#%% spectrum
conditions_spec = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']

# subj_mean_spec[cond] = list of per-subject mean log power, one array per subject
subj_mean_spec = {cond: [] for cond in conditions_spec}
freqs_spec = None

for subj, runs in pupil_dict.items():
    for cond in conditions_spec:
        epoch_powers = []
        for run_data in runs.values():
            sfreq_spec = run_data['sfreq_neon']
            for ep in run_data[cond]:
                ep = np.array(ep, dtype=float)
                if np.all(np.isnan(ep)):
                    continue
                ep = np.where(np.isnan(ep), 0.0, ep)
                f = np.fft.rfftfreq(len(ep), d=1.0 / sfreq_spec)
                p = np.abs(np.fft.rfft(ep)) ** 2
                if freqs_spec is None:
                    freqs_spec = f
                if len(p) == len(freqs_spec):
                    epoch_powers.append(np.log10(p + 1e-30))
        if epoch_powers:
            subj_mean_spec[cond].append(np.mean(epoch_powers, axis=0))

# Plot
vis_f_mask = freqs_spec <= 15
colors_spec = {
    'mnt_correct':   'royalblue',
    'mnt_incorrect': 'darkorange',
    'city_correct':  'forestgreen',
    'city_incorrect':'crimson',
}

fig, ax = plt.subplots(figsize=(8, 4))
for cond in conditions_spec:
    arr = np.array(subj_mean_spec[cond])
    if len(arr) == 0:
        continue
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0) / np.sqrt(len(arr))
    c = colors_spec[cond]
    ax.plot(freqs_spec[vis_f_mask], mean[vis_f_mask], color=c,
            label=cond.replace('_', ' '))
    ax.fill_between(freqs_spec[vis_f_mask],
                    (mean - sem)[vis_f_mask],
                    (mean + sem)[vis_f_mask],
                    alpha=0.25, color=c)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Log Power')
ax.set_title(f'Pupil power spectrum (cross-subject mean ± SEM, N={len(pupil_dict)})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Difference plots: mnt_correct - city_correct and mnt_incorrect - city_correct
# Computed per-subject then averaged
diff_specs = {
    'mnt_correct - city_correct':   ('mnt_correct',   'city_correct'),
    'mnt_incorrect - city_correct': ('mnt_incorrect', 'city_correct'),
}

diff_colors = {
    'mnt_correct - city_correct':   'royalblue',
    'mnt_incorrect - city_correct': 'darkorange',
}

fig, ax = plt.subplots(figsize=(8, 4))
for diff_label, (cond_a, cond_b) in diff_specs.items():
    arr_a = np.array(subj_mean_spec[cond_a])
    arr_b = np.array(subj_mean_spec[cond_b])
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        continue
    diff = arr_a[:n] - arr_b[:n]
    mean = diff.mean(axis=0)
    sem  = diff.std(axis=0) / np.sqrt(n)
    c = diff_colors[diff_label]
    ax.plot(freqs_spec[vis_f_mask], mean[vis_f_mask], color=c, label=diff_label)
    ax.fill_between(freqs_spec[vis_f_mask],
                    (mean - sem)[vis_f_mask],
                    (mean + sem)[vis_f_mask],
                    alpha=0.25, color=c)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Δ Log Power')
ax.set_title(f'Pupil spectrum difference (cross-subject mean ± SEM, N={n})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()

#%% city_correct spectrum: in-zone vs out-of-zone
# Compute per-subject VTC median (in-zone = VTC < median)
_all_conditions_vtc = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']
_subj_vtc_median = {}
for _subj, _runs in pupil_dict.items():
    _vtc_vals = []
    for _rd in _runs.values():
        for _cond in _all_conditions_vtc:
            _vtc_vals.extend(np.asarray(_rd.get(f'{_cond}_vtc', [])).tolist())
    _subj_vtc_median[_subj] = np.median(_vtc_vals) if _vtc_vals else np.nan

city_zone_spec = {'in_zone': [], 'out_of_zone': []}
freqs_city_zone = None

for subj, runs in pupil_dict.items():
    vtc_thr_subj = _subj_vtc_median.get(subj, np.nan)
    in_powers, out_powers = [], []
    for run_data in runs.values():
        sfreq_s = run_data['sfreq_neon']
        epochs_cc  = run_data.get('city_correct', [])
        vtc_cc     = run_data.get('city_correct_vtc', np.full(len(epochs_cc), np.nan))
        for ep, vtc_v in zip(epochs_cc, vtc_cc):
            ep = np.array(ep, dtype=float)
            if np.all(np.isnan(ep)):
                continue
            ep = np.where(np.isnan(ep), 0.0, ep)
            f   = np.fft.rfftfreq(len(ep), d=1.0 / sfreq_s)
            lp  = np.log10(np.abs(np.fft.rfft(ep)) ** 2 + 1e-30)
            if freqs_city_zone is None:
                freqs_city_zone = f
            if len(lp) != len(freqs_city_zone):
                continue
            if vtc_v < vtc_thr_subj:
                in_powers.append(lp)
            else:
                out_powers.append(lp)
    if in_powers:
        city_zone_spec['in_zone'].append(np.mean(in_powers, axis=0))
    if out_powers:
        city_zone_spec['out_of_zone'].append(np.mean(out_powers, axis=0))

vis_mask_cz = (freqs_city_zone >= 0) & (freqs_city_zone <= 6)
colors_cz = {'in_zone': 'royalblue', 'out_of_zone': 'crimson'}
labels_cz  = {'in_zone': 'in-zone',  'out_of_zone': 'out-of-zone'}

# Detect peak as frequency with largest absolute difference (in-zone minus out-of-zone)
_mean_in  = np.array(city_zone_spec['in_zone']).mean(axis=0)
_mean_out = np.array(city_zone_spec['out_of_zone']).mean(axis=0)
_diff_in = np.diff(_mean_in)
_diff_out = np.diff(_mean_out)
_peak_idx  = np.argmax(np.abs(_diff_in[vis_mask_cz[1:]]))
_peak_freq = freqs_city_zone[vis_mask_cz][_peak_idx]

fig, ax = plt.subplots(figsize=(8, 4))
for zone, arr_list in city_zone_spec.items():
    arr = np.array(arr_list)
    if len(arr) == 0:
        continue
    mean_cz = arr.mean(axis=0)
    sem_cz  = arr.std(axis=0) / np.sqrt(len(arr))
    c = colors_cz[zone]
    ax.plot(freqs_city_zone[vis_mask_cz], mean_cz[vis_mask_cz],
            color=c, label=f'{labels_cz[zone]} (N={len(arr)})')
    ax.fill_between(freqs_city_zone[vis_mask_cz],
                    (mean_cz - sem_cz)[vis_mask_cz],
                    (mean_cz + sem_cz)[vis_mask_cz],
                    alpha=0.25, color=c)
ax.axvline(_peak_freq, color='gray', linestyle='--', linewidth=1.2,
           label=f'{_peak_freq:.2f} Hz (peak)')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Log Power')
ax.set_title('city_correct pupil spectrum: in-zone vs out-of-zone (cross-subject mean ± SEM)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
