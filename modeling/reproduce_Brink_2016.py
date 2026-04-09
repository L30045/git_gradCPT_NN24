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
from pupil_labs import neon_recording as nr
import re
from utils_eyetracking import preprocess_pupil, _build_gaussian_basis, _convolve_onsets
import warnings

#%% Reproduce Brink 2016 results
"""
Definitions of the behavioral performance measures:
1. False alarm rate — proportion of mountain trials where the participant incorrectly pressed the space bar. This is the primary lapse-of-attention measure in CPT paradigms.
2. Slow quintile RT — proportion of trials whose RT fell within the slowest 20% (quintile) of RTs within that block.
3. Mean RT — average response time across city trials in a given window, measured relative to stimulus onset (so an RT of 640 ms = responding when the image was 80% city / 20% previous image).
4. RT coefficient of variation (RTCV) — the standard deviation of RT divided by the block mean RT. This captures variability in responding, independent of the overall RT level.

For each run:
1. Preprocess raw pupil data — blink interpolation, low-pass filter at 6 Hz, regress out phasic dilations. (utils_eyetracking.py>preprocess_pupil. detrend_order=None)
2. Apply sliding window (50-trial width, 15-trial steps) to the residual tonic pupil time series → produces one mean diameter value (and one derivative value) per window position.
NOTE: when creating the sliding window function, set trial width and step size as input.
3. Z-score the resulting window-level time series — the paper says they "Z-scored the time series" before fitting the regression lines (this is stated in the Results section under "Performance decrements with time-on-task," and the same procedure is applied to both behavioral and pupil measures).
4. Remove time-on-task effect using LINEAR regression
5. Fit regression (quadratic) between the z-scored pupil time series and z-scored behavioral time series.

After the sliding window, you have a time series of (pupil value, behavioral value) pairs — one per window position, within each participant and block. These are already on the same time grid, which is why pairing is straightforward.
Z-score both the pupil and behavioral time series within participant and block.
Aggregate the paired (pupil, behavior) observations across participants.
Sort all observations by pupil value, then divide into 30 equal bins.
For each bin, compute the mean behavioral value of the observations that fell into that bin.
"""

#%% subject list
dirs = os.listdir(project_path)
subject_list = sorted([d for d in dirs if 'sub' in d])

#%% functions for calculating performance
# Use the same sliding window as in preprocessing pupil time series
# (50-trial width, 15-trial steps)

def sliding_window_trials(events_df, pupil_d, t_pupil,
                          win_trials=50, step_trials=15):
    """
    Apply a sliding window over trials and compute per-window:
      - mean tonic pupil diameter (sampled at each trial onset)
      - 4 behavioral performance measures

    Parameters
    ----------
    events_df : pd.DataFrame
        BIDS events TSV with columns: onset, trial_type, response_code,
        reaction_time.
    pupil_d : np.ndarray
        Tonic pupil time series (after phasic regression), length = len(t_pupil).
    t_pupil : np.ndarray
        Time axis (seconds) corresponding to pupil_d.
    win_trials : int
        Number of trials per window. Default 50.
    step_trials : int
        Step size in trials between consecutive windows. Default 15.

    Returns
    -------
    dict with keys:
        'pupil_mean'  : mean tonic pupil per window (z-scored later)
        'fa_rate'     : false alarm rate (mnt trials pressed / total mnt trials)
        'slow_q_rt'   : slow quintile RT proportion (city trials)
        'mean_rt'     : mean RT of city correct trials
        'rtcv'        : RT coefficient of variation (city correct trials)
        'win_center'  : trial index of window centre (for reference)
    """
    trials = events_df.reset_index(drop=True)
    n_trials = len(trials)
    starts = np.arange(0, n_trials - win_trials + 1, step_trials)

    pupil_mean = []
    fa_rate    = []
    slow_q_rt  = []
    mean_rt    = []
    rtcv       = []
    win_center = []

    # block-level RT quintile threshold (city correct = pressed)
    city_correct = trials[(trials['trial_type'] == 'city') &
                          (trials['response_code'] == 1)]['reaction_time'].values
    rt_q80 = np.percentile(city_correct, 80) if len(city_correct) > 0 else np.nan

    for start in starts:
        win = trials.iloc[start: start + win_trials]

        # --- tonic pupil: mean pupil sampled at each trial onset ---
        onsets = win['onset'].values
        pupil_samples = []
        for onset in onsets:
            idx = np.argmin(np.abs(t_pupil - onset))
            pupil_samples.append(pupil_d[idx])
        pupil_mean.append(np.nanmean(pupil_samples))

        # --- 1. false alarm rate: mnt trials where space was pressed ---
        mnt = win[win['trial_type'] == 'mnt']
        if len(mnt) > 0:
            fa_rate.append(np.mean(mnt['response_code'] != 0))
        else:
            fa_rate.append(np.nan)

        # --- city correct trials in window ---
        city_win = win[(win['trial_type'] == 'city') &
                       (win['response_code'] == 1)]
        rts = city_win['reaction_time'].values

        # --- 2. slow quintile RT: proportion of city RTs above block q80 ---
        if len(rts) > 0 and not np.isnan(rt_q80):
            slow_q_rt.append(np.mean(rts > rt_q80))
        else:
            slow_q_rt.append(np.nan)

        # --- 3. mean RT ---
        mean_rt.append(np.nanmean(rts) if len(rts) > 0 else np.nan)

        # --- 4. RT coefficient of variation ---
        if len(rts) > 1 and np.nanmean(rts) > 0:
            rtcv.append(np.nanstd(rts) / np.nanmean(rts))
        else:
            rtcv.append(np.nan)

        win_center.append(start + win_trials // 2)

    return {
        'pupil_mean':  np.array(pupil_mean),
        'fa_rate':     np.array(fa_rate),
        'slow_q_rt':   np.array(slow_q_rt),
        'mean_rt':     np.array(mean_rt),
        'rtcv':        np.array(rtcv),
        'win_center':  np.array(win_center),
    }


def zscore_safe(x):
    """Z-score a 1D array, ignoring NaN. Returns NaN-filled array if std=0."""
    x = np.array(x, dtype=float)
    std = np.nanstd(x)
    if std == 0 or np.isnan(std):
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / std


#%% For each run, get z-scored pupil time series and 4 performance measures
f_lowpass    = 6      # Hz — Brink 2016 uses 6 Hz low-pass for tonic pupil
f_downsample = 60     # Hz
fit_order = 2         # For regression between pupil and measure
detrend_order = 1     # For removing time-on-task effect

perf_keys = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv']

# all_obs[perf_key] accumulates (pupil_z, behavior_z) pairs across all subj/runs
all_obs = {k: {'pupil': [], 'behavior': []} for k in perf_keys}
included_subjects = set()

for subj in subject_list:
    subj_id       = subj.replace('sub-', '')
    subj_nirs_dir = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    if not os.path.isdir(subj_nirs_dir) or not os.path.isdir(subj_neon_dir):
        continue
    neon_dirs_subj = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])

    run_ids = sorted(set(
        int(m.group(1))
        for f in os.listdir(subj_nirs_dir)
        for m in [re.search(r'task-gradCPT_run-(\d+)_events\.tsv$', f)]
        if m
    ))
    for run_id in run_ids:
        physio_file = os.path.join(subj_nirs_dir,
            f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
        if not os.path.isfile(physio_file):
            physio_file = os.path.join(subj_nirs_dir,
                f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
        event_file = os.path.join(subj_nirs_dir,
            f"{subj}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if not os.path.isfile(physio_file) or not os.path.isfile(event_file):
            continue

        neon_data  = pd.read_csv(physio_file, sep='\t')
        events_df  = pd.read_csv(event_file,  sep='\t')

        # skip run if no false alarms (mnt trials with a button press)
        n_fa = ((events_df['trial_type'] == 'mnt') & (events_df['response_code'] != 0)).sum()
        if n_fa == 0:
            print(f"Skipping {subj} run-{run_id:02d}: no false alarms.")
            continue

        # Neon recording for blink removal
        rec = None
        if neon_dirs_subj and run_id - 1 < len(neon_dirs_subj):
            try:
                rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[run_id - 1]))
            except Exception:
                pass

        # Step 1: preprocess — no polynomial detrend (detrend_order=None),
        # 6 Hz low-pass, phasic GLM regression
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t_pupil, pupil_tonic = preprocess_pupil(
                neon_data, rec=rec,
                f_lowpass=f_lowpass,
                f_downsample=f_downsample,
                detrend_order=detrend_order,
                is_rm_phasic=True,
                events_df=events_df,
            )
        if t_pupil is None:
            print(f"Skipping {subj} run-{run_id:02d}: too much missing data.")
            continue

        # Step 2: sliding window → per-window pupil mean + behavioral measures
        win_data = sliding_window_trials(events_df, pupil_tonic, t_pupil,
                                         win_trials=50, step_trials=15)

        # sanity check: all outputs must have the same length
        all_keys = ['pupil_mean'] + perf_keys
        lengths = {k: len(win_data[k]) for k in all_keys}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"{subj} run-{run_id:02d}: sliding_window_trials returned "
                f"inconsistent lengths: {lengths}"
            )

        # Step 3: z-score within this run
        pupil_z = zscore_safe(win_data['pupil_mean'])
        beh_z   = {k: zscore_safe(win_data[k]) for k in perf_keys}

        # single shared valid mask: window must be valid for pupil AND all measures
        valid = ~np.isnan(pupil_z)
        for k in perf_keys:
            valid = valid & ~np.isnan(beh_z[k])
        if valid.sum() < 2:
            print(f"Skipping {subj} run-{run_id:02d}: fewer than 2 valid windows.")
            continue

        for k in perf_keys:
            all_obs[k]['pupil'].extend(pupil_z[valid].tolist())
            all_obs[k]['behavior'].extend(beh_z[k][valid].tolist())

        included_subjects.add(subj)
        print(f"{subj} run-{run_id:02d}: {valid.sum()} windows")

n_subjects = len(included_subjects)
print(f"Included subjects: {n_subjects} — {sorted(included_subjects)}")

#%% Aggregate: sort by pupil value, divide into 30 equal bins,
#   compute mean behavioral value per bin
n_bins = 30

bin_results = {}
for k in perf_keys:
    pupil_arr = np.array(all_obs[k]['pupil'])
    beh_arr   = np.array(all_obs[k]['behavior'])
    if len(pupil_arr) == 0:
        continue
    sort_idx    = np.argsort(pupil_arr)
    pupil_sorted = pupil_arr[sort_idx]
    beh_sorted   = beh_arr[sort_idx]
    bin_edges = np.array_split(np.arange(len(pupil_sorted)), n_bins)
    bin_pupil_mean = np.array([pupil_sorted[b].mean() for b in bin_edges])
    bin_beh_mean   = np.array([beh_sorted[b].mean()   for b in bin_edges])
    bin_results[k] = {'pupil': bin_pupil_mean, 'behavior': bin_beh_mean}

#%% Plot: for each behavioral measure, scatter binned behavior vs binned pupil
#   with quadratic fit overlaid (replicating Fig. 3/4 of Brink 2016)
perf_labels = {
    'fa_rate':    'False Alarm Rate',
    'slow_q_rt':  'Slow Quintile RT (prop.)',
    'mean_rt':    'Mean RT (s)',
    'rtcv':       'RT Coefficient of Variation',
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, k in zip(axes.flat, perf_keys):
    if k not in bin_results:
        ax.set_title(perf_labels[k])
        continue
    px = bin_results[k]['pupil']
    py = bin_results[k]['behavior']
    ax.scatter(px, py, s=20, color='steelblue', zorder=3)
    # quadratic fit
    coef = np.polyfit(px, py, 2)
    x_fit = np.linspace(px.min(), px.max(), 200)
    ax.plot(x_fit, np.polyval(coef, x_fit), color='crimson', linewidth=1.5)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Tonic pupil size (z)')
    ax.set_ylabel(f'{perf_labels[k]} (z)')
    ax.set_title(perf_labels[k])
    ax.grid(True, alpha=0.4)
fig.suptitle(f'Tonic pupil vs. behavioral performance\n(binned, z-scored; replicating Brink 2016, N={n_subjects})')
plt.tight_layout()

# Combined plot: all 4 measures in one axes
perf_colors = {
    'fa_rate':   'steelblue',
    'slow_q_rt': 'darkorange',
    'mean_rt':   'forestgreen',
    'rtcv':      'crimson',
}
fig2, ax2 = plt.subplots(figsize=(8, 5))
for k in perf_keys:
    if k not in bin_results:
        continue
    px = bin_results[k]['pupil']
    py = bin_results[k]['behavior']
    c  = perf_colors[k]
    ax2.scatter(px, py, s=15, color=c, alpha=0.5, zorder=3)
    coef  = np.polyfit(px, py, fit_order)
    x_fit = np.linspace(px.min(), px.max(), 200)
    ax2.plot(x_fit, np.polyval(coef, x_fit), color=c, linewidth=2,
             label=perf_labels[k])
ax2.axhline(0, color='k', linewidth=0.6, linestyle='--')
ax2.axvline(0, color='k', linewidth=0.6, linestyle='--')
ax2.set_xlabel('Pupil diameter (z)')
ax2.set_ylabel('Behavioral measure (z)')
ax2.set_title(f'Pupil diameter vs. all behavioral measures\n(binned, z-scored; N={n_subjects})')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

#%% 