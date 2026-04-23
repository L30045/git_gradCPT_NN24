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
from utils import smoothing_VTC_gaussian_array
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from pupil_labs import neon_recording as nr
import re
from utils_eyetracking import preprocess_pupil, _build_gaussian_basis, _convolve_onsets
import warnings

dirs = os.listdir(project_path)
subject_list = sorted([d for d in dirs if 'sub' in d])

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

#%% functions
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
        'pupil_mean'            : mean tonic pupil per window (z-scored later)
        'pupil_derivative_mean' : mean of consecutive differences between
                                  trial-onset pupil samples within the window
        'fa_rate'               : false alarm rate (mnt trials pressed / total mnt trials)
        'slow_q_rt'   : slow quintile RT proportion (city trials)
        'mean_rt'     : mean RT of city correct trials
        'rtcv'         : RT coefficient of variation (city correct trials)
        'smoothed_vtc' : mean Gaussian-smoothed VTC (L=20) within each window
        'win_center'   : trial index of window centre (for reference)
    """
    trials = events_df.reset_index(drop=True)
    n_trials = len(trials)
    starts = np.arange(0, n_trials - win_trials + 1, step_trials)

    smoothed_vtc_all = smoothing_VTC_gaussian_array(events_df["VTC"], L=20)

    pupil_mean            = []
    pupil_derivative_mean = []
    fa_rate       = []
    slow_q_rt     = []
    mean_rt       = []
    rtcv          = []
    smoothed_vtc  = []
    win_center    = []

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
        diffs = np.diff(pupil_samples)
        pupil_derivative_mean.append(np.nanmean(diffs) if len(diffs) > 0 else np.nan)

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
        smoothed_vtc.append(np.nanmean(smoothed_vtc_all[start: start + win_trials]))

    return {
        'pupil_mean':            np.array(pupil_mean),
        'pupil_derivative_mean': np.array(pupil_derivative_mean),
        'fa_rate':               np.array(fa_rate),
        'slow_q_rt':             np.array(slow_q_rt),
        'mean_rt':               np.array(mean_rt),
        'rtcv':                  np.array(rtcv),
        'smoothed_vtc':          np.array(smoothed_vtc),
        'win_center':            np.array(win_center),
    }


def zscore_safe(x):
    """Z-score a 1D array, ignoring NaN. Returns NaN-filled array if std=0."""
    x = np.array(x, dtype=float)
    std = np.nanstd(x)
    if std == 0 or np.isnan(std):
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / std


# Fig 5
def plot_fig_5(subject_list,
               f_lowpass=6,
               f_downsample=60,
               fit_order=2,
               detrend_order=1,
               win_trials=50,
               step_trials=15):
    """
    Reproduce Brink 2016 Fig. 5: tonic pupil vs. behavioral performance measures.

    Parameters
    ----------
    subject_list : list of str
        List of subject directory names (e.g. ['sub-01', 'sub-02', ...]).
    f_lowpass : float
        Low-pass filter cutoff for pupil preprocessing (Hz). Default 6.
    f_downsample : float
        Downsample rate for pupil preprocessing (Hz). Default 60.
    fit_order : int
        Polynomial order for regression between pupil and behavioral measure. Default 2.
    detrend_order : int
        Polynomial order for removing time-on-task trend. Default 1.
    win_trials : int
        Sliding window width in trials. Default 35.
    step_trials : int
        Sliding window step size in trials. Default 11.
    """
    perf_keys = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']

    # all_obs[perf_key] accumulates (pupil_z, pupil_derivative_z, behavior_z) pairs across all subj/runs
    all_obs = {k: {'pupil': [], 'pupil_derivative': [], 'behavior': []} for k in perf_keys}
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

            # Step 1: preprocess — polynomial detrend (detrend_order), 6 Hz low-pass, phasic GLM regression
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t_pupil, pupil_tonic = preprocess_pupil(
                    neon_data, rec=rec,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    is_rm_phasic=False,
                    events_df=events_df,
                )
            if t_pupil is None:
                print(f"Skipping {subj} run-{run_id:02d}: too much missing data.")
                continue

            # Step 2: sliding window → per-window pupil mean + behavioral measures
            win_data = sliding_window_trials(events_df, pupil_tonic, t_pupil,
                                             win_trials=win_trials, step_trials=step_trials)

            # sanity check: all outputs must have the same length
            all_keys = ['pupil_mean', 'pupil_derivative_mean'] + perf_keys
            lengths = {k: len(win_data[k]) for k in all_keys}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"{subj} run-{run_id:02d}: sliding_window_trials returned "
                    f"inconsistent lengths: {lengths}"
                )

            # Step 2b: detrend performance measures using window index as time axis
            n_wins = len(win_data['win_center'])
            t_wins = np.arange(n_wins, dtype=float)
            detrended_perf = {}
            for k in perf_keys:
                series = win_data[k].copy().astype(float)
                valid_mask = ~np.isnan(series)
                if detrend_order:
                    if valid_mask.sum() > detrend_order:
                        coef = np.polyfit(t_wins[valid_mask], series[valid_mask], detrend_order)
                        series[valid_mask] -= np.polyval(coef, t_wins[valid_mask])
                detrended_perf[k] = series

            # Step 3: z-score within this run
            pupil_z            = zscore_safe(win_data['pupil_mean'])
            pupil_derivative_z = zscore_safe(win_data['pupil_derivative_mean'])
            beh_z              = {k: zscore_safe(detrended_perf[k]) for k in perf_keys}

            # single shared valid mask: window must be valid for pupil, derivative, AND all measures
            valid = ~np.isnan(pupil_z) & ~np.isnan(pupil_derivative_z)
            for k in perf_keys:
                valid = valid & ~np.isnan(beh_z[k])
            if valid.sum() < 2:
                print(f"Skipping {subj} run-{run_id:02d}: fewer than 2 valid windows.")
                continue

            for k in perf_keys:
                all_obs[k]['pupil'].extend(pupil_z[valid].tolist())
                all_obs[k]['pupil_derivative'].extend(pupil_derivative_z[valid].tolist())
                all_obs[k]['behavior'].extend(beh_z[k][valid].tolist())

            included_subjects.add(subj)
            print(f"{subj} run-{run_id:02d}: {valid.sum()} windows")

    n_subjects = len(included_subjects)
    print(f"Included subjects: {n_subjects} — {sorted(included_subjects)}")

    # Aggregate: sort by pupil value, divide into 30 equal bins,
    #   compute mean behavioral value per bin
    n_bins = 30

    bin_results = {}
    for k in perf_keys:
        pupil_arr      = np.array(all_obs[k]['pupil'])
        pupil_deriv_arr = np.array(all_obs[k]['pupil_derivative'])
        beh_arr        = np.array(all_obs[k]['behavior'])
        if len(pupil_arr) == 0:
            continue

        # bin by pupil mean
        sort_idx      = np.argsort(pupil_arr)
        pupil_sorted  = pupil_arr[sort_idx]
        beh_sorted    = beh_arr[sort_idx]
        bin_edges     = np.array_split(np.arange(len(pupil_sorted)), n_bins)
        bin_pupil_mean = np.array([pupil_sorted[b].mean() for b in bin_edges])
        bin_beh_mean   = np.array([beh_sorted[b].mean()   for b in bin_edges])

        # bin by pupil derivative
        sort_idx_d       = np.argsort(pupil_deriv_arr)
        deriv_sorted     = pupil_deriv_arr[sort_idx_d]
        beh_sorted_d     = beh_arr[sort_idx_d]
        bin_edges_d      = np.array_split(np.arange(len(deriv_sorted)), n_bins)
        bin_deriv_mean   = np.array([deriv_sorted[b].mean()   for b in bin_edges_d])
        bin_beh_mean_d   = np.array([beh_sorted_d[b].mean()   for b in bin_edges_d])

        bin_results[k] = {
            'pupil':            bin_pupil_mean,
            'behavior':         bin_beh_mean,
            'pupil_derivative': bin_deriv_mean,
            'behavior_d':       bin_beh_mean_d,
        }

    # Plot: for each behavioral measure, scatter binned behavior vs binned pupil
    #   with quadratic fit overlaid (replicating Fig. 3/4 of Brink 2016)
    perf_labels = {
        'fa_rate':      'False Alarm Rate',
        'slow_q_rt':    'Slow Quintile RT (prop.)',
        'mean_rt':      'Mean RT (s)',
        'rtcv':         'RT Coefficient of Variation',
        'smoothed_vtc': 'Smoothed VTC',
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes.flat, perf_keys):
        if k not in bin_results:
            ax.set_title(perf_labels[k])
            continue
        px = bin_results[k]['pupil']
        py = bin_results[k]['behavior']
        ax.scatter(px, py, s=20, color='steelblue', zorder=3)
        # quadratic fit
        coef = np.polyfit(px, py, fit_order)
        x_fit = np.linspace(px.min(), px.max(), 200)
        ax.plot(x_fit, np.polyval(coef, x_fit), color='crimson', linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Tonic pupil size (z)')
        ax.set_ylabel(f'{perf_labels[k]} (z)')
        ax.set_title(perf_labels[k])
        ax.grid(True, alpha=0.4)
    for ax in axes.flat[len(perf_keys):]:
        ax.set_visible(False)
    fig.suptitle(f'Tonic pupil vs. behavioral performance\n(binned, z-scored; replicating Brink 2016, N={n_subjects})')
    plt.tight_layout()

    # Combined plot: all measures in one axes
    perf_colors = {
        'fa_rate':      'steelblue',
        'slow_q_rt':    'darkorange',
        'mean_rt':      'forestgreen',
        'rtcv':         'crimson',
        'smoothed_vtc': 'mediumorchid',
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

    # --- Pupil derivative plots ---
    # 2x3: behavior vs pupil derivative, one panel per measure
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes3.flat, perf_keys):
        if k not in bin_results:
            ax.set_title(perf_labels[k])
            continue
        px = bin_results[k]['pupil_derivative']
        py = bin_results[k]['behavior_d']
        ax.scatter(px, py, s=20, color='steelblue', zorder=3)
        coef  = np.polyfit(px, py, fit_order)
        x_fit = np.linspace(px.min(), px.max(), 200)
        ax.plot(x_fit, np.polyval(coef, x_fit), color='crimson', linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Pupil derivative (z)')
        ax.set_ylabel(f'{perf_labels[k]} (z)')
        ax.set_title(perf_labels[k])
        ax.grid(True, alpha=0.4)
    for ax in axes3.flat[len(perf_keys):]:
        ax.set_visible(False)
    fig3.suptitle(f'Pupil derivative vs. behavioral performance\n(binned, z-scored; N={n_subjects})')
    plt.tight_layout()

    # Combined: all 4 measures vs pupil derivative
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for k in perf_keys:
        if k not in bin_results:
            continue
        px = bin_results[k]['pupil_derivative']
        py = bin_results[k]['behavior_d']
        c  = perf_colors[k]
        ax4.scatter(px, py, s=15, color=c, alpha=0.5, zorder=3)
        coef  = np.polyfit(px, py, fit_order)
        x_fit = np.linspace(px.min(), px.max(), 200)
        ax4.plot(x_fit, np.polyval(coef, x_fit), color=c, linewidth=2,
                 label=perf_labels[k])
    ax4.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax4.axvline(0, color='k', linewidth=0.6, linestyle='--')
    ax4.set_xlabel('Pupil derivative (z)')
    ax4.set_ylabel('Behavioral measure (z)')
    ax4.set_title(f'Pupil derivative vs. all behavioral measures\n(binned, z-scored; N={n_subjects})')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

#%% Figure 5
f_lowpass=6
f_downsample=60
fit_order=1
detrend_order=1
win_trials=20
step_trials=5
plot_fig_5(subject_list,
           f_lowpass=f_lowpass,
           f_downsample=f_downsample,
           fit_order=fit_order,
           detrend_order=detrend_order,
           win_trials=win_trials,
           step_trials=step_trials)

#%% Figure 4
"""
Fig 4: Regression coefficients (linear + quadratic) per pupil measure × behavioral measure.

For each subject × run × behavioral measure:
  - Build design matrix: [pupil_z, pupil_z^2, time_on_task] (all z-scored / normalized)
  - Regress z-scored behavior onto it
  - Save linear (β1) and quadratic (β2) coefficients

Then average across runs per subject, then across subjects.
Plot: bar chart with SEM error bars, one group of bars per behavioral measure,
linear and quadratic shown separately, for baseline diameter (left panel)
and diameter derivative (right panel).  Stars for significance via one-sample t-test vs 0.
"""

def compute_fig4_coefficients(subject_list,
                               f_lowpass=6,
                               f_downsample=60,
                               detrend_order=1,
                               win_trials=50,
                               step_trials=15,
                               include_tot=True):
    """
    Returns two dicts (one per pupil measure): diameter and derivative.
    Each dict has keys 'linear' and 'quadratic', each a dict keyed by
    perf measure containing a list of per-subject mean coefficients.

    Parameters
    ----------
    include_tot : bool
        If True, include a linear time-on-task regressor in the model (nuisance).
        If False, fit pupil regressors only (no time-on-task control).
    """
    perf_keys = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']

    # collect per-run coefficients: subj -> run -> measure -> {lin, quad}
    subj_run_coefs = {}

    for subj in subject_list:
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

        run_coefs = []
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

            neon_data = pd.read_csv(physio_file, sep='\t')
            events_df = pd.read_csv(event_file,  sep='\t')

            n_fa = ((events_df['trial_type'] == 'mnt') & (events_df['response_code'] != 0)).sum()
            if n_fa == 0:
                continue

            rec = None
            if neon_dirs_subj and run_id - 1 < len(neon_dirs_subj):
                try:
                    rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[run_id - 1]))
                except Exception:
                    pass

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t_pupil, pupil_tonic = preprocess_pupil(
                    neon_data, rec=rec,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    is_rm_phasic=False,
                    events_df=events_df,
                )
            if t_pupil is None:
                continue

            win_data = sliding_window_trials(events_df, pupil_tonic, t_pupil,
                                             win_trials=win_trials, step_trials=step_trials)

            n_wins = len(win_data['win_center'])
            t_wins = np.arange(n_wins, dtype=float)
            # time-on-task predictor (normalized to [0,1])
            tot = t_wins / (n_wins - 1) if n_wins > 1 else t_wins

            pupil_z  = zscore_safe(win_data['pupil_mean'])
            deriv_z  = zscore_safe(win_data['pupil_derivative_mean'])

            run_entry = {'diameter': {'linear': {}, 'quadratic': {}},
                         'derivative': {'linear': {}, 'quadratic': {}}}

            for k in perf_keys:
                beh = win_data[k].astype(float)
                beh_z = zscore_safe(beh)

                valid = (~np.isnan(pupil_z) & ~np.isnan(deriv_z) & ~np.isnan(beh_z))
                if valid.sum() < 4:
                    continue

                pz  = pupil_z[valid]
                dz  = deriv_z[valid]
                bz  = beh_z[valid]
                tv  = tot[valid]

                # Regression for baseline diameter: [1, pupil, pupil^2, (time)]
                if include_tot:
                    X_d = np.column_stack([np.ones(valid.sum()), pz, pz**2, tv])
                else:
                    X_d = np.column_stack([np.ones(valid.sum()), pz, pz**2])
                coef_d, *_ = np.linalg.lstsq(X_d, bz, rcond=None)
                run_entry['diameter']['linear'][k]    = coef_d[1]
                run_entry['diameter']['quadratic'][k] = coef_d[2]

                # Regression for derivative: [1, deriv, (time)]
                if include_tot:
                    X_r = np.column_stack([np.ones(valid.sum()), dz, tv])
                else:
                    X_r = np.column_stack([np.ones(valid.sum()), dz])
                coef_r, *_ = np.linalg.lstsq(X_r, bz, rcond=None)
                run_entry['derivative']['linear'][k]    = coef_r[1]

            run_coefs.append(run_entry)

        if run_coefs:
            subj_run_coefs[subj] = run_coefs

    # Average across runs per subject, then collect per-subject means
    results = {
        'diameter':   {'linear': {k: [] for k in perf_keys},
                       'quadratic': {k: [] for k in perf_keys}},
        'derivative': {'linear': {k: [] for k in perf_keys}},
    }
    for subj, runs in subj_run_coefs.items():
        for pm in ('diameter', 'derivative'):
            for ct in ('linear', 'quadratic'):
                if ct == 'quadratic' and pm == 'derivative':
                    continue
                for k in perf_keys:
                    vals = [r[pm][ct][k] for r in runs if k in r[pm][ct]]
                    if vals:
                        results[pm][ct][k].append(np.mean(vals))

    return results


def plot_fig4(results, title_suffix=''):
    """
    Plot Fig 4 style: two panels (baseline diameter, diameter derivative).

    Significance testing:
      - Diameter quadratic: one-tailed t-test (expected positive per Yerkes-Dodson)
      - All other terms (diameter linear, derivative linear): two-tailed t-test
    """
    from scipy import stats

    perf_keys    = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']
    perf_labels  = ['False alarms', 'Slow quintile', 'Response time', 'RTCV', 'Smoothed VTC']
    pupil_labels = {'diameter': 'Baseline diameter', 'derivative': 'Diameter derivative'}

    def sig_stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''

    def ttest_p(vals, alternative='two-sided'):
        """Return p-value from one-sample t-test vs 0."""
        if len(vals) < 2:
            return 1.0
        t, p_two = stats.ttest_1samp(vals, 0)
        if alternative == 'greater':
            return p_two / 2 if t > 0 else 1 - p_two / 2
        return p_two

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    for ax, pm in zip(axes, ('diameter', 'derivative')):
        x = np.arange(len(perf_keys))
        has_quad = pm == 'diameter'
        bar_w = 0.35 if has_quad else 0.5

        lin_means, lin_sems, lin_stars_list   = [], [], []
        quad_means, quad_sems, quad_stars_list = [], [], []

        for k in perf_keys:
            lv = np.array(results[pm]['linear'][k])
            lin_means.append(np.mean(lv) if len(lv) else 0)
            lin_sems.append(sp.stats.sem(lv) if len(lv) > 1 else 0)
            # diameter linear and derivative linear: two-tailed
            lin_stars_list.append(sig_stars(ttest_p(lv, 'two-sided')))

            if has_quad:
                qv = np.array(results[pm]['quadratic'][k])
                quad_means.append(np.mean(qv) if len(qv) else 0)
                quad_sems.append(sp.stats.sem(qv) if len(qv) > 1 else 0)
                # diameter quadratic: one-tailed (expect positive)
                quad_stars_list.append(sig_stars(ttest_p(qv, 'greater')))

        if has_quad:
            bars_lin  = ax.bar(x - bar_w/2, lin_means, bar_w, yerr=lin_sems,
                               color='gray', capsize=4, label='Linear', zorder=3)
            bars_quad = ax.bar(x + bar_w/2, quad_means, bar_w, yerr=quad_sems,
                               color='none', edgecolor='gray', hatch='///', capsize=4,
                               label='Quadratic', zorder=3)
        else:
            bars_lin  = ax.bar(x, lin_means, bar_w, yerr=lin_sems,
                               color='gray', capsize=4, label='Linear', zorder=3)
            bars_quad = []

        # significance stars
        y_offset = 0.005
        for i, blin in enumerate(bars_lin):
            if lin_stars_list[i]:
                h = blin.get_height()
                err = lin_sems[i]
                ypos = (h + err + y_offset) if h >= 0 else (h - err - y_offset * 4)
                ax.text(blin.get_x() + blin.get_width()/2, ypos,
                        lin_stars_list[i], ha='center', va='bottom', fontsize=10)

        if has_quad:
            for i, bquad in enumerate(bars_quad):
                if quad_stars_list[i]:
                    h = bquad.get_height()
                    err = quad_sems[i]
                    ypos = (h + err + y_offset) if h >= 0 else (h - err - y_offset * 4)
                    ax.text(bquad.get_x() + bquad.get_width()/2, ypos,
                            quad_stars_list[i], ha='center', va='bottom', fontsize=10)

        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(perf_labels, fontsize=9)
        ax.set_ylabel('Regression coefficient (β)')
        ax.set_title(pupil_labels[pm])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    n_subj = max(len(v) for v in results['diameter']['linear'].values())
    fig.suptitle(f'Fig 4: Pupil–behavior regression coefficients (N={n_subj}){title_suffix}', fontsize=12)
    plt.tight_layout()
    plt.show()


fig4_results = compute_fig4_coefficients(
    subject_list,
    f_lowpass=6,
    f_downsample=60,
    detrend_order=None,
    win_trials=20,
    step_trials=5,
    include_tot=False,
)
plot_fig4(fig4_results, title_suffix=' [with time-on-task effect]')

# Fig 4 without time-on-task regressor
fig4_results_notot = compute_fig4_coefficients(
    subject_list,
    f_lowpass=6,
    f_downsample=60,
    detrend_order=None,
    win_trials=20,
    step_trials=5,
    include_tot=True,
)
plot_fig4(fig4_results_notot, title_suffix=' [without time-on-task effect]')
