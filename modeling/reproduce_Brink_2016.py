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
def collect_all_obs(subject_list,
                    f_lowpass=6,
                    f_downsample=60,
                    detrend_order=1,
                    detrend_perf=False,
                    win_trials=50,
                    step_trials=15):
    """
    Collect z-scored sliding-window observations for each subject and run.

    Returns
    -------
    all_obs : dict
        all_obs[perf_key][subj][run_id] = {
            'pupil': [...], 'pupil_derivative': [...], 'behavior': [...], 'tot': [...]
        }
        where each list contains valid z-scored (or normalized) window values for
        that run. 'tot' is time-on-task normalized to [0, 1].
    included_subjects : set
        Subjects that contributed at least one valid run.
    """
    perf_keys = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']

    all_obs = {k: {} for k in perf_keys}
    included_subjects = set()

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
        for run_id in run_ids:
            physio_file = os.path.join(subj_nirs_dir,
                f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260423.tsv")
            if not os.path.isfile(physio_file):
                physio_file = os.path.join(subj_nirs_dir,
                    f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
            event_file = os.path.join(subj_nirs_dir,
                f"{subj}_task-gradCPT_run-{run_id:02d}_events.tsv")
            if not os.path.isfile(physio_file) or not os.path.isfile(event_file):
                continue

            neon_data  = pd.read_csv(physio_file, sep='\t')
            events_df  = pd.read_csv(event_file,  sep='\t')

            n_fa = ((events_df['trial_type'] == 'mnt') & (events_df['response_code'] != 0)).sum()
            if n_fa == 0:
                print(f"Skipping {subj} run-{run_id:02d}: no false alarms.")
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
                print(f"Skipping {subj} run-{run_id:02d}: too much missing data.")
                continue

            win_data = sliding_window_trials(events_df, pupil_tonic, t_pupil,
                                             win_trials=win_trials, step_trials=step_trials)

            all_keys = ['pupil_mean', 'pupil_derivative_mean'] + perf_keys
            lengths = {k: len(win_data[k]) for k in all_keys}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"{subj} run-{run_id:02d}: sliding_window_trials returned "
                    f"inconsistent lengths: {lengths}"
                )

            n_wins = len(win_data['win_center'])
            t_wins = np.arange(n_wins, dtype=float)

            pupil_z            = zscore_safe(win_data['pupil_mean'])
            pupil_derivative_z = zscore_safe(win_data['pupil_derivative_mean'])
            beh_z              = {k: zscore_safe(win_data[k]) for k in perf_keys}

            # detrend z-scored series (linear regression on window index)
            if detrend_order:
                for arr in [pupil_z, pupil_derivative_z]:
                    valid_mask = ~np.isnan(arr)
                    if valid_mask.sum() > detrend_order:
                        coef = np.polyfit(t_wins[valid_mask], arr[valid_mask], detrend_order)
                        arr[valid_mask] -= np.polyval(coef, t_wins[valid_mask])
            if detrend_perf:
                for k in perf_keys:
                    series = beh_z[k].copy()
                    valid_mask = ~np.isnan(series)
                    poly_order = detrend_perf if isinstance(detrend_perf, int) else 1
                    if valid_mask.sum() > poly_order:
                        coef = np.polyfit(t_wins[valid_mask], series[valid_mask], poly_order)
                        series[valid_mask] -= np.polyval(coef, t_wins[valid_mask])
                    beh_z[k] = series

            valid = ~np.isnan(pupil_z) & ~np.isnan(pupil_derivative_z)
            for k in perf_keys:
                valid = valid & ~np.isnan(beh_z[k])
            if valid.sum() < 2:
                print(f"Skipping {subj} run-{run_id:02d}: fewer than 2 valid windows.")
                continue

            tot = t_wins / (n_wins - 1) if n_wins > 1 else t_wins

            for k in perf_keys:
                all_obs[k].setdefault(subj, {})[run_id] = {
                    'pupil':            pupil_z[valid].tolist(),
                    'pupil_derivative': pupil_derivative_z[valid].tolist(),
                    'behavior':         beh_z[k][valid].tolist(),
                    'tot':              tot[valid].tolist(),
                }

            included_subjects.add(subj)
            print(f"{subj} run-{run_id:02d}: {valid.sum()} windows")

    n_subjects = len(included_subjects)
    print(f"Included subjects: {n_subjects} — {sorted(included_subjects)}")
    return all_obs, included_subjects


def plot_fig_5(all_obs, included_subjects, fit_order):
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

    n_subjects = len(included_subjects)

    # flatten all_obs[k][subj][run_id] -> pooled lists for binning
    pooled = {k: {'pupil': [], 'pupil_derivative': [], 'behavior': []} for k in perf_keys}
    for k in perf_keys:
        for subj_runs in all_obs[k].values():
            for run_data in subj_runs.values():
                pooled[k]['pupil'].extend(run_data['pupil'])
                pooled[k]['pupil_derivative'].extend(run_data['pupil_derivative'])
                pooled[k]['behavior'].extend(run_data['behavior'])

    # Aggregate: sort by pupil value, divide into 30 equal bins,
    #   compute mean behavioral value per bin
    n_bins = 30

    bin_results = {}
    for k in perf_keys:
        pupil_arr      = np.array(pooled[k]['pupil'])
        pupil_deriv_arr = np.array(pooled[k]['pupil_derivative'])
        beh_arr        = np.array(pooled[k]['behavior'])
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

    return bin_results

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

def compute_fig4_coefficients(all_obs, included_subjects, include_tot):
    """
    Returns two dicts (one per pupil measure): diameter and derivative.
    Each dict has keys 'linear' and 'quadratic', each a dict keyed by
    perf measure containing a list of per-subject mean coefficients.

    Parameters
    ----------
    include_tot : bool
        If True, include a linear time-on-task regressor in the model (nuisance).
        If False, fit pupil regressors only (no time-on-task control).
    all_obs : dict, optional
        Pre-built observations from collect_all_obs. If None, collect_all_obs
        is called internally with the other parameters.
    """
    perf_keys = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']

    # collect per-run coefficients keyed by subject
    subj_run_coefs = {}

    for subj in included_subjects:
        run_coefs = []
        # all_obs[k][subj][run_id] — iterate over runs present in the first perf_key
        first_key = perf_keys[0]
        for run_id in sorted(all_obs[first_key][subj].keys()):
            run_entry = {'diameter': {'linear': {}, 'quadratic': {}, 'tot': {}},
                         'derivative': {'linear': {}, 'tot': {}}}

            for k in perf_keys:
                if subj not in all_obs[k] or run_id not in all_obs[k][subj]:
                    continue
                rd = all_obs[k][subj][run_id]
                pz = np.array(rd['pupil'])
                dz = np.array(rd['pupil_derivative'])
                bz = np.array(rd['behavior'])
                tv = np.array(rd['tot'])

                n = len(bz)
                if n < 4:
                    continue

                # Regression for baseline diameter: [1, pupil, pupil^2, (time)]
                if include_tot:
                    X_d = np.column_stack([np.ones(n), pz, pz**2, tv])
                else:
                    X_d = np.column_stack([np.ones(n), pz, pz**2])
                coef_d, *_ = np.linalg.lstsq(X_d, bz, rcond=None)
                run_entry['diameter']['linear'][k]    = coef_d[1]
                run_entry['diameter']['quadratic'][k] = coef_d[2]
                if include_tot:
                    run_entry['diameter']['tot'][k] = coef_d[3]

                # Regression for derivative: [1, deriv, (time)]
                if include_tot:
                    X_r = np.column_stack([np.ones(n), dz, tv])
                else:
                    X_r = np.column_stack([np.ones(n), dz])
                coef_r, *_ = np.linalg.lstsq(X_r, bz, rcond=None)
                run_entry['derivative']['linear'][k] = coef_r[1]
                if include_tot:
                    run_entry['derivative']['tot'][k] = coef_r[2]

            run_coefs.append(run_entry)

        if run_coefs:
            subj_run_coefs[subj] = run_coefs

    # Average across runs per subject, then collect per-subject means
    results = {
        'diameter':   {'linear': {k: [] for k in perf_keys},
                       'quadratic': {k: [] for k in perf_keys},
                       'tot': {k: [] for k in perf_keys}},
        'derivative': {'linear': {k: [] for k in perf_keys},
                       'tot': {k: [] for k in perf_keys}},
    }
    coef_types = {
        'diameter':  ('linear', 'quadratic', 'tot'),
        'derivative': ('linear', 'tot'),
    }
    for subj, runs in subj_run_coefs.items():
        for pm, cts in coef_types.items():
            for ct in cts:
                for k in perf_keys:
                    vals = [r[pm][ct][k] for r in runs if k in r[pm].get(ct, {})]
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
        t, p_two = stats.ttest_1samp(vals, 0, alternative=alternative)
        # if alternative == 'greater':
        #     return p_two / 2 if t > 0 else 1 - p_two / 2
        return p_two

    has_tot = any(
        any(v for v in results[pm].get('tot', {}).values())
        for pm in ('diameter', 'derivative')
    )
    ncols = 3 if has_tot else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=False)

    for ax, pm in zip(axes[:2], ('diameter', 'derivative')):
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

    if has_tot:
        ax_tot = axes[2]
        x = np.arange(len(perf_keys))
        bar_w = 0.35
        tot_colors = {'diameter': 'steelblue', 'derivative': 'darkorange'}
        for offset, pm in zip([-bar_w/2, bar_w/2], ('diameter', 'derivative')):
            tot_means, tot_sems, tot_stars_list = [], [], []
            for k in perf_keys:
                tv = np.array(results[pm].get('tot', {}).get(k, []))
                tot_means.append(np.mean(tv) if len(tv) else 0)
                tot_sems.append(sp.stats.sem(tv) if len(tv) > 1 else 0)
                tot_stars_list.append(sig_stars(ttest_p(tv, 'two-sided')))
            bars = ax_tot.bar(x + offset, tot_means, bar_w, yerr=tot_sems,
                              color=tot_colors[pm], capsize=4,
                              label=pupil_labels[pm], zorder=3)
            y_offset = 0.005
            for i, b in enumerate(bars):
                if tot_stars_list[i]:
                    h = b.get_height()
                    err = tot_sems[i]
                    ypos = (h + err + y_offset) if h >= 0 else (h - err - y_offset * 4)
                    ax_tot.text(b.get_x() + b.get_width()/2, ypos,
                                tot_stars_list[i], ha='center', va='bottom', fontsize=10)
        ax_tot.axhline(0, color='k', linewidth=0.8)
        ax_tot.set_xticks(x)
        ax_tot.set_xticklabels(perf_labels, fontsize=9)
        ax_tot.set_ylabel('Regression coefficient (β)')
        ax_tot.set_title('Time-on-task')
        ax_tot.legend(fontsize=9)
        ax_tot.grid(axis='y', alpha=0.3)

    n_subj = max(len(v) for v in results['diameter']['linear'].values())
    fig.suptitle(f'Fig 4: Pupil–behavior regression coefficients (N={n_subj}){title_suffix}', fontsize=12)
    plt.tight_layout()
    plt.show()


#%% Figure 5
f_lowpass=6
f_downsample=60
fit_order=1
detrend_order=None
win_trials=50
step_trials=15
all_obs, included_subjects = collect_all_obs(subject_list,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    win_trials=win_trials,
                    step_trials=step_trials)
bin_results = plot_fig_5(all_obs, included_subjects, fit_order)

#%% Figure 4
fig4_results = compute_fig4_coefficients(
    all_obs,
    included_subjects,
    include_tot=False,
)
plot_fig4(fig4_results, title_suffix=' [without time-on-task regressor]')

# Fig 4 with time-on-task regressor (tot coefficient also plotted)
fig4_results_tot = compute_fig4_coefficients(
    all_obs,
    included_subjects,
    include_tot=True,
)
plot_fig4(fig4_results_tot, title_suffix=' [with time-on-task regressor]')

#%% Rerun fig 5 and fig 4 with linear detrend on both z-scored pupil diameter/derivative and performance metrics
f_lowpass=6
f_downsample=60
fit_order=2
detrend_order=1
detrend_perf=1
win_trials=20
step_trials=5
all_obs_detrend, included_subjects_detrend = collect_all_obs(subject_list,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    detrend_perf=detrend_perf,
                    win_trials=win_trials,
                    step_trials=step_trials)
bin_results_detrend = plot_fig_5(all_obs_detrend, included_subjects_detrend, fit_order)

fig4_results_detrend = compute_fig4_coefficients(
    all_obs_detrend,
    included_subjects_detrend,
    include_tot=False,
)
plot_fig4(fig4_results_detrend, title_suffix=' [linear detrend, without time-on-task regressor]')

fig4_results_detrend_tot = compute_fig4_coefficients(
    all_obs_detrend,
    included_subjects_detrend,
    include_tot=True,
)
plot_fig4(fig4_results_detrend_tot, title_suffix=' [linear detrend, with time-on-task regressor]')

# %% visualize behavior performance and pupil diameter/derivative
"""
For each subject, take average of their z-scored erformance metrics across time.
Plot the cross-subject results for each performance metrics with mean and SEM.
"""
nb_bins=100
f_lowpass=6
f_downsample=60
fit_order=1
detrend_order=2
detrend_perf=2
win_trials=20
step_trials=5
all_obs_detrend, included_subjects_detrend = collect_all_obs(subject_list,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    detrend_perf=detrend_perf,
                    win_trials=win_trials,
                    step_trials=step_trials)

perf_keys   = ['fa_rate', 'slow_q_rt', 'mean_rt', 'rtcv', 'smoothed_vtc']
perf_labels = ['False Alarm Rate', 'Slow Quintile RT', 'Mean RT', 'RTCV', 'Smoothed VTC']

# Common normalized time grid (0 = start, 1 = end of run)
t_grid = np.linspace(0, 1, nb_bins)

# Interpolate per-subject run-averaged traces onto t_grid
subj_traces    = {k: [] for k in perf_keys}
pupil_traces   = []
pupil_d_traces = []
for subj in sorted(included_subjects_detrend):
    pupil_runs, pupil_d_runs = [], []
    for k in perf_keys:
        if subj not in all_obs_detrend[k]:
            continue
        run_interps = []
        for run_data in all_obs_detrend[k][subj].values():
            tot = np.array(run_data['tot'])
            if len(tot) < 2:
                continue
            run_interps.append(np.interp(t_grid, tot, run_data['behavior']))
            if k == perf_keys[0]:
                pupil_runs.append(np.interp(t_grid, tot, run_data['pupil']))
                pupil_d_runs.append(np.interp(t_grid, tot, run_data['pupil_derivative']))
        if run_interps:
            subj_traces[k].append(np.mean(run_interps, axis=0))
    if pupil_runs:
        pupil_traces.append(np.mean(pupil_runs, axis=0))
        pupil_d_traces.append(np.mean(pupil_d_runs, axis=0))

def _plot_beh_with_pupil(subj_traces, pupil_src, perf_keys, perf_labels, t_grid,
                         pupil_label, pupil_color, suptitle, show_individual=True,
                         smooth_L=None):
    def _smooth(arr):
        if smooth_L:
            return smoothing_VTC_gaussian_array(arr, L=smooth_L)
        return arr

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    p_arr = np.array([_smooth(tr) for tr in pupil_src]) if len(pupil_src) else np.array(pupil_src)
    for ax, k, label in zip(axes.flat, perf_keys, perf_labels):
        traces = np.array([_smooth(tr) for tr in subj_traces[k]]) if subj_traces[k] else np.array([])
        if traces.ndim < 2 or len(traces) == 0:
            ax.set_title(label)
            continue
        ax2 = ax.twinx()
        if len(p_arr):
            p_mean = p_arr.mean(axis=0)
            p_sem  = sp.stats.sem(p_arr, axis=0)
            if show_individual:
                for tr in p_arr:
                    ax2.plot(t_grid, tr, color=pupil_color, linewidth=0.5, alpha=0.15)
            ax2.fill_between(t_grid, p_mean - p_sem, p_mean + p_sem,
                             alpha=0.2, color=pupil_color)
            ax2.plot(t_grid, p_mean, color=pupil_color, linewidth=1.5,
                     linestyle='--', label=pupil_label)
            ax2.set_ylabel(f'{pupil_label} (z)', color=pupil_color, fontsize=8)
            ax2.tick_params(axis='y', labelcolor=pupil_color)
        grand_mean = traces.mean(axis=0)
        sem        = sp.stats.sem(traces, axis=0)
        if show_individual:
            for trace in traces:
                ax.plot(t_grid, trace, color='steelblue', linewidth=0.6, alpha=0.3)
        ax.fill_between(t_grid, grand_mean - sem, grand_mean + sem,
                        alpha=0.3, color='steelblue')
        ax.plot(t_grid, grand_mean, color='steelblue', linewidth=2)
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_xlabel('Normalized time-on-task')
        ax.set_ylabel('Behavior (z)', color='steelblue', fontsize=8)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.set_title(f'{label} (N={len(traces)})')
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(perf_keys):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()

n_subj   = len(included_subjects_detrend)
smooth_L = int(20*nb_bins/450)
_plot_beh_with_pupil(subj_traces, pupil_traces, perf_keys, perf_labels, t_grid,
                     pupil_label='Pupil diameter', pupil_color='firebrick',
                     suptitle=f'Behavior (blue) + Pupil diameter (red dashed), N={n_subj}',
                     show_individual=False, smooth_L=smooth_L)

_plot_beh_with_pupil(subj_traces, pupil_d_traces, perf_keys, perf_labels, t_grid,
                     pupil_label='Pupil derivative', pupil_color='green',
                     suptitle=f'Behavior (blue) + Pupil derivative (green dashed), N={n_subj}',
                     show_individual=False, smooth_L=smooth_L)

# %% Single-subject example
def plot_single_subject(all_obs, subj, perf_keys, perf_labels, t_grid,
                        pupil_color='firebrick', pupil_d_color='green', smooth_L=None):
    """
    Two figures for one subject:
      Fig 1 — a single randomly chosen run (raw time series).
      Fig 2 — cross-run mean ± SEM interpolated onto t_grid.
    Each has one subplot per performance metric with behavior on the primary axis
    and pupil diameter (dashed) + pupil derivative (dotted) on the twin axis.
    """
    from matplotlib.lines import Line2D

    def _smooth(arr):
        if smooth_L:
            return smoothing_VTC_gaussian_array(arr, L=smooth_L)
        return arr

    # collect per-run interpolated traces
    run_ids_valid = [
        run_id for run_id in all_obs[perf_keys[0]].get(subj, {})
        if len(all_obs[perf_keys[0]][subj][run_id]['tot']) >= 2
    ]
    if not run_ids_valid:
        print(f"No valid runs for {subj}.")
        return

    rng        = np.random.default_rng()
    example_id = rng.choice(run_ids_valid)

    legend_elements = [
        Line2D([0], [0], color='steelblue',   linewidth=1.5,              label='Behavior'),
        Line2D([0], [0], color=pupil_color,   linewidth=1.2, linestyle='--', label='Pupil diameter'),
        Line2D([0], [0], color=pupil_d_color, linewidth=1.2, linestyle=':',  label='Pupil derivative'),
    ]

    def _fill_axes(axes, beh_dict, pup_dict, pupd_dict, title_suffix):
        for ax, k, label in zip(axes.flat, perf_keys, perf_labels):
            if k not in beh_dict:
                ax.set_title(label)
                continue
            ax2 = ax.twinx()
            beh_data  = beh_dict[k]
            pup_data  = pup_dict[k]
            pupd_data = pupd_dict[k]
            if isinstance(beh_data, dict):   # mean/sem dict
                ax.fill_between(t_grid,
                                beh_data['mean'] - beh_data['sem'],
                                beh_data['mean'] + beh_data['sem'],
                                alpha=0.25, color='steelblue')
                ax.plot(t_grid, beh_data['mean'], color='steelblue', linewidth=1.8)
                ax2.fill_between(t_grid,
                                 pup_data['mean'] - pup_data['sem'],
                                 pup_data['mean'] + pup_data['sem'],
                                 alpha=0.2, color=pupil_color)
                ax2.plot(t_grid, pup_data['mean'],  color=pupil_color,   linewidth=1.2, linestyle='--')
                ax2.fill_between(t_grid,
                                 pupd_data['mean'] - pupd_data['sem'],
                                 pupd_data['mean'] + pupd_data['sem'],
                                 alpha=0.2, color=pupil_d_color)
                ax2.plot(t_grid, pupd_data['mean'], color=pupil_d_color, linewidth=1.2, linestyle=':')
            else:                            # single array
                ax.plot(t_grid, beh_data,  color='steelblue',   linewidth=1.5)
                ax2.plot(t_grid, pup_data,  color=pupil_color,   linewidth=1.2, linestyle='--')
                ax2.plot(t_grid, pupd_data, color=pupil_d_color, linewidth=1.2, linestyle=':')
            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel('Normalized time-on-task')
            ax.set_ylabel('Behavior (z)', color='steelblue', fontsize=8)
            ax.tick_params(axis='y', labelcolor='steelblue')
            ax2.set_ylabel('Pupil (z)', fontsize=8)
            ax2.tick_params(axis='y')
            ax.set_title(f'{label}{title_suffix}')
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[len(perf_keys):]:
            ax.set_visible(False)

    # --- Figure 1: single random run ---
    beh_single  = {}
    pup_single  = {}
    pupd_single = {}
    for k in perf_keys:
        rd = all_obs[k].get(subj, {}).get(example_id)
        if rd is None or len(rd['tot']) < 2:
            continue
        tot = np.array(rd['tot'])
        beh_single[k]  = _smooth(np.interp(t_grid, tot, rd['behavior']))
        pup_single[k]  = _smooth(np.interp(t_grid, tot, rd['pupil']))
        pupd_single[k] = _smooth(np.interp(t_grid, tot, rd['pupil_derivative']))

    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    _fill_axes(axes1, beh_single, pup_single, pupd_single, '')
    fig1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    fig1.suptitle(f'Single subject example: {subj}  run-{example_id:02d}', fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- Figure 2: cross-run mean ± SEM ---
    beh_runs  = {k: [] for k in perf_keys}
    pup_runs  = {k: [] for k in perf_keys}
    pupd_runs = {k: [] for k in perf_keys}
    for k in perf_keys:
        for rd in all_obs[k].get(subj, {}).values():
            tot = np.array(rd['tot'])
            if len(tot) < 2:
                continue
            beh_runs[k].append(_smooth(np.interp(t_grid, tot, rd['behavior'])))
            pup_runs[k].append(_smooth(np.interp(t_grid, tot, rd['pupil'])))
            pupd_runs[k].append(_smooth(np.interp(t_grid, tot, rd['pupil_derivative'])))

    def _mean_sem(runs):
        arr = np.array(runs)
        if len(arr) == 0:
            return None
        return {'mean': arr.mean(axis=0),
                'sem':  sp.stats.sem(arr, axis=0) if len(arr) > 1 else np.zeros(len(t_grid))}

    beh_ms  = {k: _mean_sem(beh_runs[k])  for k in perf_keys if beh_runs[k]}
    pup_ms  = {k: _mean_sem(pup_runs[k])  for k in perf_keys if pup_runs[k]}
    pupd_ms = {k: _mean_sem(pupd_runs[k]) for k in perf_keys if pupd_runs[k]}

    n_runs = max(len(v) for v in beh_runs.values())
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    _fill_axes(axes2, beh_ms, pup_ms, pupd_ms, '')
    fig2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    fig2.suptitle(f'Single subject: {subj}  —  cross-run mean ± SEM  (N runs={n_runs})', fontsize=12)
    plt.tight_layout()
    plt.show()

# example_subj = np.random.choice(sorted(included_subjects_detrend))
example_subj = 'sub-721'
plot_single_subject(all_obs_detrend, example_subj, perf_keys, perf_labels, t_grid,
                    smooth_L=smooth_L)