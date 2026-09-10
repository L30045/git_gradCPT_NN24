#%% load library
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from params_setting import *
import xarray as xr
import cedalion.dot
head = cedalion.dot.get_standard_headmodel('icbm152')

#%% select subject / model
subject = 'sub-723'
eeg_reg_type = 'cont_EEG_cz_add_15s'
is_hp_fNIRS = False
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'
n_near_cz = 5
n_worst_trials = 10
is_plot = True  # If True, also show each figure in an interactive window before closing it
plot_dir = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/fit_vis'

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg', subject)
betas_path = os.path.join(eeg_der_dir, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')
Y_all_path = os.path.join(eeg_der_dir, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_Y_all.pkl.gz')
dm_all_path = os.path.join(eeg_der_dir, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_dm_all.pkl.gz')

with open(betas_path, 'rb') as f:
    betas_all = pickle.load(f)['betas']  # dims: parcel, chromo, regressor

with gzip.open(Y_all_path, 'rb') as f:
    Y_all = pickle.load(f)  # dims: chromo, parcel, time

with gzip.open(dm_all_path, 'rb') as f:
    dm_all = pickle.load(f)  # .common dims: time, chromo, regressor

Y_all = Y_all.pint.dequantify().sel(chromo=select_chromo).transpose('time', 'parcel')
dm_common = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
betas_sel = betas_all.sel(chromo=select_chromo)

#%% full-model fit: Y_hat = dm_all @ betas (all regressors)
Y_hat_full = xr.dot(dm_common, betas_sel, dims='regressor').transpose('time', 'parcel')

#%% non-EEG-only fit: Y_hat_nuis using only drift + GSR regressors (everything but 'bspline')
nuis_reg_names = [r for r in betas_sel.regressor.values if 'bspline' not in r]
Y_hat_nuis = xr.dot(dm_common.sel(regressor=nuis_reg_names), betas_sel.sel(regressor=nuis_reg_names),
                     dims='regressor').transpose('time', 'parcel')

#%% EEG-only fit: Y_eeg using only bspline (EEG delay) regressors
eeg_reg_names = [r for r in betas_sel.regressor.values if 'bspline' in r]
Y_eeg = xr.dot(dm_common.sel(regressor=eeg_reg_names), betas_sel.sel(regressor=eeg_reg_names),
               dims='regressor').transpose('time', 'parcel')

#%% ground truth with nuisance regressed out
Y_resid = Y_all - Y_hat_nuis

#%% find the 5 parcels nearest the Cz landmark
vertex_parcel = head.brain.vertices.parcel.values
cz_coord = head.landmarks.sel(label='Cz').pint.dequantify().values
brain_vert_coords = head.brain.vertices.pint.dequantify().values

unique_parcels_no_bg = np.array([p for p in np.unique(vertex_parcel) if not p.startswith('Background+FreeSurfer')])
parcel_centroids = {p: brain_vert_coords[vertex_parcel == p].mean(axis=0) for p in unique_parcels_no_bg}
parcel_dist_to_cz = {p: np.linalg.norm(c - cz_coord) for p, c in parcel_centroids.items()}
near_cz_parcels = sorted(parcel_dist_to_cz, key=parcel_dist_to_cz.get)[:n_near_cz]
print('Parcels nearest Cz:', near_cz_parcels)

#%% recover run boundaries (sample indices into the concatenated time axis) from the
# GS regressors, which are nonzero only within their own run's samples (see
# run_model_cont_EEG_fNIRS.py: model.get_global_mean_regressor + concatenate_runs_dms)
gs_reg_names = sorted([r for r in dm_common.regressor.values if r.startswith('GS run ')],
                       key=lambda r: int(r.split()[-1]))
run_bounds = []  # list of (start_idx, stop_idx_exclusive) into Y_all.time
for r in gs_reg_names:
    nz = np.nonzero(dm_common.sel(regressor=r).values)[0]
    run_bounds.append((nz.min(), nz.max() + 1))
n_runs = len(run_bounds)
print('Run boundaries (samples):', run_bounds)

fnirs_sfreq = 1 / np.diff(Y_all.time.values).mean()
run_time_offset = [Y_all.time.values[b[0]] for b in run_bounds]  # concatenated-time offset per run

#%% rebuild per-run trial onset tables (mnt_correct, city_correct) from the nirs-side
# events.tsv (same clock as fNIRS), mapped onto the concatenated Y_all.time axis.
# Run order gradcpt1/2/3 == nirs run-01/02/03 files, matching run_model_cont_EEG_fNIRS.py's
# concatenation order.
trial_condition = {
    'mnt_correct': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] == 0),
    'city_correct': lambda df: (df['trial_type'] == 'city') & (df['response_code'] > 0),
}

# HRF/trial window length used at fit time (matches run_model_cont_EEG_fNIRS.py's len_delay)
len_delay = 15
n_win = int(round(len_delay * fnirs_sfreq))

trial_onsets = {tt: [] for tt in trial_condition}  # tt -> list of concatenated-time onsets (sec)
for run_i in range(n_runs):
    run_num = f'{run_i + 1:02d}'
    ev_path = os.path.join(project_path, subject, 'nirs', f'{subject}_task-gradCPT_run-{run_num}_events.tsv')
    ev_df = pd.read_csv(ev_path, sep='\t')
    nirs_t_start = ev_df['onset'].values[0]
    offset = run_time_offset[run_i]
    run_stop_time = Y_all.time.values[run_bounds[run_i][1] - 1]
    for tt, cond in trial_condition.items():
        onsets_local = ev_df.loc[cond(ev_df), 'onset'].values - nirs_t_start + offset
        # keep only trials whose full [onset, onset+len_delay] window fits inside this run
        onsets_local = onsets_local[(onsets_local >= offset) & (onsets_local + len_delay <= run_stop_time)]
        trial_onsets[tt].append(onsets_local)

trial_onsets = {tt: np.concatenate(v) for tt, v in trial_onsets.items()}
for tt, onsets in trial_onsets.items():
    print(f'{tt}: {len(onsets)} trials available')

#%% helper: EV for one trial window, one parcel, given a y_true/y_hat pair (time x parcel)
def trial_window_idx(onset, n_win):
    i0 = np.searchsorted(Y_all.time.values, onset)
    return i0, i0 + n_win

def per_trial_ev(onsets, y_true_da, y_hat_da, parcel, n_win):
    y_true_p = y_true_da.sel(parcel=parcel).values
    y_hat_p = y_hat_da.sel(parcel=parcel).values
    ev_vals = []
    for onset in onsets:
        i0, i1 = trial_window_idx(onset, n_win)
        ev_vals.append(explained_variance_score(y_true_p[i0:i1], y_hat_p[i0:i1], force_finite=False))
    return np.array(ev_vals)

#%% plotting helper: 5 stacked subplots (one per parcel), either full timecourse or
# concatenated trial windows (with boundary markers), for one (y_true, y_hat) pair
def plot_fit_all_runs(y_true_da, y_hat_da, parcels, title, out_path,
                       true_label='Y (true)', pred_label='Y_hat'):
    fig, axs = plt.subplots(len(parcels), 1, figsize=(14, 2.5 * len(parcels)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, parcel in zip(axs, parcels):
        y_true = y_true_da.sel(parcel=parcel).values
        y_hat = y_hat_da.sel(parcel=parcel).values
        ev = explained_variance_score(y_true, y_hat, force_finite=False)
        ax.plot(y_true_da.time.values, y_true, color='k', linewidth=1, label=true_label)
        ax.plot(y_hat_da.time.values, y_hat, color='b', alpha=0.6, linewidth=1, label=pred_label)
        for b in run_bounds[1:]:
            ax.axvline(Y_all.time.values[b[0]], color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel(parcel, fontsize=8)
        ax.set_title(f'EV = {ev:.3f}', fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(loc='upper right')
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)


def plot_fit_trials(onsets_by_parcel, y_true_da, y_hat_da, parcels, title, out_path,
                     true_label='Y (true)', pred_label='Y_hat'):
    """onsets_by_parcel: dict parcel -> array of chosen trial onset times (concatenated time, sec)."""
    fig, axs = plt.subplots(len(parcels), 1, figsize=(14, 2.5 * len(parcels)), sharex=False)
    axs = np.atleast_1d(axs)
    for ax, parcel in zip(axs, parcels):
        onsets = onsets_by_parcel[parcel]
        y_true_p = y_true_da.sel(parcel=parcel).values
        y_hat_p = y_hat_da.sel(parcel=parcel).values
        cat_true, cat_hat, boundaries = [], [], [0]
        for onset in onsets:
            i0, i1 = trial_window_idx(onset, n_win)
            cat_true.append(y_true_p[i0:i1])
            cat_hat.append(y_hat_p[i0:i1])
            boundaries.append(boundaries[-1] + (i1 - i0))
        cat_true = np.concatenate(cat_true)
        cat_hat = np.concatenate(cat_hat)
        ev = explained_variance_score(cat_true, cat_hat, force_finite=False)
        x = np.arange(len(cat_true)) / fnirs_sfreq
        ax.plot(x, cat_true, color='k', linewidth=1, label=true_label)
        ax.plot(x, cat_hat, color='b', alpha=0.6, linewidth=1, label=pred_label)
        for b in boundaries[1:-1]:
            ax.axvline(b / fnirs_sfreq, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel(parcel, fontsize=8)
        ax.set_title(f'EV = {ev:.3f}', fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(loc='upper right')
    axs[-1].set_xlabel('Concatenated trial time (s)')
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)

#%% for each trial type, find the 10 smallest-EV trials per parcel (using the full-model fit)
worst_trials = {tt: dict() for tt in trial_condition}
for tt, onsets in trial_onsets.items():
    for parcel in near_cz_parcels:
        ev_vals = per_trial_ev(onsets, Y_all, Y_hat_full, parcel, n_win)
        worst_idx = np.argsort(ev_vals)[:n_worst_trials]
        worst_trials[tt][parcel] = onsets[worst_idx]

#%% 1. full model fit (dm_all @ betas vs Y_all): all_runs, worst mnt_correct, worst city_correct
subj_plot_dir = os.path.join(plot_dir, subject, eeg_reg_type)

plot_fit_all_runs(Y_all, Y_hat_full, near_cz_parcels,
                   f'{subject}: full-model fit, all runs',
                   os.path.join(subj_plot_dir, 'full_fit_all_runs.png'))

plot_fit_trials(worst_trials['mnt_correct'], Y_all, Y_hat_full, near_cz_parcels,
                 f'{subject}: full-model fit, {n_worst_trials} smallest-EV mnt_correct trials',
                 os.path.join(subj_plot_dir, 'full_fit_mnt_correct_worstEV.png'))

plot_fit_trials(worst_trials['city_correct'], Y_all, Y_hat_full, near_cz_parcels,
                 f'{subject}: full-model fit, {n_worst_trials} smallest-EV city_correct trials',
                 os.path.join(subj_plot_dir, 'full_fit_city_correct_worstEV.png'))

#%% 2. non-EEG (nuisance) fit: Y_hat_nuis vs Y_all, same trials as selected above
plot_fit_all_runs(Y_all, Y_hat_nuis, near_cz_parcels,
                   f'{subject}: non-EEG (nuisance) fit, all runs',
                   os.path.join(subj_plot_dir, 'nuis_fit_all_runs.png'),
                   pred_label='Y_hat_nuis')

plot_fit_trials(worst_trials['mnt_correct'], Y_all, Y_hat_nuis, near_cz_parcels,
                 f'{subject}: non-EEG (nuisance) fit, {n_worst_trials} smallest-EV mnt_correct trials',
                 os.path.join(subj_plot_dir, 'nuis_fit_mnt_correct_worstEV.png'),
                 pred_label='Y_hat_nuis')

plot_fit_trials(worst_trials['city_correct'], Y_all, Y_hat_nuis, near_cz_parcels,
                 f'{subject}: non-EEG (nuisance) fit, {n_worst_trials} smallest-EV city_correct trials',
                 os.path.join(subj_plot_dir, 'nuis_fit_city_correct_worstEV.png'),
                 pred_label='Y_hat_nuis')

#%% 3. nuisance-regressed-out ground truth vs EEG-only fit: (Y_all - Y_hat_nuis) vs Y_eeg
plot_fit_all_runs(Y_resid, Y_eeg, near_cz_parcels,
                   f'{subject}: EEG-only fit vs nuisance-regressed Y, all runs',
                   os.path.join(subj_plot_dir, 'eeg_fit_all_runs.png'),
                   true_label='Y_all - Y_hat_nuis', pred_label='Y_eeg')

plot_fit_trials(worst_trials['mnt_correct'], Y_resid, Y_eeg, near_cz_parcels,
                 f'{subject}: EEG-only fit vs nuisance-regressed Y, {n_worst_trials} smallest-EV mnt_correct trials',
                 os.path.join(subj_plot_dir, 'eeg_fit_mnt_correct_worstEV.png'),
                 true_label='Y_all - Y_hat_nuis', pred_label='Y_eeg')

plot_fit_trials(worst_trials['city_correct'], Y_resid, Y_eeg, near_cz_parcels,
                 f'{subject}: EEG-only fit vs nuisance-regressed Y, {n_worst_trials} smallest-EV city_correct trials',
                 os.path.join(subj_plot_dir, 'eeg_fit_city_correct_worstEV.png'),
                 true_label='Y_all - Y_hat_nuis', pred_label='Y_eeg')

print(f'Plots saved to {subj_plot_dir}')

#%% 4. residual diagnostics (Y_all - Y_hat_full): vs time, vs Y_all, vs Y_hat_full
Y_resid_full = Y_all - Y_hat_full

def add_fit_line(ax, x, y):
    """Overlay a least-squares fit line and annotate R-squared (correlation strength)."""
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, slope * x_line + intercept, color='r', linewidth=2,
            label=f'fit: R²={r2:.3f}')
    ax.legend(loc='upper right', fontsize=14)

def plot_residual_diagnostics(parcels, out_path):
    fig, axs = plt.subplots(len(parcels), 3, figsize=(16, 4 * len(parcels)))
    axs = np.atleast_2d(axs)
    for row, parcel in zip(axs, parcels):
        y_true = Y_all.sel(parcel=parcel).values
        y_hat = Y_hat_full.sel(parcel=parcel).values
        resid = Y_resid_full.sel(parcel=parcel).values

        row[0].plot(Y_all.time.values, resid, color='k', linewidth=0.8)
        row[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        for b in run_bounds[1:]:
            row[0].axvline(Y_all.time.values[b[0]], color='gray', linestyle='--', linewidth=0.8)
        row[0].set_xlabel('Time (s)')
        row[0].set_ylabel(f'{parcel}\nResidual (Y_all - Y_hat_full)')
        row[0].grid(alpha=0.3)

        row[1].scatter(y_true, resid, s=3, alpha=0.3, color='k')
        row[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        add_fit_line(row[1], y_true, resid)
        row[1].set_xlabel('Y_all')
        row[1].set_ylabel('Residual (Y_all - Y_hat_full)')
        row[1].grid(alpha=0.3)

        row[2].scatter(y_hat, resid, s=3, alpha=0.3, color='k')
        row[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        add_fit_line(row[2], y_hat, resid)
        row[2].set_xlabel('Y_hat_full')
        row[2].set_ylabel('Residual (Y_all - Y_hat_full)')
        row[2].grid(alpha=0.3)

    fig.suptitle(f'{subject}: residual diagnostics')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)

plot_residual_diagnostics(near_cz_parcels, os.path.join(subj_plot_dir, 'residual_diagnostics.png'))

