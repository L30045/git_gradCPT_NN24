#%% Compare y_hat from CORRECT_CODE_runGLM_on_image.py (event-based HRF GLM)
# against y_hat from run_model_cont_EEG_fNIRS.py (continuous-EEG GLM), for the same
# subject and parcel, using a 3-step sequential regression for both fits:
#   1) OLS-regress out per-run Legendre drift from Y_all.
#   2) Compute GSR from the drift-residualized all-parcel signal, then OLS-regress
#      it out of the drift-residualized single-parcel signal.
#   3) Fit AR-IRLS using only the event-based HRF (Fit 1) or EEG delay (Fit 2)
#      regressors on the twice-residualized signal, and plot the results.
import os
import gzip
import pickle
import copy
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

#%% ---- Fit 1 data prep: CORRECT_CODE_runGLM_on_image.py (event-based HRF GLM on MNT trials) ----
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

stims_pruned_list = [stims_pruned_list[i] for i in reorder_idx]

# dequantify once (drop pint units) so every OLS/AR-IRLS step below works in plain
# numeric space without repeated pint round-tripping; glm.fit dequantifies internally
# anyway, and pf.concatenate_runs handles units=None gracefully via pint.quantify(None)
all_runs = [r.pint.dequantify() for r in cropped_runs]
all_runs_allparcel = [r.pint.dequantify() for r in cropped_runs_allparcel]


def ols_regress_out_per_run(target_runs, regressor_dms):
    """OLS-fit each DesignMatrix in regressor_dms against the corresponding run in
    target_runs (matched by index, one regression per run), and return the list of
    per-run residuals (target minus the fitted regressor contribution)."""
    resid_runs = []
    for run, dm in zip(target_runs, regressor_dms):
        results = glm.fit(run, dm, noise_model='ols')
        betas = results.sm.params  # dims: parcel, chromo, regressor
        fit_vals = xr.dot(dm.common, betas, dims='regressor').transpose(*run.dims)
        resid_runs.append(run - fit_vals)
    return resid_runs


def compute_fit_metrics(y_true, dm_common, betas, nuisance_prefixes=('GS run', 'Drift')):
    """Compute R^2, partial R^2, correlation, and log RMSE for one fit.

    Partial R^2 removes any nuisance (GSR + drift) regressors' fitted contribution
    from y_true, then computes R^2 of that denuisanced signal against the fit from
    the remaining (non-nuisance, i.e. task/HRF or EEG) regressors only. Under the
    3-step pipeline the final AR-IRLS design matrix has no nuisance regressors left
    (they were already OLS-removed in steps 1-2), so this naturally reduces to the
    plain R^2 on y_true / y_hat.
    """
    y_true = np.asarray(y_true)

    reg_names = [str(r) for r in dm_common.regressor.values]
    is_nuisance = np.array([any(name.startswith(p) for p in nuisance_prefixes) for name in reg_names])

    dm_vals = dm_common.sel(chromo=select_chromo).transpose('time', 'regressor').values  # (time, n_reg)
    betas_vals = betas.sel(parcel=select_parcel, chromo=select_chromo).transpose('regressor').values  # (n_reg,)

    y_hat_full = dm_vals @ betas_vals
    n_time = dm_vals.shape[0]
    nuisance_fit = dm_vals[:, is_nuisance] @ betas_vals[is_nuisance] if is_nuisance.any() else np.zeros(n_time)
    nonnuisance_fit = dm_vals[:, ~is_nuisance] @ betas_vals[~is_nuisance] if (~is_nuisance).any() else np.zeros(n_time)

    resid = y_true - y_hat_full
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    y_denuisance = y_true - nuisance_fit
    resid_partial = y_denuisance - nonnuisance_fit
    ss_res_partial = np.sum(resid_partial ** 2)
    ss_tot_partial = np.sum((y_denuisance - y_denuisance.mean()) ** 2)
    r2_partial = 1 - ss_res_partial / ss_tot_partial if ss_tot_partial > 0 else np.nan

    corr = np.corrcoef(y_true, y_hat_full)[0, 1]
    log_rmse = np.log(np.sqrt(np.mean(resid ** 2)))

    return {'r2': r2, 'r2_partial': r2_partial, 'corr': corr, 'log_rmse': log_rmse}


def three_step_fit_event_based():
    """3-step sequential regression for the event-based fit (Fit 1)."""
    # Step 1: OLS-regress out per-run Legendre drift, from both the single-parcel
    # target and the all-parcel copy (the latter is needed for step 2's GSR)
    drift_dms = [glm.design_matrix.drift_legendre_regressors(r, cfg_GLM['drift_order']) for r in all_runs]
    runs_resid1 = ols_regress_out_per_run(all_runs, drift_dms)
    runs_ap_resid1 = ols_regress_out_per_run(all_runs_allparcel, drift_dms)

    # Step 2: GSR computed from the drift-residualized all-parcel signal, then
    # OLS-regressed out of the drift-residualized single-parcel signal
    gsr_dms = pf.get_global_mean_regressor(runs_ap_resid1)
    runs_resid2 = ols_regress_out_per_run(runs_resid1, gsr_dms)

    # Step 3: AR-IRLS using only the event-based HRF regressors on the
    # twice-residualized signal
    Y_resid2, stim_df, runs_resid2_updated = pf.concatenate_runs(runs_resid2, stims_pruned_list)
    hrf_kernel = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    hrf_dm = glm.design_matrix.hrf_regressors(Y_resid2, stim_df, hrf_kernel)
    hrf_dm.common = hrf_dm.common.fillna(0)

    print(f"Start event-based AR-IRLS fitting on drift+GSR-residualized signal ({subject})")
    ar_results = glm.fit(Y_resid2, hrf_dm, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params
    y_hat = xr.dot(hrf_dm.common, betas, dims='regressor').transpose(*Y_resid2.dims)

    y_true_vals = Y_resid2.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_resid2.time.values
    metrics = compute_fit_metrics(y_true_vals, hrf_dm.common, betas)

    # event-triggered HRF shape, per trial type (mirrors pf.GLM's own HRF-estimate logic)
    basis_hrf = hrf_kernel(Y_resid2)
    fs = frequency.sampling_rate(Y_resid2).to('Hz')
    dT = np.round(1 / fs, 3)
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    hrf_estimate_list = []
    for trial_type in stim_df['trial_type'].unique():
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f'HRF {trial_type}'))
        hrf_est = pf.estimate_HRF_from_beta(betas_hrf, basis_hrf)
        hrf_estimate_list.append(hrf_est.expand_dims({'trial_type': [trial_type]}))
    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    reltime = np.linspace(-before_samples * dT, after_samples * dT, len(hrf_estimate.time))
    hrf_estimate = hrf_estimate.assign_coords(time=reltime)
    hrf = hrf_estimate.sel(parcel=select_parcel, chromo=select_chromo)  # dims: time, trial_type

    return t, y_true_vals, y_hat_vals, hrf_dm, metrics, hrf


#%% ---- Fit 2 data prep: run_model_cont_EEG_fNIRS.py (continuous-EEG GLM) ----
# load Y_all + dm_all saved by run_model_cont_EEG_fNIRS.py for this subject; dm_all's
# drift/GSR columns are reused (extracted by name) as the step-1/step-2 design
# matrices, and its bspline columns as the step-3 EEG delay regressors
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
base = os.path.join(eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')

Y_all_path = base + '_Y_all.pkl.gz'
dm_all_path = base + '_dm_all.pkl.gz'
betas_path = base + '_betas.pkl'

for p in (Y_all_path, dm_all_path, betas_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Rerun run_model_cont_EEG_fNIRS.py with is_overwrite=True for {subject}.")

with gzip.open(Y_all_path, 'rb') as f:
    Y_all_cont_raw = pickle.load(f)  # dims: chromo, parcel, time

with gzip.open(dm_all_path, 'rb') as f:
    dm_all_cont_raw = pickle.load(f)  # .common dims: time, chromo, regressor

# basis_da (bspline basis, dims: regressor [full-resolution delay taps], component
# [bspline0..N]) is only used to expand our refit bspline betas back to a per-delay
# response curve; the betas/betas_eeg saved alongside it are NOT used since we refit.
with open(betas_path, 'rb') as f:
    basis_da = pickle.load(f)['basis_da']

fnirs_sfreq_cont = 1 / np.diff(Y_all_cont_raw.time.values).mean()
n_delay_taps = len(basis_da.regressor)
t_delay = np.arange(n_delay_taps) / fnirs_sfreq_cont

Y_all_cont_raw = Y_all_cont_raw.pint.dequantify()


def three_step_fit_continuous_eeg():
    """3-step sequential regression for the continuous-EEG fit (Fit 2), reusing the
    drift/GSR/bspline regressors already present in dm_all_cont_raw by name."""
    reg_names_all = [str(r) for r in dm_all_cont_raw.common.regressor.values]
    drift_regs = [r for r in reg_names_all if r.startswith('Drift')]
    gs_regs = [r for r in reg_names_all if r.startswith('GS run')]
    bspline_regs = [r for r in reg_names_all if r.startswith('bspline')]

    drift_dm = copy.deepcopy(dm_all_cont_raw)
    drift_dm.common = drift_dm.common.sel(regressor=drift_regs)

    Y_all_cont_parcel = Y_all_cont_raw.sel(parcel=[select_parcel])

    # Step 1: OLS-regress out drift, from select_parcel and from all parcels
    # (the latter needed for step 2's GSR)
    drift_results_1p = glm.fit(Y_all_cont_parcel, drift_dm, noise_model='ols')
    drift_fit_1p = xr.dot(drift_dm.common, drift_results_1p.sm.params, dims='regressor').transpose(*Y_all_cont_parcel.dims)
    Y_resid1_1p = Y_all_cont_parcel - drift_fit_1p

    drift_results_ap = glm.fit(Y_all_cont_raw, drift_dm, noise_model='ols')
    drift_fit_ap = xr.dot(drift_dm.common, drift_results_ap.sm.params, dims='regressor').transpose(*Y_all_cont_raw.dims)
    Y_resid1_ap = Y_all_cont_raw - drift_fit_ap

    # Step 2: recompute GSR from the drift-residualized all-parcel signal, using
    # each run's time block recovered from the original (block-diagonal) GS
    # column's nonzero range, then OLS-regress it out of the drift-residualized
    # select_parcel signal
    Y_allparcel_mean_resid1 = Y_resid1_ap.sel(chromo=select_chromo).mean('parcel').values
    gsr_dm = copy.deepcopy(dm_all_cont_raw)
    gsr_dm.common = gsr_dm.common.sel(regressor=gs_regs)
    for reg_name in gs_regs:
        old_col = dm_all_cont_raw.common.sel(regressor=reg_name, chromo=select_chromo).values
        nz = np.nonzero(old_col)[0]
        i0, i1 = nz.min(), nz.max()
        new_col = np.zeros_like(old_col)
        new_col[i0:i1 + 1] = Y_allparcel_mean_resid1[i0:i1 + 1]
        gsr_dm.common.loc[dict(regressor=reg_name, chromo=select_chromo)] = new_col

    gsr_results = glm.fit(Y_resid1_1p, gsr_dm, noise_model='ols')
    gsr_fit_1p = xr.dot(gsr_dm.common, gsr_results.sm.params, dims='regressor').transpose(*Y_resid1_1p.dims)
    Y_resid2_1p = Y_resid1_1p - gsr_fit_1p

    # Step 3: AR-IRLS using only the EEG (bspline delay) regressors on the
    # twice-residualized signal
    bspline_dm = copy.deepcopy(dm_all_cont_raw)
    bspline_dm.common = bspline_dm.common.sel(regressor=bspline_regs)

    print(f"Start continuous-EEG AR-IRLS fitting on drift+GSR-residualized signal ({subject})")
    ar_results = glm.fit(Y_resid2_1p, bspline_dm, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params  # dims: parcel, chromo, regressor

    y_hat = xr.dot(bspline_dm.common, betas, dims='regressor')  # dims: time, chromo, parcel

    y_true_vals = Y_resid2_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_resid2_1p.time.values
    metrics = compute_fit_metrics(y_true_vals, bspline_dm.common, betas)

    # reconstruct the EEG delay-response curve: expand the fitted bspline-component
    # betas back to full per-delay-tap resolution via the saved basis_da
    betas_bspline = betas.rename({'regressor': 'component'})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims='component')
    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values  # (n_delay_taps,)

    return t, y_true_vals, y_hat_vals, bspline_dm, metrics, hrf_eeg


#%% ---- run both 3-step fits ----
t_event, y_true_event, y_hat_event, dm_event, metrics_event, hrf_event = three_step_fit_event_based()
t_cont, y_true_cont, y_hat_cont, dm_cont, metrics_cont, hrf_eeg = three_step_fit_continuous_eeg()

#%% ---- summarize metrics across both fits ----
metrics_df = pd.DataFrame({
    'event-based': metrics_event,
    'continuous EEG': metrics_cont,
}).T
metrics_df.index.name = 'fit'
print(f"\nFit metrics ({subject}, {select_parcel}, {select_chromo}), 3-step regression (drift -> GSR -> AR-IRLS):")
print(metrics_df.to_string(float_format=lambda x: f'{x:.4f}'))

#%% ---- visualize y_hat comparison: top = y_hat, bottom = residual ----
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

axs[0].plot(t_event, y_true_event, label='Y (drift+GSR-residualized, event-based)', color='k', linewidth=2)
axs[0].plot(t_event, y_hat_event, 'b', label='y_hat (event-based HRF GLM)', alpha=0.6)
axs[0].plot(t_cont, y_hat_cont, 'r', label='y_hat (continuous EEG GLM)', alpha=0.6)
axs[0].set_ylabel('HbO concentration')
axs[0].set_title(f'{subject} ({select_parcel}) — 3-step regression (OLS drift -> OLS GSR -> AR-IRLS)')
axs[0].legend()
axs[0].grid()

axs[1].plot(t_event, y_true_event - y_hat_event, 'b', label='Resid (event-based)', alpha=0.7)
axs[1].plot(t_cont, y_true_cont - y_hat_cont, 'r', label='Resid (continuous EEG)', alpha=0.7)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Residual')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

#%% ---- visualize the HRF: top = Fit 1 event-triggered HRF, bottom = Fit 2 EEG delay-response curve ----
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

for trial_type in hrf_event.trial_type.values:
    axs2[0].plot(hrf_event.time.values, hrf_event.sel(trial_type=trial_type).values, label=str(trial_type))
axs2[0].axhline(0, color='gray', linewidth=0.8)
axs2[0].set_ylabel('HbO concentration')
axs2[0].set_title(f'Fit 1 event-triggered HRF ({select_parcel})')
axs2[0].set_xlabel('Time from event onset (s)')
axs2[0].legend()
axs2[0].grid()

axs2[1].plot(t_delay, hrf_eeg, color='r')
axs2[1].axhline(0, color='gray', linewidth=0.8)
axs2[1].set_ylabel('Beta (HbO per unit EEG power)')
axs2[1].set_title(f'Fit 2 EEG delay-response curve ({select_parcel})')
axs2[1].set_xlabel('Delay (s)')
axs2[1].grid()

plt.tight_layout()
plt.show()

# %%
