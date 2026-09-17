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
import cedalion.io
from cedalion import units, nirs
from cedalion.sigproc import frequency
import cedalion.models.glm as glm
from scipy.signal import filtfilt, windows

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf

from params_setting import *

import warnings
warnings.filterwarnings('ignore')

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
# (matches the mask applied in run_model_cont_EEG_fNIRS.py, so Fit 1's all-parcel
# GSR and Fit 2's saved all-parcel GSR are computed over the same parcel set)
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = cedalion.io.load_Adot(Adot_path + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
mask_medial = Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH',
    'Background+FreeSurfer_Defined_Medial_Wall_RH'
])
Adot_brain = Adot_brain.sel(vertex=~mask_medial)

intensity = np.log10(Adot_brain[:, :, 1].sum('channel'))
sensitivity_mask = (intensity > -2).drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values)
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex')
Adot_parcel = Adot_parcel.assign_coords(
    {'is_brain': ('parcel', np.ones(len(Adot_parcel.parcel), dtype=bool))}
)
sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (601 -> 417)

# pick the single most-sensitive parcel (highest total forward-model sensitivity,
# summed over channel/wavelength/vertex) to fit, rather than a fixed name
parcel_total_sensitivity = Adot_parcel.sum(['channel', 'wavelength'])
most_sensitive_parcel = str(parcel_total_sensitivity.parcel.values[np.argmax(parcel_total_sensitivity.values)])

# also pick the most-sensitive parcel within the DorsAttn network, for reference on
# the brain-location plot only (not fit)
dorsattn_parcels = [p for p in parcel_total_sensitivity.parcel.values if str(p).startswith('DorsAttn')]
dorsattn_sens = parcel_total_sensitivity.sel(parcel=dorsattn_parcels)
most_sensitive_dorsattn_parcel = str(dorsattn_sens.parcel.values[np.argmax(dorsattn_sens.values)])

#%% subject / parcel selection (shared between both fits)
subject = 'sub-723'
select_parcel = "SalVentAttnA_FrMed_5_LH"
# select_parcel = most_sensitive_parcel
# select_parcel = most_sensitive_dorsattn_parcel
select_chromo = 'HbO'
eeg_reg_type = 'cont_EEG_cz_add_15s'
is_hp_fNIRS = False
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
len_delay = 15  # must match run_model_cont_EEG_fNIRS.py's len_delay, used for the crop window
HP_CUTOFF_Y = 1 / len_delay  # Hz; matches run_model_cont_EEG_fNIRS.py's is_hp_fNIRS cutoff convention

assert select_parcel in sensitive_parcels, f"{select_parcel} was excluded by the sensitivity mask"

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

# mask out parcels with low forward-model sensitivity (601 -> 417 parcels)
all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels)) for run in all_runs]

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
    per-run residuals (target minus the fitted regressor contribution) and the list
    of per-run fitted contributions themselves."""
    resid_runs = []
    fit_runs = []
    for run, dm in zip(target_runs, regressor_dms):
        results = glm.fit(run, dm, noise_model='ols')
        betas = results.sm.params  # dims: parcel, chromo, regressor
        fit_vals = xr.dot(dm.common, betas, dims='regressor').transpose(*run.dims)
        resid_runs.append(run - fit_vals)
        fit_runs.append(fit_vals)
    return resid_runs, fit_runs


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

    metrics = {'r2': r2, 'r2_partial': r2_partial, 'corr': corr, 'log_rmse': log_rmse}
    return metrics, y_denuisance, nonnuisance_fit


def three_step_fit_event_based():
    """3-step sequential regression for the event-based fit (Fit 1)."""
    # Step 1: OLS-regress out per-run Legendre drift, from both the single-parcel
    # target and the all-parcel copy (the latter is needed for step 2's GSR)
    drift_dms = [glm.design_matrix.drift_legendre_regressors(r, cfg_GLM['drift_order']) for r in all_runs]
    runs_resid1, runs_drift_fit = ols_regress_out_per_run(all_runs, drift_dms)
    runs_ap_resid1, _ = ols_regress_out_per_run(all_runs_allparcel, drift_dms)

    # Step 2: GSR computed from the drift-residualized all-parcel signal, then
    # OLS-regressed out of the drift-residualized single-parcel signal
    gsr_dms = pf.get_global_mean_regressor(runs_ap_resid1)
    runs_resid2, runs_gsr_fit = ols_regress_out_per_run(runs_resid1, gsr_dms)

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

    # no nuisance regressors remain in hrf_dm (already OLS-removed in steps 1-2), so
    # y_nonnuisance/y_hat_nonnuisance are just the twice-residualized signal and its
    # task-only fit (used for the "nuisance removed" row of the comparison plot)
    y_nonnuisance = Y_resid2.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_nonnuisance = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    metrics, _, _ = compute_fit_metrics(y_nonnuisance, hrf_dm.common, betas)

    # reconstruct the full-scale fit: Y_all (original, unresidualized) vs the sum of
    # the drift fit + GSR fit + task-only fit contributions, all on the same time axis
    Y_all_orig, _, _ = pf.concatenate_runs(all_runs, stims_pruned_list)
    drift_fit_full, _, _ = pf.concatenate_runs(runs_drift_fit, stims_pruned_list)
    gsr_fit_full, _, _ = pf.concatenate_runs(runs_gsr_fit, stims_pruned_list)

    t = Y_all_orig.time.values
    y_true_vals = Y_all_orig.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    drift_fit_vals = drift_fit_full.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    gsr_fit_vals = gsr_fit_full.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = drift_fit_vals + gsr_fit_vals + y_hat_nonnuisance

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

    return t, y_true_vals, y_hat_vals, hrf_dm, metrics, y_nonnuisance, y_hat_nonnuisance, hrf


def one_stage_fit_event_based():
    """1-stage reference fit for Fit 1: drift + all-parcel GSR + HRF regressors all
    fit jointly in a single AR-IRLS call (no highpass, no sequential OLS removal) --
    i.e. the pre-3-stage combined design, for comparing against the 3-stage HRF."""
    gsr_per_run = pf.get_global_mean_regressor(all_runs_allparcel)
    CURRENT_OFFSET = 0
    gsr_shifted = []
    for run_ap, gsr in zip(all_runs_allparcel, gsr_per_run):
        time = run_ap.time.values
        new_time = time + CURRENT_OFFSET
        gsr_shifted.append(gsr.common.assign_coords(time=new_time))
        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])
    gsr_allparcel = xr.concat(gsr_shifted, dim='time')

    cfg = dict(cfg_GLM)
    cfg['do_GSR'] = False  # inject the all-parcel GSR manually instead of cfg_GLM's own single-parcel GSR

    # pf.GLM needs pint-quantified input (it derives hrf_mse's units from Y_all's
    # units); all_runs was dequantified earlier in this script for the 3-stage math
    all_runs_quantified = [r.pint.quantify('molar') for r in all_runs]

    print(f"Start 1-stage event-based AR-IRLS fitting (combined drift+GSR+HRF, no highpass) ({subject})")
    glm_results, hrf_estimate, hrf_mse, dms = pf.GLM(
        all_runs_quantified, cfg, geo3d, all_chs_pruned, stims_pruned_list, regressors=gsr_allparcel)
    Y_all, stim_df, runs_updated = pf.concatenate_runs(all_runs_quantified, stims_pruned_list)

    betas = glm_results.sm.params
    y_hat = (dms.common * betas).sum('regressor').transpose(*Y_all.dims)

    y_true_vals = Y_all.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify().values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_all.time.values
    metrics, y_nonnuisance, y_hat_nonnuisance = compute_fit_metrics(y_true_vals, dms.common, betas)

    hrf = hrf_estimate.sel(parcel=select_parcel, chromo=select_chromo)  # dims: time, trial_type
    return t, y_true_vals, y_hat_vals, metrics, y_nonnuisance, y_hat_nonnuisance, hrf


def highpass_fit_event_based():
    """2-step regression for the event-based fit (Fit 1), using a highpass filter on
    Y instead of OLS drift removal: (1) highpass-filter Y, (2) OLS-regress out GSR
    (computed from the highpassed all-parcel signal), (3) AR-IRLS-fit only the HRF
    regressors -- drift regressors are NOT included, since the highpass filter
    already removes the slow drift they would otherwise model."""
    # Step 1: highpass-filter Y (single-parcel target and all-parcel copy, the
    # latter needed for step 2's GSR), replacing OLS drift removal
    runs_hp = [frequency.freq_filter(r, HP_CUTOFF_Y * units.Hz, 0 * units.Hz) for r in all_runs]
    runs_ap_hp = [frequency.freq_filter(r, HP_CUTOFF_Y * units.Hz, 0 * units.Hz) for r in all_runs_allparcel]

    # Step 2: GSR computed from the highpassed all-parcel signal, then OLS-regressed
    # out of the highpassed single-parcel signal
    gsr_dms = pf.get_global_mean_regressor(runs_ap_hp)
    runs_resid2, runs_gsr_fit = ols_regress_out_per_run(runs_hp, gsr_dms)

    # Step 3: AR-IRLS using only the event-based HRF regressors (no drift regressors)
    # on the highpass+GSR-residualized signal
    Y_resid2, stim_df, runs_resid2_updated = pf.concatenate_runs(runs_resid2, stims_pruned_list)
    hrf_kernel = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    hrf_dm = glm.design_matrix.hrf_regressors(Y_resid2, stim_df, hrf_kernel)
    hrf_dm.common = hrf_dm.common.fillna(0)

    print(f"Start event-based AR-IRLS fitting on highpass+GSR-residualized signal ({subject})")
    ar_results = glm.fit(Y_resid2, hrf_dm, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params
    y_hat = xr.dot(hrf_dm.common, betas, dims='regressor').transpose(*Y_resid2.dims)

    # no nuisance regressors remain in hrf_dm, so y_nonnuisance/y_hat_nonnuisance are
    # just the highpass+GSR-residualized signal and its task-only fit
    y_nonnuisance = Y_resid2.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_nonnuisance = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    metrics, _, _ = compute_fit_metrics(y_nonnuisance, hrf_dm.common, betas)

    # reconstruct the highpassed-scale fit: highpassed Y_all vs the sum of the GSR
    # fit + task-only fit contributions (no drift term -- already removed by the filter)
    Y_all_hp, _, _ = pf.concatenate_runs(runs_hp, stims_pruned_list)
    gsr_fit_full, _, _ = pf.concatenate_runs(runs_gsr_fit, stims_pruned_list)

    t = Y_all_hp.time.values
    y_true_vals = Y_all_hp.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    gsr_fit_vals = gsr_fit_full.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = gsr_fit_vals + y_hat_nonnuisance

    # event-triggered HRF shape, per trial type
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

    return t, y_true_vals, y_hat_vals, hrf_dm, metrics, y_nonnuisance, y_hat_nonnuisance, hrf


def highpass_one_stage_fit_event_based():
    """1-stage reference fit for Fit 1 using a highpass filter instead of drift
    regressors: GSR (all-parcel, recomputed from the highpassed signal) + HRF
    regressors fit jointly in a single AR-IRLS call on the highpassed Y, with drift
    regressors disabled (redundant with, and would fight, the highpass filter)."""
    runs_hp = [frequency.freq_filter(r, HP_CUTOFF_Y * units.Hz, 0 * units.Hz) for r in all_runs]
    runs_ap_hp = [frequency.freq_filter(r, HP_CUTOFF_Y * units.Hz, 0 * units.Hz) for r in all_runs_allparcel]

    gsr_per_run = pf.get_global_mean_regressor(runs_ap_hp)
    CURRENT_OFFSET = 0
    gsr_shifted = []
    for run_ap, gsr in zip(runs_ap_hp, gsr_per_run):
        time = run_ap.time.values
        new_time = time + CURRENT_OFFSET
        gsr_shifted.append(gsr.common.assign_coords(time=new_time))
        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])
    gsr_allparcel = xr.concat(gsr_shifted, dim='time')

    cfg = dict(cfg_GLM)
    cfg['do_GSR'] = False  # inject the all-parcel GSR manually instead of cfg_GLM's own single-parcel GSR
    cfg['do_drift'] = False  # drift already removed by the highpass filter on Y
    cfg['do_drift_legendre'] = False

    runs_hp_quantified = [r.pint.quantify('molar') for r in runs_hp]

    print(f"Start 1-stage event-based AR-IRLS fitting (combined GSR+HRF, highpass, no drift) ({subject})")
    glm_results, hrf_estimate, hrf_mse, dms = pf.GLM(
        runs_hp_quantified, cfg, geo3d, all_chs_pruned, stims_pruned_list, regressors=gsr_allparcel)
    Y_all, stim_df, runs_updated = pf.concatenate_runs(runs_hp_quantified, stims_pruned_list)

    betas = glm_results.sm.params
    y_hat = (dms.common * betas).sum('regressor').transpose(*Y_all.dims)

    y_true_vals = Y_all.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify().values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_all.time.values
    metrics, y_nonnuisance, y_hat_nonnuisance = compute_fit_metrics(y_true_vals, dms.common, betas)

    hrf = hrf_estimate.sel(parcel=select_parcel, chromo=select_chromo)  # dims: time, trial_type
    return t, y_true_vals, y_hat_vals, metrics, y_nonnuisance, y_hat_nonnuisance, hrf


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

# mask out parcels with low forward-model sensitivity (601 -> 417 parcels), so any
# all-parcel GSR computed from Y_all_cont_raw below matches Fit 1's masked parcel set
Y_all_cont_raw = Y_all_cont_raw.sel(parcel=Y_all_cont_raw.parcel.isin(sensitive_parcels))


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

    # no nuisance regressors remain in bspline_dm (already OLS-removed in steps 1-2),
    # so y_nonnuisance/y_hat_nonnuisance are just the twice-residualized signal and
    # its task-only fit (used for the "nuisance removed" row of the comparison plot)
    y_nonnuisance = Y_resid2_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_nonnuisance = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    metrics, _, _ = compute_fit_metrics(y_nonnuisance, bspline_dm.common, betas)

    # reconstruct the full-scale fit: Y_all (original, unresidualized) vs the sum of
    # the drift fit + GSR fit + task-only fit contributions, all on the same time axis
    t = Y_all_cont_parcel.time.values
    y_true_vals = Y_all_cont_parcel.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    drift_fit_vals = drift_fit_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    gsr_fit_vals = gsr_fit_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = drift_fit_vals + gsr_fit_vals + y_hat_nonnuisance

    # reconstruct the EEG delay-response curve: expand the fitted bspline-component
    # betas back to full per-delay-tap resolution via the saved basis_da
    betas_bspline = betas.rename({'regressor': 'component'})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims='component')
    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values  # (n_delay_taps,)

    return t, y_true_vals, y_hat_vals, bspline_dm, metrics, y_nonnuisance, y_hat_nonnuisance, hrf_eeg


def one_stage_fit_continuous_eeg():
    """1-stage reference fit for Fit 2: drift + GSR + EEG (bspline) regressors all
    fit jointly in a single AR-IRLS call, using dm_all_cont_raw exactly as saved by
    run_model_cont_EEG_fNIRS.py (no highpass, no sequential OLS removal), except the
    GSR columns are recomputed from the sensitivity-masked (417-parcel) Y_all_cont_raw
    rather than reusing the saved GSR (computed at save-time from all 601 parcels)."""
    Y_all_cont_parcel = Y_all_cont_raw.sel(parcel=[select_parcel])

    # recompute each run's GS regressor from the masked all-parcel signal, using
    # that run's time block recovered from the saved GS column's nonzero range
    gs_regs = [r for r in dm_all_cont_raw.common.regressor.values if str(r).startswith('GS run')]
    dm_1stage = copy.deepcopy(dm_all_cont_raw)
    Y_allparcel_mean = Y_all_cont_raw.sel(chromo=select_chromo).mean('parcel').values
    for reg_name in gs_regs:
        old_col = dm_all_cont_raw.common.sel(regressor=reg_name, chromo=select_chromo).values
        nz = np.nonzero(old_col)[0]
        i0, i1 = nz.min(), nz.max()
        new_col = np.zeros_like(old_col)
        new_col[i0:i1 + 1] = Y_allparcel_mean[i0:i1 + 1]
        dm_1stage.common.loc[dict(regressor=reg_name, chromo=select_chromo)] = new_col

    print(f"Start 1-stage continuous-EEG AR-IRLS fitting (combined drift+GSR+EEG, no highpass) ({subject})")
    ar_results = glm.fit(Y_all_cont_parcel, dm_1stage, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params  # dims: parcel, chromo, regressor

    y_hat = xr.dot(dm_1stage.common, betas, dims='regressor')  # dims: time, chromo, parcel

    y_true_vals = Y_all_cont_parcel.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_all_cont_parcel.time.values
    metrics, y_nonnuisance, y_hat_nonnuisance = compute_fit_metrics(y_true_vals, dm_1stage.common, betas)

    bspline_regs = [r for r in dm_1stage.common.regressor.values if str(r).startswith('bspline')]
    betas_bspline = betas.sel(regressor=bspline_regs).rename({'regressor': 'component'})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims='component')
    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values  # (n_delay_taps,)

    return t, y_true_vals, y_hat_vals, metrics, y_nonnuisance, y_hat_nonnuisance, hrf_eeg


def highpass_fit_continuous_eeg():
    """2-step regression for the continuous-EEG fit (Fit 2), using a highpass filter
    on Y instead of OLS drift removal; drift regressors are dropped from the design
    matrix entirely, since the highpass filter already removes the slow drift.
    Note: Y_all_cont_raw is filtered across the whole concatenated series (per-run
    boundaries aren't recoverable from its time axis alone), same limitation already
    accepted elsewhere in this script for continuous-EEG-derived quantities."""
    reg_names_all = [str(r) for r in dm_all_cont_raw.common.regressor.values]
    gs_regs = [r for r in reg_names_all if r.startswith('GS run')]
    bspline_regs = [r for r in reg_names_all if r.startswith('bspline')]

    Y_all_cont_parcel = Y_all_cont_raw.sel(parcel=[select_parcel])

    # Step 1: highpass-filter Y (select_parcel and all-parcel), replacing OLS drift removal
    Y_hp_1p = frequency.freq_filter(Y_all_cont_parcel, HP_CUTOFF_Y * units.Hz, 0 * units.Hz)
    Y_hp_ap = frequency.freq_filter(Y_all_cont_raw, HP_CUTOFF_Y * units.Hz, 0 * units.Hz)

    # Step 2: recompute GSR from the highpassed all-parcel signal, using each run's
    # time block recovered from the original (block-diagonal) GS column's nonzero
    # range, then OLS-regress it out of the highpassed select_parcel signal
    Y_allparcel_mean_hp = Y_hp_ap.sel(chromo=select_chromo).mean('parcel').values
    gsr_dm = copy.deepcopy(dm_all_cont_raw)
    gsr_dm.common = gsr_dm.common.sel(regressor=gs_regs)
    for reg_name in gs_regs:
        old_col = dm_all_cont_raw.common.sel(regressor=reg_name, chromo=select_chromo).values
        nz = np.nonzero(old_col)[0]
        i0, i1 = nz.min(), nz.max()
        new_col = np.zeros_like(old_col)
        new_col[i0:i1 + 1] = Y_allparcel_mean_hp[i0:i1 + 1]
        gsr_dm.common.loc[dict(regressor=reg_name, chromo=select_chromo)] = new_col

    gsr_results = glm.fit(Y_hp_1p, gsr_dm, noise_model='ols')
    gsr_fit_1p = xr.dot(gsr_dm.common, gsr_results.sm.params, dims='regressor').transpose(*Y_hp_1p.dims)
    Y_resid2_1p = Y_hp_1p - gsr_fit_1p

    # Step 3: AR-IRLS using only the EEG (bspline delay) regressors -- no drift
    # regressors, since drift was already removed by the highpass filter
    bspline_dm = copy.deepcopy(dm_all_cont_raw)
    bspline_dm.common = bspline_dm.common.sel(regressor=bspline_regs)

    print(f"Start continuous-EEG AR-IRLS fitting on highpass+GSR-residualized signal ({subject})")
    ar_results = glm.fit(Y_resid2_1p, bspline_dm, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params  # dims: parcel, chromo, regressor

    y_hat = xr.dot(bspline_dm.common, betas, dims='regressor')  # dims: time, chromo, parcel

    y_nonnuisance = Y_resid2_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_nonnuisance = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    metrics, _, _ = compute_fit_metrics(y_nonnuisance, bspline_dm.common, betas)

    # reconstruct the highpassed-scale fit: highpassed Y vs the sum of the GSR fit +
    # task-only fit contributions (no drift term -- already removed by the filter)
    t = Y_hp_1p.time.values
    y_true_vals = Y_hp_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    gsr_fit_vals = gsr_fit_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = gsr_fit_vals + y_hat_nonnuisance

    # reconstruct the EEG delay-response curve: expand the fitted bspline-component
    # betas back to full per-delay-tap resolution via the saved basis_da
    betas_bspline = betas.rename({'regressor': 'component'})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims='component')
    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values  # (n_delay_taps,)

    return t, y_true_vals, y_hat_vals, bspline_dm, metrics, y_nonnuisance, y_hat_nonnuisance, hrf_eeg


def highpass_one_stage_fit_continuous_eeg():
    """1-stage reference fit for Fit 2 using a highpass filter instead of drift
    regressors: GSR (recomputed from the highpassed all-parcel signal) + EEG
    (bspline) regressors fit jointly in a single AR-IRLS call, with drift
    regressors dropped entirely from the design matrix."""
    Y_all_cont_parcel = Y_all_cont_raw.sel(parcel=[select_parcel])
    Y_hp_1p = frequency.freq_filter(Y_all_cont_parcel, HP_CUTOFF_Y * units.Hz, 0 * units.Hz)
    Y_hp_ap = frequency.freq_filter(Y_all_cont_raw, HP_CUTOFF_Y * units.Hz, 0 * units.Hz)

    reg_names_all = [str(r) for r in dm_all_cont_raw.common.regressor.values]
    gs_regs = [r for r in reg_names_all if r.startswith('GS run')]
    non_drift_regs = [r for r in reg_names_all if not r.startswith('Drift')]

    dm_1stage = copy.deepcopy(dm_all_cont_raw)
    dm_1stage.common = dm_1stage.common.sel(regressor=non_drift_regs)  # drop drift: already removed by highpass

    Y_allparcel_mean_hp = Y_hp_ap.sel(chromo=select_chromo).mean('parcel').values
    for reg_name in gs_regs:
        old_col = dm_all_cont_raw.common.sel(regressor=reg_name, chromo=select_chromo).values
        nz = np.nonzero(old_col)[0]
        i0, i1 = nz.min(), nz.max()
        new_col = np.zeros_like(old_col)
        new_col[i0:i1 + 1] = Y_allparcel_mean_hp[i0:i1 + 1]
        dm_1stage.common.loc[dict(regressor=reg_name, chromo=select_chromo)] = new_col

    print(f"Start 1-stage continuous-EEG AR-IRLS fitting (combined GSR+EEG, highpass, no drift) ({subject})")
    ar_results = glm.fit(Y_hp_1p, dm_1stage, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params  # dims: parcel, chromo, regressor

    y_hat = xr.dot(dm_1stage.common, betas, dims='regressor')  # dims: time, chromo, parcel

    y_true_vals = Y_hp_1p.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    y_hat_vals = y_hat.sel(parcel=select_parcel, chromo=select_chromo).values.flatten()
    t = Y_hp_1p.time.values
    metrics, y_nonnuisance, y_hat_nonnuisance = compute_fit_metrics(y_true_vals, dm_1stage.common, betas)

    bspline_regs = [r for r in dm_1stage.common.regressor.values if str(r).startswith('bspline')]
    betas_bspline = betas.sel(regressor=bspline_regs).rename({'regressor': 'component'})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims='component')
    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values  # (n_delay_taps,)

    return t, y_true_vals, y_hat_vals, metrics, y_nonnuisance, y_hat_nonnuisance, hrf_eeg


#%% ---- run both 3-step fits ----
(t_event, y_true_event, y_hat_event, dm_event, metrics_event,
 y_nonnuisance_event, y_hat_nonnuisance_event, hrf_event) = three_step_fit_event_based()
(t_cont, y_true_cont, y_hat_cont, dm_cont, metrics_cont,
 y_nonnuisance_cont, y_hat_nonnuisance_cont, hrf_eeg) = three_step_fit_continuous_eeg()

#%% ---- run both 1-stage (no highpass) reference fits ----
(t_event_1stage, y_true_event_1stage, y_hat_event_1stage, metrics_event_1stage,
 y_nonnuisance_event_1stage, y_hat_nonnuisance_event_1stage, hrf_event_1stage) = one_stage_fit_event_based()
(t_cont_1stage, y_true_cont_1stage, y_hat_cont_1stage, metrics_cont_1stage,
 y_nonnuisance_cont_1stage, y_hat_nonnuisance_cont_1stage, hrf_eeg_1stage) = one_stage_fit_continuous_eeg()

#%% ---- run both highpass fits (2-step and 1-stage, no drift regressors) ----
(t_event_hp, y_true_event_hp, y_hat_event_hp, dm_event_hp, metrics_event_hp,
 y_nonnuisance_event_hp, y_hat_nonnuisance_event_hp, hrf_event_hp) = highpass_fit_event_based()
(t_cont_hp, y_true_cont_hp, y_hat_cont_hp, dm_cont_hp, metrics_cont_hp,
 y_nonnuisance_cont_hp, y_hat_nonnuisance_cont_hp, hrf_eeg_hp) = highpass_fit_continuous_eeg()

(t_event_hp_1stage, y_true_event_hp_1stage, y_hat_event_hp_1stage, metrics_event_hp_1stage,
 y_nonnuisance_event_hp_1stage, y_hat_nonnuisance_event_hp_1stage, hrf_event_hp_1stage) = highpass_one_stage_fit_event_based()
(t_cont_hp_1stage, y_true_cont_hp_1stage, y_hat_cont_hp_1stage, metrics_cont_hp_1stage,
 y_nonnuisance_cont_hp_1stage, y_hat_nonnuisance_cont_hp_1stage, hrf_eeg_hp_1stage) = highpass_one_stage_fit_continuous_eeg()

#%% ---- summarize metrics across both fits and all regression strategies ----
metrics_df = pd.DataFrame({
    ('event-based', '3-step'): metrics_event,
    ('event-based', '1-stage'): metrics_event_1stage,
    ('event-based', '2-step-HP'): metrics_event_hp,
    ('event-based', '1-stage-HP'): metrics_event_hp_1stage,
    ('continuous EEG', '3-step'): metrics_cont,
    ('continuous EEG', '1-stage'): metrics_cont_1stage,
    ('continuous EEG', '2-step-HP'): metrics_cont_hp,
    ('continuous EEG', '1-stage-HP'): metrics_cont_hp_1stage,
}).T
metrics_df.index.names = ['fit', 'regression']
print(f"\nFit metrics ({subject}, {select_parcel}, {select_chromo}):")
print(metrics_df.to_string(float_format=lambda x: f'{x:.4f}'))

#%% ---- visualize Y vs y_hat: col 1 = 1-stage, col 2 = 3-stage;
# row 1 = Y_all vs y_hat_all, row 2 = Y_nonnuisance vs y_hat_nonnuisance ----
fig, axs = plt.subplots(2, 2, figsize=(20, 8), sharex=False)

cols = [
    ('1-stage (combined drift+GSR+task, single AR-IRLS, no highpass)',
     t_event_1stage, y_true_event_1stage, y_hat_event_1stage, y_nonnuisance_event_1stage, y_hat_nonnuisance_event_1stage,
     t_cont_1stage, y_hat_cont_1stage, y_hat_nonnuisance_cont_1stage),
    ('3-step regression (OLS drift -> OLS GSR -> AR-IRLS)',
     t_event, y_true_event, y_hat_event, y_nonnuisance_event, y_hat_nonnuisance_event,
     t_cont, y_hat_cont, y_hat_nonnuisance_cont),
]

for col_i, (col_label, t_ev, y_true_ev, y_hat_ev, y_nn_ev, y_hat_nn_ev,
            t_c, y_hat_c, y_hat_nn_c) in enumerate(cols):
    ax_top = axs[0, col_i]
    ax_bot = axs[1, col_i]

    ax_top.plot(t_ev, y_true_ev, label='Y (event-based)', color='k', linewidth=2)
    ax_top.plot(t_ev, y_hat_ev, 'b', label='y_hat (event-based HRF GLM)', alpha=0.6)
    ax_top.plot(t_c, y_hat_c, 'r', label='y_hat (continuous EEG GLM)', alpha=0.6)
    ax_top.set_ylabel('HbO concentration')
    ax_top.set_title(f'{subject} ({select_parcel}) — {col_label}')
    ax_top.legend()
    ax_top.grid()

    ax_bot.plot(t_ev, y_nn_ev, label='Y_nonnuisance (event-based)', color='k', linewidth=2)
    ax_bot.plot(t_ev, y_hat_nn_ev, 'b', label='y_hat_nonnuisance (event-based HRF GLM)', alpha=0.6)
    ax_bot.plot(t_c, y_hat_nn_c, 'r', label='y_hat_nonnuisance (continuous EEG GLM)', alpha=0.6)
    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('HbO concentration')
    ax_bot.set_title(f'{subject} ({select_parcel}) — {col_label} (nuisance removed)')
    ax_bot.legend()
    ax_bot.grid()

plt.tight_layout()
plt.show()

#%% ---- visualize Y vs y_hat for the highpass models: col 1 = 1-stage-HP, col 2 = 2-step-HP;
# row 1 = Y_all (highpassed) vs y_hat_all, row 2 = Y_nonnuisance vs y_hat_nonnuisance ----
fig_hp, axs_hp = plt.subplots(2, 2, figsize=(20, 8), sharex=False)

cols_hp = [
    ('1-stage-HP (combined GSR+task, single AR-IRLS, highpass, no drift)',
     t_event_hp_1stage, y_true_event_hp_1stage, y_hat_event_hp_1stage,
     y_nonnuisance_event_hp_1stage, y_hat_nonnuisance_event_hp_1stage,
     t_cont_hp_1stage, y_hat_cont_hp_1stage, y_hat_nonnuisance_cont_hp_1stage),
    ('2-step-HP (highpass -> OLS GSR -> AR-IRLS, no drift)',
     t_event_hp, y_true_event_hp, y_hat_event_hp, y_nonnuisance_event_hp, y_hat_nonnuisance_event_hp,
     t_cont_hp, y_hat_cont_hp, y_hat_nonnuisance_cont_hp),
]

for col_i, (col_label, t_ev, y_true_ev, y_hat_ev, y_nn_ev, y_hat_nn_ev,
            t_c, y_hat_c, y_hat_nn_c) in enumerate(cols_hp):
    ax_top = axs_hp[0, col_i]
    ax_bot = axs_hp[1, col_i]

    ax_top.plot(t_ev, y_true_ev, label='Y (event-based, highpassed)', color='k', linewidth=2)
    ax_top.plot(t_ev, y_hat_ev, 'b', label='y_hat (event-based HRF GLM)', alpha=0.6)
    ax_top.plot(t_c, y_hat_c, 'r', label='y_hat (continuous EEG GLM)', alpha=0.6)
    ax_top.set_ylabel('HbO concentration')
    ax_top.set_title(f'{subject} ({select_parcel}) — {col_label}')
    ax_top.legend()
    ax_top.grid()

    ax_bot.plot(t_ev, y_nn_ev, label='Y_nonnuisance (event-based)', color='k', linewidth=2)
    ax_bot.plot(t_ev, y_hat_nn_ev, 'b', label='y_hat_nonnuisance (event-based HRF GLM)', alpha=0.6)
    ax_bot.plot(t_c, y_hat_nn_c, 'r', label='y_hat_nonnuisance (continuous EEG GLM)', alpha=0.6)
    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('HbO concentration')
    ax_bot.set_title(f'{subject} ({select_parcel}) — {col_label} (nuisance removed)')
    ax_bot.legend()
    ax_bot.grid()

plt.tight_layout()
plt.show()

#%% ---- visualize the HRF: col 1 = no-highpass models, col 2 = highpass models;
# top = Fit 1 event-triggered HRF, bottom = Fit 2 EEG delay-response curve.
# solid = 3-step/2-step-HP, dashed = 1-stage/1-stage-HP reference, same color per trial type
fig2, axs2 = plt.subplots(2, 2, figsize=(20, 8), sharex=False)

hrf_cols = [
    ('no highpass', hrf_event, '3-step', hrf_event_1stage, '1-stage', hrf_eeg, hrf_eeg_1stage),
    ('highpass', hrf_event_hp, '2-step-HP', hrf_event_hp_1stage, '1-stage-HP', hrf_eeg_hp, hrf_eeg_hp_1stage),
]

for col_i, (col_label, hrf_ev, ev_label, hrf_ev_1stage, ev_1stage_label, eeg_curve, eeg_curve_1stage) in enumerate(hrf_cols):
    ax_top = axs2[0, col_i]
    ax_bot = axs2[1, col_i]

    for trial_type in hrf_ev.trial_type.values:
        line, = ax_top.plot(hrf_ev.time.values, hrf_ev.sel(trial_type=trial_type).values,
                             label=f'{trial_type} ({ev_label})')
        ax_top.plot(hrf_ev_1stage.time.values, hrf_ev_1stage.sel(trial_type=trial_type).values,
                    color=line.get_color(), linestyle='--', label=f'{trial_type} ({ev_1stage_label})')
    ax_top.axhline(0, color='gray', linewidth=0.8)
    ax_top.set_ylabel('HbO concentration')
    ax_top.set_title(f'Fit 1 event-triggered HRF ({select_parcel}) — {col_label}')
    ax_top.set_xlabel('Time from event onset (s)')
    ax_top.legend()
    ax_top.grid()

    ax_bot.plot(t_delay, eeg_curve, color='r', label=ev_label)
    ax_bot.plot(t_delay, eeg_curve_1stage, color='r', linestyle='--', label=ev_1stage_label)
    ax_bot.axhline(0, color='gray', linewidth=0.8)
    ax_bot.set_ylabel('Beta (HbO per unit EEG power)')
    ax_bot.set_title(f'Fit 2 EEG delay-response curve ({select_parcel}) — {col_label}')
    ax_bot.set_xlabel('Delay (s)')
    ax_bot.legend()
    ax_bot.grid()

plt.tight_layout()
plt.show()

#%% ---- visualize where select_parcel is located on the brain surface ----
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

highlight_vals = np.where(vertex_parcel == select_parcel, 1.0, 0.0)
X_highlight = xr.DataArray(
    np.stack([highlight_vals, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)

parcel_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'parcel_location')
os.makedirs(parcel_plot_dir, exist_ok=True)
parcel_plot_path = os.path.join(parcel_plot_dir, f'{subject}_{select_parcel}_location')
image_recon_multi_view(
    X_ts=X_highlight, head=head, cmap='Reds', clim=(0, 1),
    view_type='hbo_brain',
    title_str=select_parcel,
    SAVE=True, filename=parcel_plot_path,
    wdw_size=(1600, 800),
)
print(f'Saved parcel location plot to {parcel_plot_path}.png')

# %%
