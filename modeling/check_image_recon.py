
# %% Imports
##############################################################################
import os
import gzip
import pickle
import sys
import glob
import re
import copy

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.signal

import cedalion
import cedalion.nirs
import cedalion.math.ar_model
from cedalion import units
import cedalion.models.glm as glm
from cedalion.sigproc import quality
import cedalion.sigproc.frequency
from cedalion import units
# import cedalion.sigproc.motion_correct as motion
# from cedalion.plots import scalp_plot
from scipy.signal import filtfilt, windows

# import my own functions from a different directory
# sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
# import processing_func as pf
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules')
import processing_func as pf
# import image_recon_func as irf

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

#%%
subject = 'sub-723'
select_parcel='DorsAttnA_ParOcc_1_RH'
select_chromo='HbO'

#%% Initial directory and analysis parameters
SPLIT_VTC = False
SAVE_RESIDUAL = False
USE_GSR = False
NOISE_MODEL = 'ar_irls'
root_dir = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
ADOT_FLAG = 'probe'
weight_flag = 'post'
spatial_dim = 'vertex'
hrf_basis = 'cons_gaussians'# double_gamma_deriv, gamma_deriv, cons_gaussians
flag = ''
if NOISE_MODEL == 'ols':
    DO_TDDR = True
    DO_DRIFT = True
    DO_DRIFT_LEGENDRE = False
    DRIFT_ORDER = 3
    F_MIN = 0 * units.Hz
    F_MAX = 0.5 * units.Hz
elif NOISE_MODEL == 'ar_irls':
    DO_TDDR = False
    DO_DRIFT = False
    DO_DRIFT_LEGENDRE = True
    DRIFT_ORDER = 3
    F_MAX = 0
    F_MIN = 0
else:
    print('Not a valid noise model - please select ols or ar_irls')

cfg_GLM = {
    'do_drift': DO_DRIFT,
    'do_drift_legendre': DO_DRIFT_LEGENDRE,
    'do_short_sep': False,
    'drift_order' : DRIFT_ORDER,
    'do_GSR': USE_GSR,
    'GSR_weight': None,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    'HRF_basis': hrf_basis, 

    # double gamma deriv
    'peak_time': 4*units.s,
    'peak_disp': 1*units.s,
    'undershoot_time': 16*units.s,
    'undershoot_disp': 1*units.s,
    'ratio': 1/6,
    'duration': 18*units.s,

    # gamma deriv
    'tau': {'HbO': 1.8*units.s, 'HbR':2.5*units.s}, 
    'sigma': {'HbO':3*units.s, 'HbR':3*units.s}, 
    'dur': 4*units.s,

    # consecutive gaussians
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 18*units.s
    }

#%% RUN PREPROCESSING
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

print('LOADING PREPROCESSED CHANNEL DATA')
with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}.pkl'), 'rb') as f:
    results = pickle.load(f)

# all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']

print('LOADING IMAGE SPACE RESULTS')
folder =  os.path.join(der_dir, subject)
filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}.pkl'

with open(filepath, 'rb') as f:
    image_results = pickle.load(f)

if weight_flag == 'aca':
    all_runs = image_results['parcel_ts_aca']
    vv = image_results['vertex_aca']
elif weight_flag == 'post': 
    all_runs = image_results['parcel_ts_post']
    vv = image_results['vertex_mse']
else: 
    all_runs = image_results['parcel_ts_none']
    vv = image_results['vertex_mse']

n_runs = len(vv)
vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

if USE_GSR: 
    aca_lst = image_results['vertex_aca']
    aca_p_lst = []
    for aca in aca_lst:
        aca_p = aca.groupby('parcel').sum('vertex') / aca.groupby('parcel').count()**2
        aca_p = aca_p.sel(parcel = aca_p.parcel != 'scalp')
        aca_p_lst.append(aca_p)

    cfg_GLM['GSR_weight'] = aca_p_lst

#%% RUN HRF ESTIMATION
# L = 20  # <-- set this appropriately
# W = windows.gaussian(L, std=L/6) / 2  
                    
if SPLIT_VTC:
    possible_trial_types = ['mnt-correct-in', 'mnt-correct-out', 'mnt-incorrect', 'city-incorrect']
else:
    possible_trial_types = ['mnt-correct', 'mnt-incorrect', 'city-incorrect']    

trial_presence_list = []
stims_pruned_list = []

for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
    city_trials['trial_type'] = 'city-incorrect'
    
    if SPLIT_VTC:
        VTC = stim['VTC'].to_numpy()
        # VTC = filtfilt(W, sum(W), RT)
        median = np.median(VTC)

        in_zone = np.where(VTC <= median)[0]
        out_zone = np.where(VTC > median)[0]
        mnt_trials.loc[
                        (mnt_trials['trial_type'] == 'mnt-correct') & 
                        (mnt_trials.index.isin(in_zone)),
                        'trial_type'
                    ] = 'mnt-correct-in'

        mnt_trials.loc[
                        (mnt_trials['trial_type'] == 'mnt-correct') & 
                        (mnt_trials.index.isin(out_zone)),
                        'trial_type'
                    ] = 'mnt-correct-out'


    # Combine the filtered trials
    stims_pruned = pd.concat([mnt_trials, city_trials], ignore_index=True)
    # run.stim = stims_pruned
    stims_pruned_list.append(stims_pruned)

# run_ts_list = [image_results['parcel_ts_weights']]
all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

all_runs_tmp = []
for run in all_runs:
    run.time.attrs['units'] = units.s
    run = run.sel(parcel = run.parcel != 'scalp')
    all_runs_tmp.append(run)
all_runs = all_runs_tmp.copy()

# select only one parcel and one chromo
all_runs = [x.sel(parcel=[select_parcel], chromo=[select_chromo]) for x in all_runs]

results, hrf_estimate, hrf_mse, dms = pf.GLM(all_runs, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
Y_all, stim_df, runs_updated = pf.concatenate_runs(all_runs, stims_pruned_list)

# examine OLS model
NOISE_MODEL = 'ols'

DO_TDDR = True
DO_DRIFT = True
DO_DRIFT_LEGENDRE = False
DRIFT_ORDER = 3
F_MIN = 0 * units.Hz
F_MAX = 0.5 * units.Hz

cfg_GLM_ols = {
    'do_drift': DO_DRIFT,
    'do_drift_legendre': DO_DRIFT_LEGENDRE,
    'do_short_sep': False,
    'drift_order' : DRIFT_ORDER,
    'do_GSR': USE_GSR,
    'GSR_weight': None,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    'HRF_basis': hrf_basis, 

    # double gamma deriv
    'peak_time': 4*units.s,
    'peak_disp': 1*units.s,
    'undershoot_time': 16*units.s,
    'undershoot_disp': 1*units.s,
    'ratio': 1/6,
    'duration': 18*units.s,

    # gamma deriv
    'tau': {'HbO': 1.8*units.s, 'HbR':2.5*units.s}, 
    'sigma': {'HbO':3*units.s, 'HbR':3*units.s}, 
    'dur': 4*units.s,

    # consecutive gaussians
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 18*units.s
    }

results_ols, hrf_estimate_ols, hrf_mse_ols, dms_ols = pf.GLM(all_runs, cfg_GLM_ols, geo3d, all_chs_pruned, stims_pruned_list)

# visualize results
betas = results.sm.params
y_hat = (dms.common * betas).sum('regressor')
y_hat_laura = y_hat.transpose('chromo', 'parcel', 'time')

betas_ols = results_ols.sm.params
y_hat_ols = (dms_ols.common * betas_ols).sum('regressor')
y_hat_ols = y_hat_ols.transpose('chromo', 'parcel', 'time')

fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

y_true = Y_all.values.flatten()
y_hat_vals = y_hat_laura.values.flatten()
y_hat_ols_vals = y_hat_ols.values.flatten()

axs[0].plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=1)
axs[0].plot(y_hat_laura.time.values, y_hat_vals,
        label='Laura', alpha=0.7)
axs[0].plot(y_hat_ols.time.values, y_hat_ols_vals,
        label='OLS', alpha=0.7)
axs[0].set_ylabel(f'HbO concentration')
axs[0].set_title(f'Parcel activities estimation ({select_parcel})')
axs[0].legend()
axs[0].grid()

y_true_norm = (y_true - np.nanmean(y_true)) / np.nanstd(y_true)
y_hat_norm = (y_hat_vals - np.nanmean(y_hat_vals)) / np.nanstd(y_hat_vals)
y_hat_ols_norm = (y_hat_ols_vals - np.nanmean(y_hat_ols_vals)) / np.nanstd(y_hat_ols_vals)

axs[1].plot(Y_all.time.values, y_true_norm, label='Y (true, normalized)', color='k', linewidth=1)
axs[1].plot(y_hat_laura.time.values, y_hat_norm,
        label='Laura (normalized)', alpha=0.7)
axs[1].plot(y_hat_ols.time.values, y_hat_ols_norm,
        label='OLS (normalized)', alpha=0.7)
axs[1].set_ylabel('Normalized (z-score)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title(f'Normalized parcel activities estimation ({select_parcel})')
axs[1].legend()
axs[1].grid()


#%% High pass parcel signal before training
HPF_FREQ = 0.02 * units.Hz

all_runs_hpf = [
    cedalion.sigproc.frequency.freq_filter(run, fmin=HPF_FREQ, fmax=0 * units.Hz, butter_order=4)
    for run in all_runs
]

cfg_GLM_hpf = copy.deepcopy(cfg_GLM)
cfg_GLM_hpf['do_drift'] = False
cfg_GLM_hpf['do_drift_legendre'] = False
cfg_GLM_hpf['do_GSR'] = False
cfg_GLM_hpf['GSR_weight'] = None
cfg_GLM_hpf['noise_model'] = 'ar_irls'

cfg_GLM_ols_hpf = copy.deepcopy(cfg_GLM_ols)
cfg_GLM_ols_hpf['do_drift'] = False
cfg_GLM_ols_hpf['do_drift_legendre'] = False
cfg_GLM_ols_hpf['do_GSR'] = False
cfg_GLM_ols_hpf['GSR_weight'] = None
cfg_GLM_ols_hpf['noise_model'] = 'ols'

results_hpf, hrf_estimate_hpf, hrf_mse_hpf, dms_hpf = pf.GLM(all_runs_hpf, cfg_GLM_hpf, geo3d, all_chs_pruned, stims_pruned_list)
results_ols_hpf, hrf_estimate_ols_hpf, hrf_mse_ols_hpf, dms_ols_hpf = pf.GLM(all_runs_hpf, cfg_GLM_ols_hpf, geo3d, all_chs_pruned, stims_pruned_list)
Y_all_hpf, stim_df_hpf, runs_updated_hpf = pf.concatenate_runs(all_runs_hpf, stims_pruned_list)

betas_hpf = results_hpf.sm.params
y_hat_hpf = (dms_hpf.common * betas_hpf).sum('regressor')
y_hat_hpf = y_hat_hpf.transpose('chromo', 'parcel', 'time')

betas_ols_hpf = results_ols_hpf.sm.params
y_hat_ols_hpf = (dms_ols_hpf.common * betas_ols_hpf).sum('regressor')
y_hat_ols_hpf = y_hat_ols_hpf.transpose('chromo', 'parcel', 'time')

fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

y_true_hpf = Y_all_hpf.values.flatten()
y_hat_hpf_vals = y_hat_hpf.values.flatten()
y_hat_ols_hpf_vals = y_hat_ols_hpf.values.flatten()

axs[0].plot(Y_all_hpf.time.values, y_true_hpf, label='Y (true, HPF)', color='k', linewidth=1)
axs[0].plot(y_hat_hpf.time.values, y_hat_hpf_vals,
        label='Laura (HPF)', alpha=0.7)
axs[0].plot(y_hat_ols_hpf.time.values, y_hat_ols_hpf_vals,
        label='OLS (HPF)', alpha=0.7)
axs[0].set_ylabel(f'HbO concentration')
axs[0].set_title(f'Parcel activities estimation, high-pass filtered ({select_parcel})')
axs[0].legend()
axs[0].grid()

y_true_hpf_norm = (y_true_hpf - np.nanmean(y_true_hpf)) / np.nanstd(y_true_hpf)
y_hat_hpf_norm = (y_hat_hpf_vals - np.nanmean(y_hat_hpf_vals)) / np.nanstd(y_hat_hpf_vals)
y_hat_ols_hpf_norm = (y_hat_ols_hpf_vals - np.nanmean(y_hat_ols_hpf_vals)) / np.nanstd(y_hat_ols_hpf_vals)

axs[1].plot(Y_all_hpf.time.values, y_true_hpf_norm, label='Y (true, HPF, normalized)', color='k', linewidth=1)
axs[1].plot(y_hat_hpf.time.values, y_hat_hpf_norm,
        label='Laura (HPF, normalized)', alpha=0.7)
axs[1].plot(y_hat_ols_hpf.time.values, y_hat_ols_hpf_norm,
        label='OLS (HPF, normalized)', alpha=0.7)
axs[1].set_ylabel('Normalized (z-score)')
axs[1].set_xlabel('Time (s)')
axs[1].set_title(f'Normalized parcel activities estimation, high-pass filtered ({select_parcel})')
axs[1].legend()
axs[1].grid()

#%% compare AR-IRLS vs OLS betas (high-pass filtered fit)
betas_hpf_1d = betas_hpf.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify()
betas_ols_hpf_1d = betas_ols_hpf.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify()

betas_compare = pd.DataFrame({
    'ar_irls': betas_hpf_1d.to_pandas(),
    'ols': betas_ols_hpf_1d.to_pandas(),
})
betas_compare['diff'] = betas_compare['ar_irls'] - betas_compare['ols']
betas_compare['pct_diff'] = 100 * betas_compare['diff'] / betas_compare['ols'].abs()
print(betas_compare)

fig, axs = plt.subplots(2, 1, figsize=(14, 8))

reg_names = betas_compare.index.astype(str)
x_pos = np.arange(len(reg_names))
width = 0.35

axs[0].bar(x_pos - width / 2, betas_compare['ar_irls'].values, width, label='AR-IRLS')
axs[0].bar(x_pos + width / 2, betas_compare['ols'].values, width, label='OLS')
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(reg_names, rotation=90, fontsize=7)
axs[0].set_ylabel('Beta value')
axs[0].set_title(f'AR-IRLS vs OLS betas, high-pass filtered ({select_parcel})')
axs[0].legend()
axs[0].grid(axis='y')

axs[1].bar(x_pos, betas_compare['diff'].values, color='gray')
axs[1].set_xticks(x_pos)
axs[1].set_xticklabels(reg_names, rotation=90, fontsize=7)
axs[1].set_ylabel('AR-IRLS - OLS')
axs[1].set_title('Beta difference (AR-IRLS minus OLS)')
axs[1].grid(axis='y')
axs[1].axhline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

#%% visualize regressors in AR-IRLS for each iteration
dm_common_1d = dms_hpf.common.sel(chromo=select_chromo)
x = pd.DataFrame(
    dm_common_1d.transpose('time', 'regressor').values,
    columns=dm_common_1d.regressor.values,
)
y_1d = Y_all_hpf.pint.dequantify().sel(parcel=select_parcel, chromo=select_chromo)
y = pd.Series(y_1d.transpose('time').values)

mask = np.isfinite(y.values)
yorg = pd.Series(y.values[mask].copy())
xorg = x[mask].reset_index(drop=True)

M = sm.robust.norms.TukeyBiweight(c=4.685)
pmax = 30

y_ar = yorg.copy()
x_ar = xorg.copy()

rlm_model = sm.RLM(y_ar, x_ar, M=M)
params = rlm_model.fit()
resid = pd.Series(y_ar - x_ar @ params.params)

xf_list = [xorg.copy()]  # iteration 0: unwhitened regressors
y_hat_list = [pd.Series(xorg @ params.params)]  # iteration 0: fit from unwhitened regressors

for it in range(4):  # TODO - check convergence
    y_ar = yorg.copy()
    x_ar = xorg.copy()

    arcoef = cedalion.math.ar_model.bic_arfit(resid, pmax=pmax)
    wf = np.hstack([1, -arcoef.params[1:]])
    p = len(wf) - 1

    yf = pd.Series(scipy.signal.lfilter(wf, 1, y_ar))

    xf = np.zeros(x_ar.shape)
    xx = x_ar.to_numpy()
    for i in range(xx.shape[1]):
        xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i])

    xf = pd.DataFrame(xf)
    xf.columns = x_ar.columns

    rlm_model = sm.RLM(yf[p:], xf.iloc[p:], M=M)
    params = rlm_model.fit()

    resid = pd.Series(yorg - xorg @ params.params)
    xf_list.append(xf.copy())
    y_hat_list.append(pd.Series(xorg @ params.params))  # fit uses unwhitened x, current betas

all_cols = list(xorg.columns)
iter_tags = ['iteration 0 (before whitening)'] + [f'iteration {i + 1}' for i in range(4)]

for xf_it, tag in zip(xf_list, iter_tags):
    fig, axs = plt.subplots(len(all_cols), 1, figsize=(14, 2 * len(all_cols)), sharex=True)
    if len(all_cols) == 1:
        axs = [axs]
    for ax, col in zip(axs, all_cols):
        ax.plot(xf_it[col].values)
        ax.set_ylabel(col, fontsize=8)
        ax.grid()
    fig.suptitle(f'Regressors - {tag}')
    plt.tight_layout()
    plt.show()

#%% visualize regression fit (Y true vs y_hat) for each AR-IRLS iteration
fig, axs = plt.subplots(len(y_hat_list), 1, figsize=(14, 3 * len(y_hat_list)), sharex=True)
if len(y_hat_list) == 1:
    axs = [axs]
for ax, y_hat_it, tag in zip(axs, y_hat_list, iter_tags):
    ax.plot(yorg.values, label='Y (true)', color='k', linewidth=1)
    ax.plot(y_hat_it.values, label='y_hat', alpha=0.7)
    ax.set_title(tag)
    ax.legend()
    ax.grid()
fig.suptitle(f'AR-IRLS fit per iteration ({select_parcel})')
plt.tight_layout()
plt.show()


#%% hrf estimate
hrf_estimate = hrf_estimate.transpose('parcel', 'time', 'chromo', 'trial_type')
hrf_mse_uncorrected = hrf_mse.transpose('parcel', 'time', 'chromo', 'trial_type')

Y_all, _, _ = pf.concatenate_runs(all_runs, stims_pruned_list)

cov_params = results.sm.cov_params()
betas = results.sm.params
resid = Y_all.pint.dequantify() - xr.dot(dms.common, betas, dim='regressor')

var_resid = resid.var('time')
weight = (var_resid + vp.pint.dequantify()) / var_resid

if cfg_GLM['HRF_basis'] == 'cons_gaussians':    
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)
elif cfg_GLM['HRF_basis'] == 'gamma_deriv': 
    basis_hrf = glm.GammaDeriv(cfg_GLM['tau'], cfg_GLM['sigma'], cfg_GLM['dur'])(Y_all)
elif cfg_GLM['HRF_basis'] == 'double_gamma_deriv': 
    basis_hrf = glm.DoubleGammaDeriv(cfg_GLM['peak_time'], cfg_GLM['peak_disp'], 
                                            cfg_GLM['undershoot_time'], cfg_GLM['undershoot_disp'],
                                            cfg_GLM['ratio'], cfg_GLM['duration'])(Y_all)

hrf_mse_lst = []

for trial_type in hrf_estimate.trial_type.values:

    cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                                regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                )
    cov_params_reweighted = weight * cov_hrf
    hrf_mse_corrected = pf.estimate_HRF_cov(cov_params_reweighted, basis_hrf)

    hrf_mse_corrected = hrf_mse_corrected.expand_dims({'trial_type': [trial_type]})
    hrf_mse_lst.append(hrf_mse_corrected)

hrf_mse_corrected = xr.concat(hrf_mse_lst, dim='trial_type')
hrf_mse_corrected = hrf_mse_corrected.assign_coords({'time': hrf_estimate.time})
hrf_mse_corrected = hrf_mse_corrected.transpose('parcel', 'time', 'chromo', 'trial_type')

hrf_estimate = hrf_estimate.reindex({'trial_type': possible_trial_types})
hrf_mse_corrected = hrf_mse_corrected.reindex({'trial_type': possible_trial_types})
hrf_mse_uncorrected = hrf_mse_uncorrected.reindex({'trial_type': possible_trial_types})

hrf_per_subj = hrf_estimate.expand_dims('subj')
hrf_per_subj = hrf_per_subj.assign_coords(subj=subject)

hrf_mse_corr_per_subj = hrf_mse_corrected.expand_dims('subj')
hrf_mse_corr_per_subj = hrf_mse_corr_per_subj.assign_coords(subj=subject)

hrf_mse_uncorr_per_subj = hrf_mse_uncorrected.expand_dims('subj')
hrf_mse_uncorr_per_subj = hrf_mse_uncorr_per_subj.assign_coords(subj=subject)

print('HRF estimation complete')
# save per subject results concentration and then image recon will take and convert to OD 
if SPLIT_VTC:
    flag += '_VTC_split'

if USE_GSR: 
    flag += '_weightedGSR'

# flag += '_earlierhbo'
file_path_pkl = os.path.join(der_dir, subject,
                                f"{subject}_adot-{ADOT_FLAG}_hrf_estimates_{NOISE_MODEL}{flag}_{weight_flag}_{hrf_basis}.pkl.gz")

# save the individual results to a pickle file for image recon 
file = gzip.GzipFile(file_path_pkl, 'wb')
all_results = {
            'hrf_per_subj': hrf_per_subj,  # always unweighted   - load into img recon
            'hrf_mse_uncorr_per_subj': hrf_mse_uncorr_per_subj, # - load into img reconstructed
            'hrf_mse_corr_per_subj': hrf_mse_corr_per_subj, # - load into img reconstructed
        }

file.write(pickle.dumps(all_results))
file.close()

if SAVE_RESIDUAL:
    file_path_pkl = os.path.join(der_dir, subject,
                                f"{subject}_adot-{ADOT_FLAG}_glm_residual_{NOISE_MODEL}{flag}_{weight_flag}_{hrf_basis}.pkl")

    # residual = results.sm.resid
    with open(file_path_pkl, 'wb') as f:
        pickle.dump(resid, f)
    
    # if NOISE_MODEL == 'ar_irls':
    #     file_path_pkl = os.path.join(der_dir, subject,
    #                                 f"{subject}_adot-{ADOT_FLAG}_glm_weights_{NOISE_MODEL}{flag}_{weight_flag}_{hrf_basis}.pkl")

    #     weights = results.sm.weights
    #     with open(file_path_pkl, 'wb') as f:
    #         pickle.dump(weights, f)

print('Saved individual HRF to ' + file_path_pkl)

print('Job Complete.')
# %%
