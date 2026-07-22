
# %% Imports
##############################################################################
import os
import gzip
import pickle
import sys
import glob 
import re 

import pandas as pd 
import numpy as np 
import xarray as xr
import cedalion.xrutils as xrutils
import copy
import matplotlib.pyplot as plt

import cedalion
import cedalion.nirs
from cedalion import units
import cedalion.models.glm as glm
from cedalion.sigproc import quality
from cedalion import units
from scipy.signal import filtfilt, windows

import model

# import my own functions from a different directory
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules')
import processing_func as pf

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

#%%
subject = 'sub-723'

#%% Initial root directory and analysis parameters
RUN_PREPROCESS = False
RUN_HRF_ESTIMATION = True
SPLIT_VTC = False
SAVE_RESIDUAL = False
NOISE_MODEL = 'ar_irls'
root_dir = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
flag = '_chanOD'

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
    'do_short_sep': True,
    'drift_order' : DRIFT_ORDER,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 18*units.s
    }

run_files = glob.glob(os.path.join(root_dir,  subject, 'nirs',  f"{subject}_task-gradCPT_run-*_nirs.snirf"))
runs_with_events = []
for run_file in run_files:
    # extract run number using regex
    match = re.search(r"run-(\d+)", os.path.basename(run_file))
    if match:
        run_num = match.group(1)
        events_file = os.path.join(root_dir, subject, 'nirs',  f"{subject}_task-gradCPT_run-{run_num}_events.tsv")

        if os.path.exists(events_file):
            runs_with_events.append(run_num)

print(f"{subject}: runs with events = {runs_with_events}")

cfg_dataset = {

    'root_dir' : root_dir,
    'subj_ids' : [subject],
    'file_ids' : [f'gradCPT_run-{run}' for run in runs_with_events], 
    'subj_id_exclude' :[]
}

cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_thresh' : [1, 40]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_thresh' : [1e-5, 0.84]*units.V, # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : False,
    'flag_use_psp' : False,
    'channel_sel': None
}


cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, 
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : DO_TDDR,
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : False,
}

cfg_bandpass = { 
    'fmin' : F_MIN,
    'fmax' : F_MAX
}

cfg_preprocess = {
    'median_filt' : 1, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'cfg_GLM': cfg_GLM
}

# if block averaging on OD:
cfg_mse = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
     'mse_min_thresh' : 1e-6
    }

n_files_per_subject = len(cfg_dataset['file_ids'])

#%% RUN PREPROCESSING
print('LOADING PREPROCESSED DATA')
with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'cedalion', 'processed_data', subject, f'{subject}_preprocessed_results_{NOISE_MODEL}.pkl'), 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']

#%% RUN HRF ESTIMATION
# L = 20  # <-- set this appropriately
# W = windows.gaussian(L, std=L/6) / 2  

cfg_GLM['HRF_basis'] = 'cons_gaussians'
cfg_GLM['do_GSR']=False

wavelengths =  all_runs[0]['amp'].wavelength
dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": wavelengths},
                    )
                    
# REC_STR = 'od_o'
REC_STR = 'conc_o'
    
possible_trial_types = ['mnt-correct', 'mnt-incorrect', 'city-incorrect']    
    
trial_presence_list = []
stims_pruned_list = []

stim = all_stims[-1]
run = all_runs[-1]
mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
city_trials['trial_type'] = 'city-incorrect'
        
# Combine the filtered trials
stims_pruned = pd.concat([mnt_trials, city_trials], ignore_index=True)
run.stim = stims_pruned
stims_pruned_list.append(stims_pruned)

# reset the values for bad channels 
amp = all_runs[0]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
n_chs = len(amp.channel)
idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
idx_sat = np.where(all_chs_pruned[0] == 0.0)[0]
bad_indices = np.unique(np.concat([idx_amp, idx_sat]))
print(f'S10D127 is a bad channel: {np.where(run[REC_STR].channel.values=="S10D127") in bad_indices}')

#%% Training GLM
run_ts_list = [run[REC_STR]]
results, hrf_estimate, hrf_mse, dms = pf.GLM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)

# reset the values for bad channels 
amp = all_runs[0]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
n_chs = len(amp.channel)
idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
idx_sat = np.where(all_chs_pruned[0] == 0.0)[0]
bad_indices = np.unique(np.concat([idx_amp, idx_sat]))

if REC_STR=='conc_o':
    hrf_mse = hrf_mse.transpose('channel', 'time', 'chromo', 'trial_type')
    hrf_estimate = hrf_estimate.transpose('channel', 'time', 'chromo', 'trial_type')
else:
    hrf_mse = hrf_mse.transpose('channel', 'time', 'wavelength', 'trial_type')
    hrf_estimate = hrf_estimate.transpose('channel', 'time', 'wavelength', 'trial_type')

hrf_estimate = hrf_estimate - hrf_estimate.sel(time=(hrf_estimate.time < 0)).mean('time')

present_trial_types = hrf_estimate.trial_type.values.tolist()
presence = [tt in present_trial_types for tt in possible_trial_types] # creates bool mask
trial_presence_list.append(presence)

hrf_estimate = hrf_estimate.reindex({'trial_type': possible_trial_types})
hrf_mse = hrf_mse.reindex({'trial_type': possible_trial_types})

hrf_per_subj = hrf_estimate.expand_dims('subj')
hrf_per_subj = hrf_per_subj.assign_coords(subj=subject)

hrf_mse_per_subj = hrf_mse.expand_dims('subj')
hrf_mse_per_subj = hrf_mse_per_subj.assign_coords(subj=subject)

trial_presence_mask = xr.DataArray(trial_presence_list, 
                                dims = ['subj', 'trial_type'],
                                coords = {'subj': subject,
                                        'trial_type': possible_trial_types}
                                        )

print('HRF estimation complete')


all_results = {
            'hrf_per_subj': hrf_per_subj,  # always unweighted   - load into img recon
            'hrf_mse_per_subj': hrf_mse_per_subj, # - load into img recon
            'bad_indices': bad_indices,
        }


print('Job Complete.')

if REC_STR=='conc_o':
    ts = run[REC_STR].sel(chromo=['HbO'],channel=['S10D127'])
    betas = results.sm.params
    y_hat = (dms.common * betas).sum('regressor')
    y_hat_stim_cedalion = y_hat.transpose('chromo', 'channel', 'time')
    y_hat_stim_cedalion = y_hat_stim_cedalion.sel(chromo=['HbO'])
else:
    ts = run[REC_STR].sel(wavelength=[760],channel=['S10D127'])
    betas = results.sm.params
    y_hat = (dms.common * betas).sum('regressor')
    y_hat_stim_cedalion = y_hat.transpose('wavelength', 'channel', 'time')
    y_hat_stim_cedalion = y_hat_stim_cedalion.sel(wavelength=[760])


fig, axs = plt.subplots(1, 1, figsize=(14, 4), sharex=True)

axs.plot(ts.time.values, ts.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs.plot(dms.common.time.values, y_hat_stim_cedalion.values.flatten(),
        label='Stim only (No HPF)', alpha=0.7)
axs.set_ylabel('HbO concentration')
axs.set_title(f'Cedalion (AR-IRLS)')
axs.legend()
axs.grid()

#%% visualize my GLM
# open up pf.GLM
Y_all, stim_df, runs_updated = pf.concatenate_runs(run_ts_list, stims_pruned_list)
# get my DMs
stim_dm = model.get_GLM_copy_from_pf_DM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
# compare if my DMs is different from Cedalion DMs
print(f'If DMs are different: {np.any(stim_dm.common.values -dms.common.values)}')

#%% ===================
# Check cedalion.math.ar_irls.py ar_irls_GLM
# ===================
if REC_STR=='conc_o':
    ts = Y_all.sel(chromo=['HbO'],channel=['S10D127'])
    stim_dm.common = stim_dm.common.sel(chromo=['HbO'])
else:
    ts = Y_all.sel(wavelength=[760],channel=['S10D127'])
    stim_dm.common = stim_dm.common.sel(wavelength=[760])
design_matrix = stim_dm
verbose = False

ts = ts.pint.dequantify()
dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")
reg_results = xr.DataArray(
np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
dims=("channel", dim3_name),
coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
)

for (
dim3,
group_channels,
group_design_matrix,
) in design_matrix.iter_computational_groups(ts):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
                "time", "channel"
        )
        # pass x as a DataFrame to statsmodel to make it aware of regressor names
        x = pd.DataFrame(
                group_design_matrix.values, columns=group_design_matrix.regressor.values
        )
        for chan in group_y.channel.values:
                result = cedalion.math.ar_irls.ar_irls_GLM(group_y.loc[:, chan], x, pmax=30)
                reg_results.loc[chan, dim3] = result

description = 'Cedalion'
reg_results.attrs["description"] = description

betas = reg_results.sm.params
y_hat = (design_matrix.common * betas).sum('regressor')
if REC_STR=='conc_o':
    y_hat_stim_cedalion = y_hat.transpose('chromo', 'channel', 'time')
else:
    y_hat_stim_cedalion = y_hat.transpose('wavelength', 'channel', 'time')
# ===================
# Check my ar_irls_GLM
# ===================
if REC_STR=='conc_o':
    ts = Y_all.sel(chromo=['HbO'],channel=['S10D127'])
    stim_dm.common = stim_dm.common.sel(chromo=['HbO'])
else:
    ts = Y_all.sel(wavelength=[760],channel=['S10D127'])
    stim_dm.common = stim_dm.common.sel(wavelength=[760])
design_matrix = stim_dm
verbose = False

ts = ts.pint.dequantify()
dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")
reg_results = xr.DataArray(
np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
dims=("channel", dim3_name),
coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
)

for (
dim3,
group_channels,
group_design_matrix,
) in design_matrix.iter_computational_groups(ts):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
                "time", "channel"
        )
        # pass x as a DataFrame to statsmodel to make it aware of regressor names
        x = pd.DataFrame(
                group_design_matrix.values, columns=group_design_matrix.regressor.values
        )
        for chan in group_y.channel.values:
                result,_ = model.my_ar_irls_GLM(group_y.loc[:, chan], x, pmax=30)
                reg_results.loc[chan, dim3] = result

description = 'My AR-IRLS'
reg_results.attrs["description"] = description

betas = reg_results.sm.params
y_hat = (design_matrix.common * betas).sum('regressor')
if REC_STR=='conc_o':
    y_hat_stim_mine = y_hat.transpose('chromo', 'channel', 'time')
else:
    y_hat_stim_mine = y_hat.transpose('wavelength', 'channel', 'time')
fig, axs = plt.subplots(1, 1, figsize=(14, 4), sharex=True)

axs.plot(ts.time.values, ts.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs.plot(ts.time.values, y_hat_stim_mine.values.flatten(),
        label='Mine', alpha=0.7, marker='x')
axs.plot(ts.time.values, y_hat_stim_cedalion.values.flatten(),
        label='Cedalion', alpha=0.7)
axs.set_ylabel(f'{REC_STR} concentration')
axs.set_title(f'HbO estimation with Drift (AR-IRLS)')
axs.legend()
axs.grid()

print(f"If yhat is different: {np.any(y_hat_stim_cedalion.values-y_hat_stim_mine.values)}")