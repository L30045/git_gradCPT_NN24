
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
import matplotlib.pyplot as plt

import cedalion
from cedalion import units, nirs
from cedalion.sigproc import frequency
import cedalion.models.glm as glm
from scipy.signal import filtfilt, windows

# import my own functions from a different directory
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
# sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules')
import processing_func as pf
# import image_recon_func as irf

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

#%%
# subject = str(sys.argv[1])
subject = 'sub-723' 
select_parcel='DorsAttnA_ParOcc_1_RH'
select_chromo='HbO'

# Initial root directory and analysis parameters
SPLIT_VTC = False
SAVE_RESIDUAL = False
USE_GSR = False
NOISE_MODEL = 'ar_irls'
root_dir = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
ADOT_FLAG = 'probe'
spatial_dim = 'vertex'
flag = ''
hrf_basis = 'cons_gaussians'

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
with gzip.open( os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    results = pickle.load(f)

# all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']

print('LOADING IMAGE SPACE RESULTS')
folder =  os.path.join(der_dir, subject)
filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'

with open(filepath, 'rb') as f:
    image_results = pickle.load(f)

all_runs = image_results['parcel_ts']
vv = image_results['vertex_mse']

n_runs = len(vv)
vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

# RUN HRF ESTIMATION
L = 20  # <-- set this appropriately
W = windows.gaussian(L, std=L/6) / 2  
                    
if SPLIT_VTC:
    possible_trial_types = ['mnt-correct-in', 'mnt-correct-out', 'mnt-incorrect', 'city-incorrect']
else:
    possible_trial_types = ['mnt-correct', 'mnt-incorrect']    

trial_presence_list = []
stims_pruned_list = []

all_runs_tmp = []
for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    # city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
    # city_trials['trial_type'] = 'city-incorrect'
    
    if SPLIT_VTC:
        VTC = stim['VTC'].to_numpy()
        VTC = filtfilt(W, sum(W), VTC)
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
    # stims_pruned = pd.concat(, ignore_index=True)
    # run.stim = stims_pruned
    if F_MIN > 0: 
        # TODO HP the timeseries data 
        run.time.attrs['units'] = units.s
        run_filt = frequency.freq_filter(run, 
                                    F_MIN*units.Hz, 
                                    F_MAX*units.Hz)
        all_runs_tmp.append(run_filt)
    else:
        all_runs_tmp.append(run)

    stims_pruned_list.append(mnt_trials)
    

# run_ts_list = [image_results['parcel_ts_weights']]
all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs_tmp]

all_runs_tmp = []
for run in all_runs:
    run.time.attrs['units'] = units.s
    run = run.sel(parcel = run.parcel != 'scalp')
    all_runs_tmp.append(run)

all_runs = all_runs_tmp.copy()

# select only one parcel and one chromo
all_runs = [x.sel(parcel=[select_parcel], chromo=[select_chromo]) for x in all_runs]

#
results, hrf_estimate, hrf_mse, dms = pf.GLM(all_runs, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
Y_all, stim_df, runs_updated = pf.concatenate_runs(all_runs, stims_pruned_list)

# visualize results
betas = results.sm.params
# y_hat = xr.dot(dms.common, betas, dims='regressor')
y_hat = (dms.common * betas).sum('regressor')
y_hat_laura = y_hat.transpose('chromo', 'parcel', 'time')

#
fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

y_true = Y_all.values.flatten()
y_hat_vals = y_hat_laura.values.flatten()


axs[0].plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=2)
axs[0].plot(y_hat_laura.time.values, y_hat_vals,'b',
        label='Laura', alpha=0.5)
axs[0].set_ylabel(f'HbO concentration')
axs[0].set_title(f'Parcel activities estimation ({select_parcel})')
axs[0].legend()
axs[0].grid()
axs[1].plot(Y_all.time.values, y_true-y_hat_vals, label='Resid')
axs[1].legend()
axs[1].grid()
