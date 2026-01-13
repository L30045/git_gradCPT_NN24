#%% load library
import numpy as np
import pickle
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from utils import *
import model
from model import extract_n2_p3_features
from params_setting import *

#%% load HbO
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']

#%% get epoched concentration
run_id = 1
event_file = os.path.join(project_path, f"sub-{subj_id}", 'nirs',  f"sub-{subj_id}_task-gradCPT_run-0{run_id}_events.tsv")
event_df = pd.read_csv(event_file,sep='\t')
# find corresponding runs in all_runs
for r_i, run in enumerate(all_runs):
    if np.all(run.stim.iloc[0]==event_df.iloc[0]):
        target_run = run
        conc_ts = run['conc_o']
        chs_pruned = all_chs_pruned[r_i]
        break
# get mnt_correct event onset time
mnt_df = event_df[(event_df['trial_type']=='mnt')&(event_df["response_code"]==0)]
# epoch HbO
len_epoch = 12 # seconds
t_conc_ts = conc_ts.time
sfreq_conc = 1/np.diff(t_conc_ts)[0]
len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

#%% get epoched EEG
# load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)

#%% get event trials
def get_valid_event_idx(ev_name, single_subj_epoch_dict):
    run_keys = np.sort([x for x in single_subj_epoch_dict.keys()])
    ev_preserved_idx_list = []
    ev_rejected_idx_list = []
    for run_key in run_keys:
        # get preserved trial index
        ev_preserved_idx = np.where([len(log) == 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
        # get rejected trial index
        ev_rejected_idx = np.where([len(log) != 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
        ev_preserved_idx_list.append(ev_preserved_idx)
        ev_rejected_idx_list.append(ev_rejected_idx)
    return ev_preserved_idx, ev_rejected_idx, run_keys

#%% get ERP area
def get_ERP_area(ev_name, single_subj_epoch_dict, is_norm=True):
    # define output dict
    erp_area_dict = dict()
    # define ERP period of interest
    if ev_name.endswith('response'):
        n2_window=(0,0.2)
        p3_window=(0,0.2)
    else:
        n2_window=(0.4, 0.7)
        p3_window=(0.6, 1.1)
    # for each run
    for run_key in single_subj_epoch_dict.keys():
        erp_area_dict[run_key]=dict()
        t_vector = single_subj_epoch_dict[run_key][ev_name].times
        # for each channel, extract ERP area
        ev_eeg = single_subj_epoch_dict[run_key][ev_name].pick(picks='eeg')
        for ch_name in ev_eeg.ch_names:
            ev_ch_eeg = ev_eeg.get_data()[:,ev_eeg.ch_names.index(ch_name),:]            
            # Extract N2 and P3 features
            area_list = []
            for eeg_i in range(len(ev_ch_eeg)):
                n2_p3_features = extract_n2_p3_features(ev_ch_eeg[eeg_i], t_vector,
                                                        n2_window=n2_window,
                                                        p3_window=p3_window)
                n2_area = np.abs(n2_p3_features['n2_area'])
                p3_area = np.abs(n2_p3_features['p3_area'])
                if ev_name.endswith('respons'):
                    area_list.append(n2_area)
                else:
                    area_list.append(n2_area+p3_area)
            # rescale area to range 0 to 1. (0 as 0, 1 as max(area))
            area_list = np.array(area_list)
            if is_norm:
                area_list = area_list/np.max(area_list)
            # store results
            erp_area_dict[run_key][ch_name] = area_list
    return erp_area_dict

#%% get mnt_correct trials 
mnt_correct_preserved_idx, mnt_correct_rejected_idx, run_keys = get_valid_event_idx('mnt_correct',single_subj_epoch_dict)
mnt_correct_area_dict = get_ERP_area('mnt_correct', single_subj_epoch_dict)

#%% get mnt_incorrect trials
mnt_incorrect_preserved_idx, mnt_incorrect_rejected_idx, run_keys = get_valid_event_idx('mnt_incorrect_response',single_subj_epoch_dict)
mnt_incorrect_area_dict = get_ERP_area('mnt_incorrect_response', single_subj_epoch_dict)

#%%
# preserve mnt event with corresponding EEG in event_df
mnt_df = mnt_df.iloc[mnt_correct_preserved_idx]


#%% create design matrix for each event, scale HRF based on Cz variance
# for each event, create a design matrix and scale the gaussian kernels
run_unit = target_run[0].pint.units
dm_list = []
for ev_i in range(len(mnt_df)):
    # create design matrix for single event
    dm = glm.design_matrix.hrf_regressors(
                                        target_run[0],
                                        mnt_df.iloc[[ev_i]],
                                        glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                    )
    # rescale by Cz area
    dm.common = dm.common*area_list[ev_i]
    # append
    dm_list.append(dm)
# Create a new design matrix object with the concatenated common regressors
dms = dm_list.pop()
dms_common = dms.common
# merge all dms along time axis
while len(dm_list)>0:
    dms_common += dm_list.pop().common
# assign merged common back to dms
dms.common = dms_common

# Combine drift and short-separation regressors (if any)
if cfg_GLM['do_drift']:
    drift_regressors = model.get_drift_regressors([target_run['conc_o']], cfg_GLM)
    dms &= reduce(operator.and_, drift_regressors)

if cfg_GLM['do_drift_legendre']:
    drift_regressors = model.get_drift_legendre_regressors([target_run['conc_o']], cfg_GLM)
    dms &= reduce(operator.and_, drift_regressors)

if cfg_GLM['do_short_sep']:
    ss_regressors = model.get_short_regressors([target_run['conc_o']], [chs_pruned], geo3d, cfg_GLM)
    dms &= reduce(operator.and_, ss_regressors)

dms.common = dms.common.fillna(0)

#%% check dm
plt_dm = dms
# using xr.DataArray.plot
f, ax = plt.subplots(1,1,figsize=(12,10))
plt_dm.common.sel(chromo="HbO", time=plt_dm.common.time < 600).T.plot(vmin=-2,vmax=2)
plt.title("Shared Regressors")
#p.xticks(rotation=90)
plt.show()

#%% GLM fitting from shank Jun 02 2025
# 3. get betas and covariance
results = glm.fit(target_run[0], dms, noise_model=cfg_GLM['noise_model']) 
betas = results.sm.params
cov_params = results.sm.cov_params()

#%% 4. estimate HRF and MSE
basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(target_run[0])

trial_type_list = mnt_df['trial_type'].unique()

hrf_mse_list = []
hrf_estimate_list = []

for trial_type in trial_type_list:
    
    betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
    hrf_estimate = model.estimate_HRF_from_beta(betas_hrf, basis_hrf)
    
    cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                        regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                )
    hrf_mse = model.estimate_HRF_cov(cov_hrf, basis_hrf)

    hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
    hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

    hrf_estimate_list.append(hrf_estimate)
    hrf_mse_list.append(hrf_mse)

hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
hrf_estimate = hrf_estimate.pint.quantify(run_unit)

hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
hrf_mse = hrf_mse.pint.quantify(run_unit**2)

# set universal time so that all hrfs have the same time base 
fs = frequency.sampling_rate(target_run[0]).to('Hz')
before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

dT = np.round(1 / fs, 3)  # millisecond precision
n_timepoints = len(hrf_estimate.time)
reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

hrf_mse = hrf_mse.assign_coords({'time': reltime})
hrf_mse.time.attrs['units'] = 'second'

hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
hrf_estimate.time.attrs['units'] = 'second'

#%% Visualizing MSE
summarized_method = lambda x, y: np.sum(x[:,:,:,y].values,axis=x.dims.index('time')).reshape(-1)
mse_check = 'HbO'
if mse_check == 'HbT':
    vis_mse = summarized_method(hrf_mse,0)+summarized_method(hrf_mse,1)
else:
    vis_mse = summarized_method(hrf_mse,0 if mse_check=='HbO' else 1)
f, ax = plt.subplots(1, 2, figsize=(15, 8))
scalp_plot(
conc_ts,
geo3d,
vis_mse,
ax[1],
cmap='jet',
# vmin=-5,
# vmax=0,
optode_labels=False,
title="MSE",
optode_size=6,
)

#%% add other EEG channels

#%% Earse DM during missing events


#%% Add other events to DM
# for mnt_incorrect trials, look for N1 (0-200ms after response)
# ignore city incorrect for now



#%% load Laura's results
laura_results_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
with gzip.open(os.path.join(laura_results_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
    hrf_laura = pickle.load(f)
