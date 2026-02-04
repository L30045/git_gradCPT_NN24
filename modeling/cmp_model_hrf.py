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
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from utils import *
import model
from params_setting import *
import cedalion.xrutils as xrutils
import scipy
import cedalion.typing as cdt
from cedalion.models.glm.design_matrix import DesignMatrix
from joblib import Parallel, delayed, parallel_config
import processing_func as pf


#%% get DM
subj_id = 695
print(f"Start processing sub-{subj_id}")
# load HbO
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)
all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']
cfg_GLM['geo3d'] = geo3d

#%% get epoched concentration
run_dict = dict()
# Find all event files in project_path
event_files = glob.glob(os.path.join(project_path, f"sub-{subj_id}", 'nirs', f"sub-{subj_id}_task-gradCPT_run-*_events.tsv"))
event_files = sorted(event_files)  # Sort to ensure consistent ordering

# Load each event file into run_dict
for event_file in event_files:
    # Extract run number from filename (e.g., run-01 -> 1)
    run_num = event_file.split('run-')[1].split('_')[0]
    run_key = f'run{run_num}'

    # Initialize run dict if not exists
    if run_key not in run_dict:
        run_dict[run_key] = dict()

    # Load event dataframe
    run_dict[run_key]['ev_df'] = pd.read_csv(event_file, sep='\t')

# find corresponding runs in all_runs and assign to run_dict
for r_i, run in enumerate(all_runs):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        ev_df = run_dict[run_key]['ev_df']
        if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
            run_dict[run_key]['run'] = run[0]
            run_dict[run_key]['conc_ts'] = run['conc_o']
            run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
            break

# epoch HbO
len_epoch = 12 # seconds
t_conc_ts = run['conc_o'].time
sfreq_conc = 1/np.diff(t_conc_ts)[0]
len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

#%% get epoched EEG
# load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)


# get mnt_correct trials 
mnt_correct_idx_dict = model.get_valid_event_idx('mnt_correct',single_subj_epoch_dict)
mnt_correct_area_dict = model.get_ERP_area('mnt_correct', single_subj_epoch_dict)

# get mnt_incorrect trials
mnt_incorrect_idx_dict = model.get_valid_event_idx('mnt_incorrect_response',single_subj_epoch_dict)
mnt_incorrect_area_dict = model.get_ERP_area('mnt_incorrect_response', single_subj_epoch_dict)

# combine mnt_correct_idx_dict, mnt_correct_area_dict, mnt_incorrect_idx_dict, mnt_incorrect_area_dict into a dict
ev_dict = dict()
for run_key in mnt_correct_idx_dict.keys():
    ev_dict[run_key] = {
        'mnt_correct': {
            'idx': mnt_correct_idx_dict[run_key],
            'area': mnt_correct_area_dict[run_key]
        },
        'mnt_incorrect': {
            'idx': mnt_incorrect_idx_dict[run_key],
            'area': mnt_incorrect_area_dict[run_key]
        }
    }

#%% Get reduced model DM
run_list = []
pruned_chans_list = []
stim_list = []
for run_key in run_dict.keys():
    run_list.append(run_dict[run_key]['run'])
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
reduced_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
model.vis_dm(reduced_dm)

#%% Get stim DM from Laura's code directly (Do this after sorting runs since this code changes run.stim)
REC_STR = 'conc_o'
stims_pruned_list = []
for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    # city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
    # city_trials['trial_type'] = 'city-incorrect'

    # Combine the filtered trials
    # stims_pruned = pd.concat([mnt_trials, city_trials], ignore_index=True)
    stims_pruned = mnt_trials
    run.stim = stims_pruned
    stims_pruned_list.append(stims_pruned)

run_ts_list = [run[REC_STR] for run in all_runs]
reduced_dm_laura = model.get_GLM_copy_from_pf_DM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
model.vis_dm(reduced_dm_laura)

#%% Compare Stim-only DM to make sure they are the same
drift_laura = reduced_dm_laura.common.sel(regressor=reduced_dm_laura.common.regressor.str.startswith(f"Drift")).values
drift_mine = reduced_dm.common.sel(regressor=reduced_dm_laura.common.regressor.str.startswith(f"Drift")).values
drift_diff = drift_laura-drift_mine
short_laura = reduced_dm_laura.common.sel(regressor=reduced_dm_laura.common.regressor.str.startswith(f"short")).values
short_mine = reduced_dm.common.sel(regressor=reduced_dm_laura.common.regressor.str.startswith(f"short")).values
short_diff = short_laura-short_mine
# different from first regressors HbO
plt.figure()
plt.plot(drift_diff[:,0,0],linewidth=5,label='Drift Diff')
plt.plot(short_diff[:,0,0],linewidth=5,label='Short Diff')
plt.axvline(run_list[0].shape[-1], color='r',linestyle='--', label='Mine Run Boundary')
plt.axvline(len(drift_diff)-run_list[-1].shape[-1], color='r',linestyle='--')
plt.axvline(run_ts_list[0].shape[-1], color='g', label='Laura''s Run Boundary',linestyle='--')
plt.axvline(len(drift_diff)-run_ts_list[-1].shape[-1], color='g',linestyle='--')
plt.legend()

#%% Train by pf.GLM
results_laura, hrf_estimate, hrf_mse = pf.GLM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)

#%% Get Basis DM, EEG DM, and Full DM
Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)
eeg_dm_dict = model.create_eeg_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])
Y_all, dm_all, runs_updated = model.concatenate_runs_dms(run_dict, eeg_dm_dict)
dm_all = model.combine_dm(dm_all, reduced_dm)
basis_dm = model.create_no_info_dm(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
print("Done loading")

#%% start my fitting



#%% check saved models
filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
# load full model
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
    full_model_result = pickle.load(f)
resid=full_model_result['resid']
betas_full=full_model_result['betas']

# load stim only results
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_stim-only_correct_trial_type.pkl"), 'rb') as f:
    reduced_model_result = pickle.load(f)
betas_reduced=reduced_model_result['betas']

# load drift results
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_drift_ss.pkl"), 'rb') as f:
    basis_model_result = pickle.load(f)
betas_basis=basis_model_result['betas']

#%% load laura's model
with open("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-695/sub-695_conc_o_glm_residual_ar_irls.pkl", 'rb') as f:
    resid_laura = pickle.load(f)

        

#%% visualize
x_full = np.squeeze(dm_all.common.values[:,:,0])
x_basis = np.squeeze(basis_dm.common.values[:,:,0])
x_reduced = np.squeeze(reduced_dm.common.values[:,:,0])
y = np.squeeze(Y_all.values[0,0,:])
beta_full=np.squeeze(betas_full.values[0,0,:])
beta_reduced=np.squeeze(betas_reduced.values[0,0,:])
beta_basis=np.squeeze(betas_basis.values[0,0,:])
fit_full = x_full@beta_full
fit_reduced = x_reduced@beta_reduced
fit_basis = x_basis@beta_basis
resid_full = y-fit_full
resid_reduced = y-fit_reduced
resid_basis = y-fit_basis

fig, ax = plt.subplots(2,1)
ax[0].plot(fit_full,label='Xorg @ beta (full)')
ax[0].plot(fit_reduced,label='Xorg @ beta (reduced)')
ax[0].plot(fit_basis,label='Xorg @ beta (reduced)')
ax[0].plot(y, label='Y')
ax[0].legend()
ax[1].plot(resid_full,label='resid (full)')
ax[1].plot(resid_reduced,label='resid (reduced)')
ax[1].plot(resid_basis,label='resid (basis)')
ax[1].plot(np.squeeze(resid_laura.values[0,0,:]),label='resid (laura)')
ax[1].legend()

#%%
