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

# get epoched concentration
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
my_all_runs = copy.deepcopy(all_runs)
for r_i, run in enumerate(my_all_runs):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        ev_df = run_dict[run_key]['ev_df']
        if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
            run_dict[run_key]['run'] = run
            run_dict[run_key]['conc_o'] = run['conc_o']
            run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
            break

# epoch HbO
len_epoch = 12 # seconds
t_conc_ts = run['conc_o'].time
sfreq_conc = 1/np.diff(t_conc_ts)[0]
len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

#%% Get reduced model DM
run_list = []
pruned_chans_list = []
stim_list = []

for run_key in run_dict.keys():
    # rename trial_type
    ev_df = run_dict[run_key]['ev_df'].copy()
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim = ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')]
    stim_list.append(stim)
    run = run_dict[run_key]['run']
    run.stim = stim
    run_list.append(run['conc_o'])
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    
reduced_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# model.vis_dm(reduced_dm)

# Get stim DM from Laura's code directly (Do this after sorting runs since this code changes run.stim)
REC_STR = 'conc_o'
stims_pruned_list = []
for stim, run in zip(all_stims, all_runs):
    ev_df = stim.copy()
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    mnt_trials = ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')]
    
    # Combine the filtered trials
    # stims_pruned = pd.concat([mnt_trials, city_trials], ignore_index=True)
    stims_pruned = mnt_trials
    # run.stim = stims_pruned
    stims_pruned_list.append(stims_pruned)

run_ts_list = [run[REC_STR] for run in all_runs]
#=============#
# Swap the order to RUN 1, RUN 2, RUN 3
new_order_run_ts_list = [run_ts_list[2], run_ts_list[1], run_ts_list[0]]
new_order_chs_pruned = [all_chs_pruned[2], all_chs_pruned[1], all_chs_pruned[0]]
new_order_stims_pruned_list = [stims_pruned_list[2], stims_pruned_list[1], stims_pruned_list[0]]
reduced_dm_laura = model.get_GLM_copy_from_pf_DM(new_order_run_ts_list, cfg_GLM, geo3d, new_order_chs_pruned, new_order_stims_pruned_list)
#=============#
# reduced_dm_laura = model.get_GLM_copy_from_pf_DM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
# model.vis_dm(reduced_dm_laura)

# %%
new_order_Y_all, new_order_stim_df, runs_updated = model.concatenate_runs(new_order_run_ts_list, new_order_stims_pruned_list)
Y_all, stim_df, runs_updated = model.concatenate_runs(run_ts_list, stims_pruned_list)
fig, ax = plt.subplots(2,1)
ax[0].plot(new_order_stim_df['onset'])
ax[0].plot(stim_df['onset'])


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
# results_laura, hrf_estimate, hrf_mse = pf.GLM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
results_laura, hrf_estimate, hrf_mse = pf.GLM(new_order_run_ts_list, cfg_GLM, geo3d, new_order_chs_pruned, new_order_stims_pruned_list)

#%% Use mine run and fit pf.GLM
results_mine, _, _ = pf.GLM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

#%%
reduced_dm_laura = model.get_GLM_copy_from_pf_DM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
x_reduced_laura = np.squeeze(reduced_dm_laura.common.values[:,:,0])
beta_laura=np.squeeze(results_laura.sm.params.values[0,0,:])
fit_laura = x_reduced_laura@beta_laura
Y_all_laura, _, runs_updated = model.concatenate_runs(new_order_run_ts_list, stims_pruned_list)
# reduced_dm_mine = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# x_reduced_mine = np.squeeze(reduced_dm_mine.common.values[:,:,0])
# beta_mine=np.squeeze(results_mine.sm.params.values[0,0,:])
# fit_mine = x_reduced_mine@beta_mine
# Y_all_mine, _, runs_updated = model.concatenate_runs(run_list, stim_list)

select_ch = 'S10D127'
plt.figure()
plt.plot(Y_all_laura.sel(chromo='HbO',channel=select_ch),label='Y_all (Laura)')
plt.plot(fit_laura,label='fit (Laura)')
# plt.plot(Y_all_mine.sel(chromo='HbO',channel=select_ch),label='Y_all (Mine)')
# plt.plot(fit_mine,label='fit (Mine)')
plt.legend()
plt.show()

#%% fitting run one-by-one
pick_run_i = 0
run_ts_list = [run[REC_STR].sel(chromo=['HbO']) for run in all_runs]
modified_y = run_ts_list[pick_run_i]
modified_y.values[0,:,0] += 1e-4
results_single_run, hrf_estimate, hrf_mse = pf.GLM([modified_y], cfg_GLM, geo3d, [all_chs_pruned[pick_run_i]], [stims_pruned_list[pick_run_i]])

reduced_dm_run = model.get_GLM_copy_from_pf_DM([run_ts_list[pick_run_i]], cfg_GLM, geo3d, [all_chs_pruned[pick_run_i]], [stims_pruned_list[pick_run_i]])
x_reduced_run = np.squeeze(reduced_dm_run.common.values[:,:,0])
beta_run=np.squeeze(results_single_run.sm.params.values[0,0,:])
fit_run = x_reduced_run@beta_run
Y_all_run, _, _ = model.concatenate_runs([modified_y],[stims_pruned_list[pick_run_i]])

#%% Testing my fit
reg_results, autoReg_dict = model.my_fit(Y_all_run, reduced_dm_run) 
beta_mine=np.squeeze(reg_results.sm.params.values[0,0,:])
fit_mine = x_reduced_run@beta_mine

#%%
select_ch = 'S10D127'
# plt_fit = fit_run-Y_all_run.sel(chromo='HbO',channel=select_ch)[0].values
plt_fit = fit_run
plt.figure()
plt.plot(Y_all_run.sel(chromo='HbO',channel=select_ch),label='Y_all')
plt.plot(plt_fit,label=f'fit (f0={plt_fit[0]})')
plt.plot(fit_mine,label=f'fit (Mine) (f0={plt_fit[0]})')
plt.legend()
plt.show()


#%% Compare my HRF with Laura's
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)
all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d_695 = results['geo3d']

# pick_ch = 'S6D42'
fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
    results = pickle.load(f)
    hrf_per_subj = results['hrf_per_subj']
    hrf_mse_per_subj = results['hrf_mse_per_subj']
    bad_indices = results['bad_indices']

filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
# load full model
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
    full_model_result = pickle.load(f)
# load reduced model
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
    reduced_model_result = pickle.load(f)
# load dev branch laura code
with open(os.path.join(filepath,f'sub-{subj_id}_hrf_dev_Laura_code.pkl'),'rb') as f:
    result_dict = pickle.load(f)
    hrf_dev_Laura_code = result_dict['hrf_per_subj']
    coords_fixed = {k: v for k, v in hrf_dev_Laura_code._coords.items() if k != 'subj'}
    hrf_per_subj_fixed_dev = model.xr.DataArray(hrf_dev_Laura_code.variable, coords=coords_fixed)

with open(os.path.join(filepath,f'sub-{subj_id}_dev_reduced.pkl'),'rb') as f:
    reduced_model_result_dev = pickle.load(f)
    
#%% compare at 10 random channels
clean_chs = hrf_per_subj.channel.values.copy()
clean_chs = np.delete(clean_chs,bad_indices)
pick_chs = np.random.choice(clean_chs,size=10)
coords_fixed = {k: v for k, v in hrf_per_subj._coords.items() if k != 'subj'}
hrf_per_subj_fixed = model.xr.DataArray(hrf_per_subj.variable, coords=coords_fixed)


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, pick_ch in enumerate(pick_chs):
    hrf_laura = hrf_per_subj_fixed.sel(chromo='HbO',trial_type='mnt-correct',channel=pick_ch).values.reshape(-1)
    hrf_dev_laura = hrf_per_subj_fixed_dev.sel(chromo='HbO',trial_type='mnt-correct',channel=pick_ch).values.reshape(-1)
    hrf_estimate_full = full_model_result['hrf_estimate'].sel(chromo='HbO',trial_type='mnt-correct',channel=pick_ch).values
    hrf_estimate_reduced = reduced_model_result['hrf_estimate'].sel(chromo='HbO',trial_type='mnt-correct',channel=pick_ch).values
    hrf_estimate_reduced_dev = reduced_model_result_dev['hrf_estimate'].sel(chromo='HbO',trial_type='mnt-correct',channel=pick_ch).values
    axes[i].plot(hrf_laura,'k',label='Laura')
    axes[i].plot(hrf_dev_laura,'b',label='Dev (Laura)')
    # axes[i].plot(hrf_estimate_reduced_dev,'m-o',label='Dev (Mine)')
    axes[i].plot(hrf_estimate_full,'g',label='Full')
    axes[i].plot(hrf_estimate_reduced,'r',label='Reduced')
    axes[i].legend()
    axes[i].set_title(pick_ch)
plt.tight_layout()
plt.show()
    