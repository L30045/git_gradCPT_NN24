#%% load library
import numpy as np
import pickle
import copy
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
import statsmodels.api as sm
import scipy
import cedalion
import statsmodels.api as sm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.xrutils as xrutils
import copy

#%%
subj_id = 723
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")
hpf_freq = 0.02* units.Hz

# load HbO
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']
cfg_GLM['geo3d'] = geo3d

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

run_list = []
run_list_HPF = []
pruned_chans_list = []
stim_list = []
for run_key in run_dict.keys():
    local_run = run_dict[run_key]['run']
    run_list.append(local_run)
    # high pass filter    
    hpf_run = cedalion.sigproc.frequency.freq_filter(
        local_run, fmin=hpf_freq, fmax=0 * units.Hz, butter_order=4
    )
    run_list_HPF.append(hpf_run)
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)
Y_HPF, _, runs_updated = model.concatenate_runs(run_list_HPF, stim_list)
Y_test = Y_all.sel(chromo=['HbO'],channel=[debug_channel])
Y_test_HPF = Y_HPF.sel(chromo=['HbO'],channel=[debug_channel])

#%% ===================
# get drift only results
# ===================
noSS_cfg_GLM = copy.deepcopy(cfg_GLM)
noSS_cfg_GLM['do_short_sep']=False
drift_dm = model.create_no_info_dm(run_list, noSS_cfg_GLM, noSS_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
drift_dm.common = drift_dm.common.sel(chromo=['HbO'])
drift_results, autoReg_dict = model.my_fit(Y_test, drift_dm)

betas = drift_results.sm.params
y_hat = (drift_dm.common * betas).sum('regressor')
y_hat_drift = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get ss only results
# ===================
noDrift_cfg_GLM = copy.deepcopy(cfg_GLM)
noDrift_cfg_GLM['do_drift']=False
noDrift_cfg_GLM['do_drift_legendre']=False
ss_dm = model.create_no_info_dm(run_list, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
ss_dm.common = ss_dm.common.sel(chromo=['HbO'])
ss_results, autoReg_dict = model.my_fit(Y_test, ss_dm)

betas = ss_results.sm.params
y_hat = (ss_dm.common * betas).sum('regressor')
y_hat_ss = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get ss only results (HPF)
# ===================
ss_dm = model.create_no_info_dm(run_list_HPF, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
ss_dm.common = ss_dm.common.sel(chromo=['HbO'])
ss_results, autoReg_dict = model.my_fit(Y_test_HPF, ss_dm)

betas = ss_results.sm.params
y_hat = (ss_dm.common * betas).sum('regressor')
y_hat_ss_hpf = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim with no drift results
# ===================
stim_dm = model.get_GLM_copy_from_pf_DM(run_list, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm.common = stim_dm.common.sel(chromo=['HbO'])
stim_results, autoReg_dict = model.my_fit(Y_test, stim_dm)

betas = stim_results.sm.params
y_hat = (stim_dm.common * betas).sum('regressor')
y_hat_stim_noHPF_noDrift = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim only results (HPF)
# ===================
stim_dm_hpf = model.get_GLM_copy_from_pf_DM(run_list_HPF, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_hpf.common = stim_dm_hpf.common.sel(chromo=['HbO'])
stim_results_hpf, autoReg_dict = model.my_fit(Y_test_HPF, stim_dm_hpf)

betas = stim_results_hpf.sm.params
y_hat = (stim_dm_hpf.common * betas).sum('regressor')
y_hat_stim_HPF_noDrift = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get drift+ss results (HPF)
# ===================
basis_dm_hpf = model.create_no_info_dm(run_list_HPF, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
basis_dm_hpf.common = basis_dm_hpf.common.sel(chromo=['HbO'])
basis_results_hpf, autoReg_dict = model.my_fit(Y_test_HPF, basis_dm_hpf)

betas = basis_results_hpf.sm.params
y_hat = (basis_dm_hpf.common * betas).sum('regressor')
y_hat_basis_hpf = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim results (HPF)
# ===================
stim_dm_hpf = model.get_GLM_copy_from_pf_DM(run_list_HPF, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_hpf.common = stim_dm_hpf.common.sel(chromo=['HbO'])
stim_results_hpf, autoReg_dict = model.my_fit(Y_test_HPF, stim_dm_hpf)

betas = stim_results_hpf.sm.params
y_hat = (stim_dm_hpf.common * betas).sum('regressor')
y_hat_stim_HPF = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get drift+ss results
# ===================
basis_dm = model.create_no_info_dm(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
basis_dm.common = basis_dm.common.sel(chromo=['HbO'])
basis_results, autoReg_dict = model.my_fit(Y_test, basis_dm)

betas = basis_results.sm.params
y_hat = (basis_dm.common * betas).sum('regressor')
y_hat_basis = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim results
# ===================
stim_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm.common = stim_dm.common.sel(chromo=['HbO'])
stim_results, autoReg_dict = model.my_fit(Y_test, stim_dm)

betas = stim_results.sm.params
y_hat = (stim_dm.common * betas).sum('regressor')
y_hat_stim = y_hat.transpose('chromo', 'channel', 'time')


#%% subplot comparing HRF results: no-HPF (top) vs HPF (bottom)
fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

axs[0].plot(Y_test.time.values, Y_test.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[0].plot(Y_test.time.values, y_hat_ss.values.flatten(),
        label='Basis (No HPF, No Drift)', alpha=0.7)
axs[0].plot(Y_test.time.values, y_hat_stim_noHPF_noDrift.values.flatten(),
        label='Stim only (No HPF, No Drift)', alpha=0.7)
axs[0].set_ylabel(f'{debug_chromo} concentration')
axs[0].set_title('No HPF, No Drift')
axs[0].legend()
axs[0].grid()

axs[1].plot(Y_test.time.values, Y_test.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[1].plot(Y_test.time.values, y_hat_basis.values.flatten(),
        label='Basis', alpha=0.7)
axs[1].plot(Y_test.time.values, y_hat_stim.values.flatten(),
        label='Stim only', alpha=0.7)
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel(f'{debug_chromo} concentration')
axs[1].set_title('No HPF, Drift')
axs[1].legend()
axs[1].grid()

axs[2].plot(Y_test_HPF.time.values, Y_test_HPF.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[2].plot(Y_test_HPF.time.values, y_hat_ss_hpf.values.flatten(),
        label='Basis (HPF, No Drift)', alpha=0.7)
axs[2].plot(Y_test_HPF.time.values, y_hat_stim_HPF_noDrift.values.flatten(),
        label='Stim only (HPF, No Drift)', alpha=0.7)
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel(f'{debug_chromo} concentration')
axs[2].set_title('HPF (0.02 Hz), No Drift')
axs[2].legend()
axs[2].grid()

axs[3].plot(Y_test_HPF.time.values, Y_test_HPF.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[3].plot(Y_test_HPF.time.values, y_hat_basis_hpf.values.flatten(),
        label='Basis (HPF)', alpha=0.7)
axs[3].plot(Y_test_HPF.time.values, y_hat_stim_HPF.values.flatten(),
        label='Stim only (HPF)', alpha=0.7)
axs[3].set_xlabel('time (s)')
axs[3].set_ylabel(f'{debug_chromo} concentration')
axs[3].set_title('HPF (0.02 Hz), Drift')
axs[3].legend()
axs[3].grid()


fig.suptitle(f'sub-{subj_id} channel={debug_channel}')
plt.tight_layout()
plt.show()

#%% ===================
# get drift only results (run by run)
# ===================
noSS_cfg_GLM = copy.deepcopy(cfg_GLM)
noSS_cfg_GLM['do_short_sep']=False
y_true_list = []
y_hat_drift_list = []

for run,chans,stim in zip(run_list,pruned_chans_list,stim_list):
    y_true = run.sel(chromo=['HbO'],channel=[debug_channel])
    y_true_list.append(y_true)
    drift_dm = model.create_no_info_dm([run], noSS_cfg_GLM, noSS_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
    # select chromo=HbO only to save time
    drift_dm.common = drift_dm.common.sel(chromo=['HbO'])
    drift_results, autoReg_dict = model.my_fit(y_true, drift_dm)

    betas = drift_results.sm.params
    y_hat = (drift_dm.common * betas).sum('regressor')
    y_hat_drift = y_hat.transpose('chromo', 'channel', 'time')
    y_hat_drift_list.append(y_hat_drift)

fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax_i in range(3):
    axs[ax_i].plot(y_true_list[ax_i].time.values, y_true_list[ax_i].values.flatten(), label='Y (true)', color='k', linewidth=1)
    axs[ax_i].plot(y_true_list[ax_i].time.values, y_hat_drift_list[ax_i].values.flatten(),
            label='Drift only (No HPF)', alpha=0.7)
    axs[ax_i].set_ylabel(f'{debug_chromo} concentration')
    axs[ax_i].set_title(f'run {ax_i}')
    axs[ax_i].legend()
    axs[ax_i].grid()

#%% ===================================================================
# Rerun the whole analysis using OLS instead of AR-IRLS
# Y_all and design matrices are identical to the AR-IRLS run above.
# ===================================================================

#%% ===================
# get drift only results (OLS)
# ===================
noSS_cfg_GLM = copy.deepcopy(cfg_GLM)
noSS_cfg_GLM['do_short_sep']=False
drift_dm_ols = model.create_no_info_dm(run_list, noSS_cfg_GLM, noSS_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
drift_dm_ols.common = drift_dm_ols.common.sel(chromo=['HbO'])
drift_results_ols, _ = model.my_ols_fit(Y_test, drift_dm_ols)

betas = drift_results_ols.sm.params
y_hat = (drift_dm_ols.common * betas).sum('regressor')
y_hat_drift_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get ss only results (OLS)
# ===================
noDrift_cfg_GLM = copy.deepcopy(cfg_GLM)
noDrift_cfg_GLM['do_drift']=False
noDrift_cfg_GLM['do_drift_legendre']=False
ss_dm_ols = model.create_no_info_dm(run_list, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
ss_dm_ols.common = ss_dm_ols.common.sel(chromo=['HbO'])
ss_results_ols, _ = model.my_ols_fit(Y_test, ss_dm_ols)

betas = ss_results_ols.sm.params
y_hat = (ss_dm_ols.common * betas).sum('regressor')
y_hat_ss_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get ss only results (HPF, OLS)
# ===================
ss_dm_hpf_ols = model.create_no_info_dm(run_list_HPF, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
ss_dm_hpf_ols.common = ss_dm_hpf_ols.common.sel(chromo=['HbO'])
ss_results_hpf_ols, _ = model.my_ols_fit(Y_test_HPF, ss_dm_hpf_ols)

betas = ss_results_hpf_ols.sm.params
y_hat = (ss_dm_hpf_ols.common * betas).sum('regressor')
y_hat_ss_hpf_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim with no drift results (OLS)
# ===================
stim_dm_noHPF_noDrift_ols = model.get_GLM_copy_from_pf_DM(run_list, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_noHPF_noDrift_ols.common = stim_dm_noHPF_noDrift_ols.common.sel(chromo=['HbO'])
stim_results_ols, _ = model.my_ols_fit(Y_test, stim_dm_noHPF_noDrift_ols)

betas = stim_results_ols.sm.params
y_hat = (stim_dm_noHPF_noDrift_ols.common * betas).sum('regressor')
y_hat_stim_noHPF_noDrift_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim only results (HPF, OLS)
# ===================
stim_dm_hpf_noDrift_ols = model.get_GLM_copy_from_pf_DM(run_list_HPF, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_hpf_noDrift_ols.common = stim_dm_hpf_noDrift_ols.common.sel(chromo=['HbO'])
stim_results_hpf_noDrift_ols, _ = model.my_ols_fit(Y_test_HPF, stim_dm_hpf_noDrift_ols)

betas = stim_results_hpf_noDrift_ols.sm.params
y_hat = (stim_dm_hpf_noDrift_ols.common * betas).sum('regressor')
y_hat_stim_HPF_noDrift_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get drift+ss results (HPF, OLS)
# ===================
basis_dm_hpf_ols = model.create_no_info_dm(run_list_HPF, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
basis_dm_hpf_ols.common = basis_dm_hpf_ols.common.sel(chromo=['HbO'])
basis_results_hpf_ols, _ = model.my_ols_fit(Y_test_HPF, basis_dm_hpf_ols)

betas = basis_results_hpf_ols.sm.params
y_hat = (basis_dm_hpf_ols.common * betas).sum('regressor')
y_hat_basis_hpf_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim results (HPF, OLS)
# ===================
stim_dm_hpf_ols = model.get_GLM_copy_from_pf_DM(run_list_HPF, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_hpf_ols.common = stim_dm_hpf_ols.common.sel(chromo=['HbO'])
stim_results_hpf_ols, _ = model.my_ols_fit(Y_test_HPF, stim_dm_hpf_ols)

betas = stim_results_hpf_ols.sm.params
y_hat = (stim_dm_hpf_ols.common * betas).sum('regressor')
y_hat_stim_HPF_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get drift+ss results (OLS)
# ===================
basis_dm_ols = model.create_no_info_dm(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
basis_dm_ols.common = basis_dm_ols.common.sel(chromo=['HbO'])
basis_results_ols, _ = model.my_ols_fit(Y_test, basis_dm_ols)

betas = basis_results_ols.sm.params
y_hat = (basis_dm_ols.common * betas).sum('regressor')
y_hat_basis_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% ===================
# get stim results (OLS)
# ===================
stim_dm_ols = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
# select chromo=HbO only to save time
stim_dm_ols.common = stim_dm_ols.common.sel(chromo=['HbO'])
stim_results_ols, _ = model.my_ols_fit(Y_test, stim_dm_ols)

betas = stim_results_ols.sm.params
y_hat = (stim_dm_ols.common * betas).sum('regressor')
y_hat_stim_ols = y_hat.transpose('chromo', 'channel', 'time')

#%% subplot comparing HRF results (OLS): no-HPF (top) vs HPF (bottom)
fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

axs[0].plot(Y_test.time.values, Y_test.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[0].plot(Y_test.time.values, y_hat_ss_ols.values.flatten(),
        label='Basis (No HPF, No Drift)', alpha=0.7)
axs[0].plot(Y_test.time.values, y_hat_stim_noHPF_noDrift_ols.values.flatten(),
        label='Stim only (No HPF, No Drift)', alpha=0.7)
axs[0].set_ylabel(f'{debug_chromo} concentration')
axs[0].set_title('No HPF, No Drift (OLS)')
axs[0].legend()
axs[0].grid()

axs[1].plot(Y_test.time.values, Y_test.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[1].plot(Y_test.time.values, y_hat_basis_ols.values.flatten(),
        label='Basis', alpha=0.7)
axs[1].plot(Y_test.time.values, y_hat_stim_ols.values.flatten(),
        label='Stim only', alpha=0.7)
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel(f'{debug_chromo} concentration')
axs[1].set_title('No HPF, Drift (OLS)')
axs[1].legend()
axs[1].grid()

axs[2].plot(Y_test_HPF.time.values, Y_test_HPF.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[2].plot(Y_test_HPF.time.values, y_hat_ss_hpf_ols.values.flatten(),
        label='Basis (HPF, No Drift)', alpha=0.7)
axs[2].plot(Y_test_HPF.time.values, y_hat_stim_HPF_noDrift_ols.values.flatten(),
        label='Stim only (HPF, No Drift)', alpha=0.7)
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel(f'{debug_chromo} concentration')
axs[2].set_title('HPF (0.02 Hz), No Drift (OLS)')
axs[2].legend()
axs[2].grid()

axs[3].plot(Y_test_HPF.time.values, Y_test_HPF.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs[3].plot(Y_test_HPF.time.values, y_hat_basis_hpf_ols.values.flatten(),
        label='Basis (HPF)', alpha=0.7)
axs[3].plot(Y_test_HPF.time.values, y_hat_stim_HPF_ols.values.flatten(),
        label='Stim only (HPF)', alpha=0.7)
axs[3].set_xlabel('time (s)')
axs[3].set_ylabel(f'{debug_chromo} concentration')
axs[3].set_title('HPF (0.02 Hz), Drift (OLS)')
axs[3].legend()
axs[3].grid()

fig.suptitle(f'sub-{subj_id} channel={debug_channel} (OLS)')
plt.tight_layout()
plt.show()

#%% ===================
# get drift only results (run by run, OLS)
# ===================
y_true_list_ols = []
y_hat_drift_list_ols = []

for run,chans,stim in zip(run_list,pruned_chans_list,stim_list):
    y_true = run.sel(chromo=['HbO'],channel=[debug_channel])
    y_true_list_ols.append(y_true)
    drift_dm_run_ols = model.create_no_info_dm([run], noSS_cfg_GLM, noSS_cfg_GLM['geo3d'], pruned_chans_list, stim_list)
    # select chromo=HbO only to save time
    drift_dm_run_ols.common = drift_dm_run_ols.common.sel(chromo=['HbO'])
    drift_results_run_ols, _ = model.my_ols_fit(y_true, drift_dm_run_ols)

    betas = drift_results_run_ols.sm.params
    y_hat = (drift_dm_run_ols.common * betas).sum('regressor')
    y_hat_drift_run_ols = y_hat.transpose('chromo', 'channel', 'time')
    y_hat_drift_list_ols.append(y_hat_drift_run_ols)

fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax_i in range(3):
    axs[ax_i].plot(y_true_list_ols[ax_i].time.values, y_true_list_ols[ax_i].values.flatten(), label='Y (true)', color='k', linewidth=1)
    axs[ax_i].plot(y_true_list_ols[ax_i].time.values, y_hat_drift_list_ols[ax_i].values.flatten(),
            label='Drift only (No HPF, OLS)', alpha=0.7)
    axs[ax_i].set_ylabel(f'{debug_chromo} concentration')
    axs[ax_i].set_title(f'run {ax_i} (OLS)')
    axs[ax_i].legend()
    axs[ax_i].grid()

#%% Check cedalion.math.ar_irls.py ar_irls_GLM
ts = Y_test
stim_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
stim_dm.common = stim_dm.common.sel(chromo=['HbO'])
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
        for chan in tqdm(group_y.channel.values, disable=not verbose):
                result = cedalion.math.ar_irls.ar_irls_GLM(group_y.loc[:, chan], x)
                reg_results.loc[chan, dim3] = result

description = 'Cedalion'
reg_results.attrs["description"] = description

betas = reg_results.sm.params
y_hat = (design_matrix.common * betas).sum('regressor')
y_hat_stim_cedalion = y_hat.transpose('chromo', 'channel', 'time')

fig, axs = plt.subplots(1, 1, figsize=(14, 4), sharex=True)

axs.plot(ts.time.values, ts.values.flatten(), label='Y (true)', color='k', linewidth=1)
axs.plot(ts.time.values, y_hat_stim_cedalion.values.flatten(),
        label='Stim only (No HPF)', alpha=0.7)
axs.set_ylabel(f'{debug_chromo} concentration')
axs.set_title(f'Cedalion (AR-IRLS)')
axs.legend()
axs.grid()