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

#%% settings
subj_id = 723
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")

#%% load design matrices and dependent variable (Y_all, as in model_EEG_inform_single_subject.py)
with open(os.path.join(save_file_path, 'dm_dict.pkl'), 'rb') as f:
    dm_dict = pickle.load(f)

Y_all = dm_dict['Y_all']  # dims: chromo, channel, time
y_true = Y_all.sel(channel=[debug_channel])

#%% retrain stim and EEG only
ar_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_full_cedalion.pkl')
with open(ar_path, 'rb') as f:
    old_full_result = pickle.load(f)
    autoReg_dict = old_full_result['autoReg_dict']

beta_dict = dict()

#%% open my_fit
ts = y_true
design_matrix = dm_dict['onlyStim']
autoReg = None
ar_order = 30
verbose = False

ts = ts.pint.dequantify()

dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


reg_results = xr.DataArray(
    np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
    dims=("channel", dim3_name),
    coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
)
autoReg_dict = dict()

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
        result = cedalion.math.ar_irls.ar_irls_GLM(group_y.loc[:, chan], x, pmax=ar_order)
        reg_results.loc[chan, dim3] = result

description='AR_IRLS' # FIXME
reg_results.attrs["description"] = description
betas = reg_results.sm.params

 # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
y_hat_ori_stim = (design_matrix.common * betas).sum('regressor')
y_hat_ori_stim = y_hat_ori_stim.transpose('chromo', 'channel', 'time')

#%% retrain stim-only model using original cedalion ar_irls_GLM instead of my_ar_irls_GLM
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)
y_pred = y_hat_ori_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Cedalion AR-IRLS', alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% train model using my_ar_irls_GLM
# stim only model
stim_results, stim_ar = model.my_fit(y_true, dm_dict['onlyStim'], autoReg=None)
betas = stim_results.sm.params
y_hat = (dm_dict['onlyStim'].common * betas).sum('regressor')
y_hat_my_stim = y_hat.transpose('chromo', 'channel', 'time')

#%% Using AR from Full
full_results, full_ar = model.my_fit(y_true, dm_dict['full'], autoReg=None)
betas = full_results.sm.params
y_hat = (dm_dict['full'].common * betas).sum('regressor')
y_hat_full = y_hat.transpose('chromo', 'channel', 'time')

stim_results, _ = model.my_fit(y_true, dm_dict['onlyStim'], autoReg=full_ar)
betas = stim_results.sm.params
y_hat = (dm_dict['onlyStim'].common * betas).sum('regressor')
y_hat_my_stim_full_ar = y_hat.transpose('chromo', 'channel', 'time')

#%% compare stim-only, EEG-only, and full model y_hat for a single channel
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)

y_pred = y_hat_my_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='My Stim', alpha=0.7)

y_pred = y_hat_ori_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Cedalion Stim', alpha=0.7)

y_pred = y_hat_full.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='My Full', alpha=0.7)


y_pred = y_hat_my_stim_full_ar.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Full-AR Stim', alpha=0.7)


plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#%%
subj_id = 723
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")
is_hpf = False

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
pruned_chans_list = []
stim_list = []
for run_key in run_dict.keys():
    local_run = run_dict[run_key]['run']
    # high pass filter
    if is_hpf:
        local_run = cedalion.sigproc.frequency.freq_filter(
            local_run, fmin=hpf_freq, fmax=0 * units.Hz, butter_order=4
        )
    run_list.append(local_run)
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
stim_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)

# get drift and ss
noDrift_cfg_GLM = copy.deepcopy(cfg_GLM)
noDrift_cfg_GLM['do_drift']=False
noDrift_cfg_GLM['do_drift_legendre']=False
ss_dm = model.create_no_info_dm(run_list, noDrift_cfg_GLM, noDrift_cfg_GLM['geo3d'], pruned_chans_list, stim_list)

# select chromo=HbO only to save time
Y_test = Y_all.sel(chromo=['HbO'],channel=[debug_channel])
ss_dm.common = ss_dm.common.sel(chromo=['HbO'])

#
glm_results, autoReg_dict = model.my_fit(Y_test, ss_dm)