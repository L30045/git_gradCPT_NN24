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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# load template run and geo3d
subj_id_array = [670, 695, 721, 723]
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d_695 = results['geo3d']

#%% get sig channels
sig_list = []
model_type = 'full'
model_cmp = 'f_test_full_stim'
ch_crit = 'f' # 'rss'

# for HRF
fig, axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.flatten()

# for scalp plot
f, axs = plt.subplots(2, 2, figsize=(10,8))
axs = axs.flatten()

for s_i, subj_id in enumerate(subj_id_array):
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_{model_type}.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
    clean_chs = full_model_result['resid'].channel.values[clean_chs_idx]
    # get stats
    f_score_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp], chromo='HbO', stat_val='F')
    p_val_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp],
                                                        chromo='HbO', stat_val='p')
    # remove bad channels
    f_score_full_reduced = f_score_full_reduced[clean_chs_idx]
    p_val_full_reduced = p_val_full_reduced[clean_chs_idx]
    # correct p-values using FDR
    rejected, p_values_fdr = fdrcorrection(p_val_full_reduced, alpha=0.05)
    sig_list.append(np.sum(rejected)/len(rejected))

    # visualize HRF
    trial_type = 'mnt-correct'
    chromo = 'HbO'
    select_ch_crit = np.argmin
    if ch_crit == 'rss':
        rss_all = np.sum(full_model_result['resid'].sel(chromo=chromo,channel=clean_chs).values**2,axis=1)
        pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[select_ch_crit(rss_all)]
    else:
        pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[np.argmax(f_score_full_reduced)]
    hrf_estimate = full_model_result['hrf_estimate'].sel(chromo=chromo,trial_type=trial_type,channel=pick_ch)
    # plt.figure()
    # plt.plot(hrf_estimate)

    # visualize select channel
    plt_scalp = np.where(full_model_result['resid'].channel.values==pick_ch, 1, np.nan)
    
    model.scalp_plot(
        all_runs[0]['conc_o'],
        geo3d_695,
        plt_scalp,
        ax = axs[s_i],
        cmap='RdBu_r',
        vmin=0,        
        vmax=1,
        optode_labels=False,
        optode_size=6,
    )

    # get epoched concentration
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
    for r_i, run in enumerate(all_runs):
        # Match this run to the correct run_dict entry by comparing first event
        for run_key in run_dict.keys():
            ev_df = run_dict[run_key]['ev_df']
            if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
                run_dict[run_key]['run'] = run[0]
                run_dict[run_key]['conc_ts'] = run['conc_o']
                run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
                break

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

    hbo_dict = dict()
    len_epoch = 12 # seconds

    for run_key in run_dict.keys():
        hbo_dict[run_key] = dict()
        run = run_dict[run_key]['conc_ts'].copy()
        run = run.pint.dequantify()
        ev_df = run_dict[run_key]['ev_df'].copy()
        # get time
        sfreq_conc = 1/np.diff(run.time)[0]
        len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)
        # rename trial_type
        ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
        ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
        mnt_correct_df = ev_df[(ev_df['trial_type']=='mnt-correct-stim')]
        mnt_incorrect_df = ev_df[(ev_df['trial_type']=='mnt-incorrect-stim')]

        # mnt correct trial (channel, trial, trial length)
        _, nb_ch, len_record = run.values.shape
        hbo_epoch = np.zeros((nb_ch,len(ev_dict[run_key]['mnt_correct']['idx']['preserved']),len_epoch_sample))
        for p_i, p_idx in enumerate(ev_dict[run_key]['mnt_correct']['idx']['preserved']):
            t_onset = mnt_correct_df['onset'].values[p_idx]
            ev_i = np.where(run.time.values>=t_onset)[0][0]
            if ev_i+len_epoch_sample > len_record:
                hbo_epoch[:,p_i,:] = np.nan
                continue
            hbo_epoch[:,p_i,:] = run.values[0,:,ev_i:ev_i+len_epoch_sample]
        # remove bad channels
        hbo_epoch = hbo_epoch[clean_chs_idx]
        # (channel, trial, eopch length)
        hbo_dict[run_key]['mnt_correct'] = hbo_epoch

        # mnt incorrect trial (channel, trial, trial length)
        hbo_epoch = np.zeros((nb_ch,len(ev_dict[run_key]['mnt_incorrect']['idx']['preserved']),len_epoch_sample))
        for p_i, p_idx in enumerate(ev_dict[run_key]['mnt_incorrect']['idx']['preserved']):
            t_onset = mnt_correct_df['onset'].values[p_idx]
            ev_i = np.where(run.time.values>=t_onset)[0][0]
            if ev_i+len_epoch_sample > len_record:
                hbo_epoch[:,p_i,:] = np.nan
                continue
            hbo_epoch[:,p_i,:] = run.values[0,:,ev_i:ev_i+len_epoch_sample]
        # remove bad channels
        hbo_epoch = hbo_epoch[clean_chs_idx]
        # (channel, trial, eopch length)
        hbo_dict[run_key]['mnt_incorrect'] = hbo_epoch

    # visualize HRF
    plt_hbo = []
    plt_time = np.arange(0, 12, 1/sfreq_conc)
    for run_key in hbo_dict.keys():
        plt_hbo.append(hbo_dict[run_key]['mnt_correct'][pick_ch_i])
    plt_hbo = np.vstack(plt_hbo)


    hbo_mean = np.mean(plt_hbo, axis=0)
    hbo_sem = np.std(plt_hbo, axis=0) / np.sqrt(plt_hbo.shape[0])
    # axes[s_i].plot(plt_time, hbo_mean, 'b', label='Ground Truth HbO')
    # axes[s_i].fill_between(plt_time, hbo_mean - 1.96*hbo_sem, hbo_mean + 1.96*hbo_sem, color='b', alpha=0.2, label='95% CI')
    axes[s_i].plot(np.arange(0, len(hrf_estimate)/sfreq_conc, 1/sfreq_conc), hrf_estimate,'g',label='HRF')
    axes[s_i].legend()
    axes[s_i].grid()

plt.tight_layout()
plt.show()


#%% compare Full model vs Stim model
model_cmp = 'f_test_full_stim'
ch_crit = 'f' # 'rss'

# for HRF
fig, axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.flatten()

# for scalp plot
f, axs = plt.subplots(2, 2, figsize=(10,8))
axs = axs.flatten()

for s_i, subj_id in enumerate(subj_id_array):
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load reduced model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        reduced_model_result = pickle.load(f)
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
    clean_chs = full_model_result['resid'].channel.values[clean_chs_idx]
    f_score_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp], chromo='HbO', stat_val='F')
    p_val_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp],
                                                        chromo='HbO', stat_val='p')
    # remove bad channels
    f_score_full_reduced = f_score_full_reduced[clean_chs_idx]
    p_val_full_reduced = p_val_full_reduced[clean_chs_idx]
    # correct p-values using FDR
    rejected, p_values_fdr = fdrcorrection(p_val_full_reduced, alpha=0.05)
    sig_list.append(np.sum(rejected)/len(rejected))

    # visualize HRF
    trial_type = 'mnt-correct'
    chromo = 'HbO'
    select_ch_crit = np.argmin
    if ch_crit == 'rss':
        rss_all = np.sum(full_model_result['resid'].sel(chromo=chromo,channel=clean_chs).values**2,axis=1)
        pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[select_ch_crit(rss_all)]
    else:
        pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[np.argmax(f_score_full_reduced)]
    hrf_estimate_full = full_model_result['hrf_estimate'].sel(chromo=chromo,trial_type=trial_type,channel=pick_ch)
    hrf_estimate_reduced = reduced_model_result['hrf_estimate'].sel(chromo=chromo,trial_type=trial_type,channel=pick_ch)
    # plt.figure()
    # plt.plot(hrf_estimate)

    # visualize RSS scalp plot (log scale)
    plt_scalp = np.where(full_model_result['resid'].channel.values==pick_ch, 1, np.nan)
    
    model.scalp_plot(
        all_runs[0]['conc_o'],
        geo3d_695,
        plt_scalp,
        ax = axs[s_i],
        cmap='RdBu_r',
        vmin=0,        
        vmax=1,
        optode_labels=False,
        optode_size=6,
    )

    # visualize HRF
    axes[s_i].plot(np.arange(0, len(hrf_estimate_full)/sfreq_conc, 1/sfreq_conc), hrf_estimate_full,'g',label='HRF (Full)')
    axes[s_i].plot(np.arange(0, len(hrf_estimate_reduced)/sfreq_conc, 1/sfreq_conc), hrf_estimate_reduced,'r',label='HRF (Reduced)')
    axes[s_i].legend()
    axes[s_i].grid()

plt.tight_layout()
plt.show()

#%% estimate HRF based on Design matrix and sorted by EEG amplitude
subj_id = 723
ch_crit = 'rss'
model_cmp = 'f_test_full_stim'
filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
# load runs
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
for r_i, run in enumerate(all_runs):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        ev_df = run_dict[run_key]['ev_df']
        if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
            run_dict[run_key]['run'] = run[0]
            run_dict[run_key]['conc_ts'] = run['conc_o']
            run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
            break
# load full model
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
    full_model_result = pickle.load(f)
fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
    results = pickle.load(f)
    hrf_per_subj = results['hrf_per_subj']
    hrf_mse_per_subj = results['hrf_mse_per_subj']
    bad_indices = results['bad_indices']
clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
clean_chs = full_model_result['resid'].channel.values[clean_chs_idx]
# get stats
f_score_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp], chromo='HbO', stat_val='F')
# remove bad channels
f_score_full_reduced = f_score_full_reduced[clean_chs_idx]

# visualize HRF
trial_type = 'mnt-correct'
chromo = 'HbO'
select_ch_crit = np.argmin
if ch_crit == 'rss':
    rss_all = np.sum(full_model_result['resid'].sel(chromo=chromo,channel=clean_chs).values**2,axis=1)
    pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[select_ch_crit(rss_all)]
else:
    pick_ch = full_model_result['resid'].channel.sel(channel=clean_chs).values[np.argmax(f_score_full_reduced)]

# load betas
betas = full_model_result['betas'].sel(chromo=chromo,channel=clean_chs,regressor=full_model_result['betas'].regressor.str.startswith(f"HRF {trial_type}"))

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

# hrf estimation for each trial
hbo_list = []
area_list = []
trial_type_key = 'mnt_correct'
for run_key in ev_dict.keys():
    # define basis hrf
    basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])
    # for each trial
    for area in ev_dict[run_key][trial_type_key]['area']['cz']:
        basis_hrf_eeg = basis_hrf.copy()
        basis_hrf_eeg.values = basis_hrf_eeg.values*area
        trial_hrf = model.xr.concat([basis_hrf.copy(),basis_hrf_eeg],dim='component')
        hrf_estimate = model.estimate_HRF_from_beta(betas, trial_hrf)
        hbo_list.append(hrf_estimate)
        area_list.append(area)
# concatenate all trials
hbo_epoch = model.xr.concat(hbo_list, dim='trial')
area_list = np.array(area_list)

#%% pick ch to visualize
time_vector = hbo_epoch.time.values
plt_hrf = hbo_epoch.sel(channel=pick_ch).values
plt_ERPImage(time_vector, plt_hrf, sort_idx=area_list, smooth_window_size=10, clim=[-1*1e-6, 1*1e-6], title_txt=None, ref_onset=None, sort_label='EEG scale factor')

# visualize RSS scalp plot (log scale)
plt_scalp = np.where(full_model_result['resid'].channel.values==pick_ch, 1, np.nan)

fig, ax = plt.subplots(1,1)
model.scalp_plot(
    all_runs[0]['conc_o'],
    geo3d_695,
    plt_scalp,
    ax = ax,
    cmap='RdBu_r',
    title=pick_ch,
    vmin=0,        
    vmax=1,
    optode_labels=False,
    optode_size=6,
)


#%% pick a channel that everyone has
clean_chs_set = set()
for subj_id in subj_id_array:
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
    clean_chs = hrf_per_subj.channel.values[clean_chs_idx]
    if len(clean_chs_set)==0:
        clean_chs_set = set(clean_chs)
    else:
        clean_chs_set = clean_chs_set.intersection(set(clean_chs))
pick_ch = np.random.choice(list(clean_chs_set))
# pick_ch = 'S18D107'
# visualize RSS scalp plot (log scale)
plt_scalp = np.where(hrf_per_subj.channel.values==pick_ch, 1, np.nan)

fig, ax = plt.subplots(1,1)
model.scalp_plot(
    all_runs[0]['conc_o'],
    geo3d_695,
    plt_scalp,
    ax = ax,
    cmap='RdBu_r',
    title=pick_ch,
    vmin=0,        
    vmax=1,
    optode_labels=False,
    optode_size=6,
)


#%%
plt_hrf_list = []
trial_type = 'mnt-correct'
trial_type_key = 'mnt_correct'

for subj_id in subj_id_array:
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load runs
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
    for r_i, run in enumerate(all_runs):
        # Match this run to the correct run_dict entry by comparing first event
        for run_key in run_dict.keys():
            ev_df = run_dict[run_key]['ev_df']
            if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
                run_dict[run_key]['run'] = run[0]
                run_dict[run_key]['conc_ts'] = run['conc_o']
                run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
                break

    # load betas
    betas = full_model_result['betas'].sel(chromo='HbO',channel=clean_chs,regressor=full_model_result['betas'].regressor.str.startswith(f"HRF {trial_type}"))

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

    # hrf estimation for each trial
    hbo_list = []
    area_list = []
    for run_key in ev_dict.keys():
        # define basis hrf
        basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])
        # for each trial
        for area in ev_dict[run_key][trial_type_key]['area']['cz']:
            basis_hrf_eeg = basis_hrf.copy()
            basis_hrf_eeg.values = basis_hrf_eeg.values*area
            trial_hrf = model.xr.concat([basis_hrf.copy(),basis_hrf_eeg],dim='component')
            hrf_estimate = model.estimate_HRF_from_beta(betas, trial_hrf)
            hbo_list.append(hrf_estimate)
            area_list.append(area)
    # concatenate all trials
    hbo_epoch = model.xr.concat(hbo_list, dim='trial')
    area_list = np.array(area_list)

    # pick ch to visualize
    time_vector = hbo_epoch.time.values
    plt_hrf = hbo_epoch.sel(channel=pick_ch).values
    plt_hrf_list.append(plt_hrf)
    plt_ERPImage(time_vector, plt_hrf, sort_idx=area_list, smooth_window_size=10, clim=[-10*1e-6, 10*1e-6], title_txt=None, ref_onset=None, sort_label='EEG scale factor')

# # visualize RSS scalp plot (log scale)
# plt_scalp = np.where(full_model_result['resid'].channel.values==pick_ch, 1, np.nan)

# fig, ax = plt.subplots(1,1)
# model.scalp_plot(
#     all_runs[0]['conc_o'],
#     geo3d_695,
#     plt_scalp,
#     ax = ax,
#     cmap='RdBu_r',
#     title=pick_ch,
#     vmin=0,        
#     vmax=1,
#     optode_labels=False,
#     optode_size=6,
# )

# plt.figure(figsize=(1,2.5))
# plt.plot(np.sort(area_list), np.arange(len(area_list)))  # swap x and y
# plt.grid()
# plt.xlabel('EEG scale factor')

#%%
avg_hrf = np.vstack([np.mean(x,axis=0) for x in plt_hrf_list])
mean_hrf = np.mean(avg_hrf, axis=0)
sem_hrf = np.std(avg_hrf, axis=0) / np.sqrt(avg_hrf.shape[0])

plt.figure()
plt.plot(time_vector, mean_hrf, label='mean')
plt.fill_between(time_vector, mean_hrf - sem_hrf, mean_hrf + sem_hrf, alpha=0.3, label='SEM')
plt.xlabel('Time (s)')
plt.ylabel('HbO (M)')
plt.title(f'{pick_ch} - {trial_type}')
plt.legend()
plt.grid()
