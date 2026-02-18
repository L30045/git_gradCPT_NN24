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
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d_695 = results['geo3d']

#%% f test from model results
subj_id_array = [670, 695, 721, 723]
sig_list = []
model_type = 'full'
model_cmp = 'f_test_full_eeg'

for subj_id in subj_id_array:
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_{model_type}.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # f_score_full_reduced = model.extract_val_across_channels(full_model_result['f_test'], chromo='HbO', stat_val='F')
    p_val_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp],
                                                           chromo='HbO', stat_val='p')
    # correct p-values using FDR
    rejected, p_values_fdr = fdrcorrection(p_val_full_reduced, alpha=0.05)
    sig_list.append(np.sum(rejected)/len(rejected))

plt.figure()
ratios = np.array(sig_list)*100
bars = plt.bar(np.arange(len(subj_id_array)), ratios)

# Add text labels on top of bars
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{ratio:.2f}%',  # Format to 3 decimal places
             ha='center', va='bottom', fontsize=14)

plt.xlabel("Subject ID", fontsize=14)
plt.ylabel("Proportion of significant channels (FDR p ≤ 0.05)", fontsize=12)
plt.xticks(np.arange(len(subj_id_array)), subj_id_array, ha='center')
plt.ylim([0,100])
plt.grid()
plt.tight_layout()

#%% visualize HRF
trial_type = 'mnt-correct'
chromo = 'HbO'
select_ch_crit = np.argmin
rss_all = np.sum(full_model_result['resid'].sel(chromo=chromo).values**2,axis=1)
pick_ch = full_model_result['resid'].channel.values[select_ch_crit(rss_all)]
hrf_estimate = full_model_result['hrf_estimate'].sel(chromo=chromo,trial_type=trial_type,channel=pick_ch)

#%% visualize RSS (log scale)
f, axs = plt.subplots(1, 1, figsize=(10,8))
scalp_plot(
    all_runs[0]['conc_o'],
    geo3d_695,
    np.log(rss_all),
    ax = axs,
    cmap='RdBu_r',
    # vmin=0,        
    # vmax=1,
    optode_labels=False,
    optode_size=6,
)
plt.tight_layout()

#%% get epoched concentration
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

#%%
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
    # (channel, trial, eopch length)
    hbo_dict[run_key]['mnt_incorrect'] = hbo_epoch

#%%

