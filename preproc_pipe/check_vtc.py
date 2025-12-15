#%% load library
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from tqdm import tqdm
import pickle
import glob
import time
import sys

#%%
subj_id = 695
# smooth setting
L = 20 # trial
radius = np.ceil((L-1)/2).astype(int)
alpha = 2.5 # default in Matlab
sigma = (L-1)/(2*alpha)

# for each run
original_vtc = []
smoothed_vtc = []
rt_array = []

filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/VTC_analysis/sub-{subj_id}"
# original
with open(os.path.join(filepath,f'sub-{subj_id}_VTC_original.pkl'),'rb') as f:
    ori_vtc = pickle.load(f)
# smooth
with open(os.path.join(filepath,f'sub-{subj_id}_VTC_smoothed.pkl'),'rb') as f:
    smooth_vtc = pickle.load(f)

#%% load mat from Michael's code
mat_vtc = sp.io.loadmat(os.path.join(os.pardir,"matlab_vtc.mat"))
mat_vtc_run1 = mat_vtc["VARIABILITY_TC"]
py_vtc_run1 = smooth_vtc["run-01"]
fig, ax = plt.subplots(2,1)
ax[0].plot(mat_vtc_run1,label='Matlab')
ax[0].plot(py_vtc_run1,label='Python')
ax[1].plot(mat_vtc_run1.flatten()-py_vtc_run1, label=f'Mat-Py ({np.sum(mat_vtc_run1.flatten()-py_vtc_run1):.3f})')
for ax_i in range(2):
    ax[ax_i].legend()
    ax[ax_i].grid()
plt.show()


#%%
for run_id in np.arange(1,4):
    # load corresponding event file
    event_file = os.path.join(data_save_path,f"sub-{subj_id}",
                            f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
    events_df = pd.read_csv(event_file,sep='\t')
    event_ids = events_df["response_code"].astype(int)
    rt_array.append(events_df["reaction_time"])
    
    original_vtc.append(ori_vtc[f"run-0{run_id}"])
    smoothed_vtc.append(smooth_vtc[f"run-0{run_id}"])

# visualization - create figure with 3 subplots for each run
fig, axes = plt.subplots(3,1, figsize=(12, 10))

mean_rt_out = []
mean_rt_in = []
for run_id in range(3):
    ax = axes[run_id]
    plt_vtc = smoothed_vtc[run_id]
    plt_react = rt_array[run_id]

    # Remove trials where reaction time is 0
    valid_trials = plt_react != 0
    plt_vtc_valid = plt_vtc[valid_trials]
    plt_react_valid = plt_react[valid_trials]

    # Calculate medians using all trials
    median_vtc = np.median(plt_vtc)
    
    # Create masks for VTC based on valid trials
    above_median_vtc = plt_vtc_valid >= median_vtc
    below_median_vtc = plt_vtc_valid < median_vtc

    # Create arrays with NaN for segments we don't want to plot
    vtc_above = np.where(above_median_vtc, plt_vtc_valid, np.nan)
    vtc_below = np.where(below_median_vtc, plt_vtc_valid, np.nan)
    react_above = np.where(above_median_vtc, plt_react_valid, np.nan)
    react_below = np.where(below_median_vtc, plt_react_valid, np.nan)

    # Calculate mean RT for above and below median VTC
    mean_rt_above = np.nanmean(react_above)
    mean_rt_below = np.nanmean(react_below)
    mean_rt_out.append(mean_rt_above)
    mean_rt_in.append(mean_rt_below)

    # Plot original VTC (all trials including zeros)
    ax.plot(original_vtc[run_id], color='grey', label='VTC (original)', alpha=0.3)

    # Plot VTC segments (only valid trials)
    valid_indices = np.where(valid_trials)[0]
    ax.plot(valid_indices[above_median_vtc], vtc_above[above_median_vtc], '-o', color='red',
            label=f'VTC Out zone (RT: {mean_rt_above:.3f}s)')
    ax.plot(valid_indices[below_median_vtc], vtc_below[below_median_vtc], '-o', color='blue',
            label=f'VTC In zone (RT: {mean_rt_below:.3f}s)')

    # Add median lines
    ax.axhline(y=median_vtc, color='purple', linestyle='--', alpha=0.7, label=f'VTC Median: {median_vtc:.2f}')

    ax.set_xlabel('Trials')
    ax.set_ylabel('VTC')
    ax.legend()
    ax.set_title(f"Sub-{subj_id}, run {run_id+1:02d}")

plt.tight_layout()
plt.show()

#
print(f"Mean In-Zone RT = {np.mean(mean_rt_in)*1000:.2f} ms")
print(f"Mean Out-of-Zone RT = {np.mean(mean_rt_out)*1000:.2f} ms")
print(f"Difference between mean in/out zone = {(np.mean(mean_rt_in)-np.mean(mean_rt_out))*1000:.2f} ms")

#%% change smoothing parameter
sigma_list = np.arange(1,100,0.5)
rt_diff = []
for run_id in np.arange(1,4):
    loc_rt_diff = []
    # load corresponding event file
    event_file = os.path.join(data_save_path,f"sub-{subj_id}","eeg",
                            f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
    events_df = pd.read_csv(event_file,sep='\t')
    event_ids = events_df["response_code"].astype(int)
    original_vtc = events_df["VTC"]
    rt = events_df["reaction_time"]
    # get only non-zero RT
    rt_valid = rt[rt!=0]
    # Calculate medians using all trials
    median_vtc = np.median(original_vtc)
    # Create masks for VTC based on valid trials
    above_median_vtc = original_vtc[rt!=0] >= median_vtc
    below_median_vtc = original_vtc[rt!=0] < median_vtc
    # Create arrays with NaN for segments we don't want to plot
    react_out = np.where(above_median_vtc, rt_valid, np.nan)
    react_in = np.where(below_median_vtc, rt_valid, np.nan)
    # Calculate mean RT for above and below median VTC
    loc_rt_diff.append(np.nanmean(react_in)-np.nanmean(react_out))
    # smooth VTC using Gaussian window
    for sigma in sigma_list:
        smoothed_vtc = gaussian_filter1d(original_vtc, sigma=sigma, truncate=4) # kernel size = round(truncate*sigma)*2+1
        # Calculate medians using all trials
        median_vtc = np.median(smoothed_vtc)
        # Create masks for VTC based on valid trials
        above_median_vtc = smoothed_vtc[rt!=0] >= median_vtc
        below_median_vtc = smoothed_vtc[rt!=0] < median_vtc
        # Create arrays with NaN for segments we don't want to plot
        react_out = np.where(above_median_vtc, rt_valid, np.nan)
        react_in = np.where(below_median_vtc, rt_valid, np.nan)
        # Calculate mean RT for above and below median VTC
        loc_rt_diff.append(np.nanmean(react_in)-np.nanmean(react_out))
    rt_diff.append(np.array(loc_rt_diff))
mean_rt_diff = np.mean(np.vstack(rt_diff),axis=0)
plt.figure()
plt.plot(np.insert(sigma_list,0,0), mean_rt_diff*1000,label="Mean RT diff (In - Out)")
plt.legend()
plt.grid()
plt.xlabel("Sigma for gaussian smoothing")
plt.ylabel("Time diff (ms)")

