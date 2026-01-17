"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
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
from spectral_connectivity import Multitaper, Connectivity
from spectral_connectivity.transforms import prepare_time_series

#%% preprocessing parameter setting
# subj_id_array = [670, 671, 673, 695]
# subj_id_array = [670, 671, 673, 695, 719, 721, 723, 726, 727, 730, 733]
subj_id_array = [670, 695, 719, 721, 723, 726, 727, 730]
ch_names = ['fz','cz','pz','oz']
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
# reref_ch = None # reref to average
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None
is_overwrite = False # Force to re run preprocessing if it is True

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    select_event = select_event,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names,
    is_overwrite = is_overwrite
)

#%% load epoch for each condition. Epoch from each run is combined for each subject.
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict, (subj_EEG_dict, subj_epoch_dict, subj_vtc_dict, subj_react_dict) = load_epoch_dict(subj_id_array, preproc_params)

#%% remove subjects with number of epoch less than half of the target number of epoch (2700/2)
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict = remove_subject_by_nb_epochs_preserved(subj_id_array, combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict)

#%% Compare in-zone/out-of-zone reaction time
check_ch = 'cz'
in_zone_RT = [x[y] for x,y in zip(combine_react_dict['city_correct'][check_ch],in_out_zone_dict['city_correct'][check_ch])]
out_zone_RT = [x[~y] for x,y in zip(combine_react_dict['city_correct'][check_ch],in_out_zone_dict['city_correct'][check_ch])]
print(f'RT diff (in/out zone) = {np.mean([np.mean(x)-np.mean(y) for x,y in zip(in_zone_RT,out_zone_RT)])*1000:.2f} ms')

# Plot distribution of RT differences
rt_diff_dist = [1000*(np.mean(x)-np.mean(y)) for x,y in zip(in_zone_RT,out_zone_RT)]
plt.figure(figsize=(8, 5))
plt.hist(rt_diff_dist, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(rt_diff_dist), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(rt_diff_dist):.2f} ms')
plt.xlabel('RT Difference (in-zone - out-zone) [ms]')
plt.ylabel('Frequency')
plt.title(f'Distribution of RT Differences ({check_ch.upper()})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Check VTC and Reaction time
plt_vtc = subj_vtc_dict['sub-733']['run02']['city_correct']
plt_react = subj_react_dict['sub-733']['run02']['city_correct']

# Calculate medians
median_vtc = np.median(plt_vtc)
median_react = np.median(plt_react)

# Create masks for VTC
above_median_vtc = plt_vtc >= median_vtc
below_median_vtc = plt_vtc < median_vtc

# Create arrays with NaN for segments we don't want to plot
vtc_above = np.where(above_median_vtc, plt_vtc, np.nan)
vtc_below = np.where(below_median_vtc, plt_vtc, np.nan)
react_above = np.where(above_median_vtc, plt_react, np.nan)
react_below = np.where(below_median_vtc, plt_react, np.nan)

plt.figure(figsize=(15,8))
# Plot VTC segments
plt.plot(vtc_above, '-o', color='red', label='VTC Out zone')
plt.plot(vtc_below, '-o', color='blue', label='VTC In zone')

# Plot React segments
plt.plot(react_above, color='orange', label=f'RT Out {np.mean(plt_react[above_median_vtc]):.3f}')
plt.plot(react_below, color='green', label=f'RT In {np.mean(plt_react[below_median_vtc]):.3f}')

# Add median lines
plt.axhline(y=median_vtc, color='purple', linestyle='--', alpha=0.7, label=f'VTC Median: {median_vtc:.2f}')
plt.axhline(y=median_react, color='brown', linestyle='--', alpha=0.7, label=f'React Median: {median_react:.2f}')

plt.legend()
plt.show()

#%% compare city and mountain ERP
is_save_fig = False
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['fz','cz','pz','oz']

# Extract cross-subject ERPs for both conditions
condition_data = {}
for select_event in select_events:
    condition_data[select_event] = dict()
    for ch in vis_ch:
        subj_epoch_array = combine_epoch_dict[select_event][ch]
        xSubj_erps = []
        for epoch in subj_epoch_array:
            # If subject epoch exist, Get average ERP for this subject
            if len(epoch)>0:
                evoked = epoch.average()
                xSubj_erps.append(evoked.data)
        xSubj_erps = np.vstack(xSubj_erps)
        condition_data[select_event][ch] = {'erps': xSubj_erps, 'n_subjects': xSubj_erps.shape[0]}

# Plot comparison for each channel
for ch in vis_ch:
    plt.figure(figsize=(10, 6))

    for idx, select_event in enumerate(select_events):
        # xSubj_erps = condition_data[select_event][ch]['erps']
        n_subjects = condition_data[select_event][ch]['n_subjects']
        plt_erps = condition_data[select_event][ch]['erps']
        # plt_erps = np.vstack([x[ch_i,:] for x in xSubj_erps])
        # Calculate mean and SEM across subjects
        mean_erp = np.mean(plt_erps, axis=0)
        sem_erp = np.std(plt_erps, axis=0) / np.sqrt(n_subjects)
        upper_bound = mean_erp + 2 * sem_erp
        lower_bound = mean_erp - 2 * sem_erp

        # Get time vector and convert to milliseconds
        time_vector = combine_epoch_dict[select_event][ch][0].times * 1000

        # Plot
        # label = select_event.replace('_', ' ').title()
        label = 'City' if select_event.split('_')[0]=='city' else 'Mountain'
        plt.plot(time_vector, mean_erp, color=colors[idx], linewidth=2, label=f'{label} ({n_subjects})')
        plt.fill_between(time_vector, lower_bound, upper_bound, alpha=0.3, color=colors[idx])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{ch.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'mntC_vs_mntIC_{ch}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()

#%% ERP Image
"""
Plot ERP Image and sorted by VTC. Merge all subjects's epochs into one big epoch.
"""
select_event = "mnt_correct"
ch = 'cz'
window_size = None  # Number of trials to average. If None, window_size equals to 1% of the data length.
clim = [-10*1e-6, 10*1e-6]
plt_epoch = mne.concatenate_epochs([x for x in combine_epoch_dict[select_event][ch] if len(x)>0])
time_vector = plt_epoch.times
plt_epoch = np.squeeze(plt_epoch.get_data())
if window_size is None:
    window_size = np.max([4,np.floor(plt_epoch.shape[0]*0.01).astype(int)])
plt_vtc = np.concatenate(combine_vtc_dict[select_event][ch])
plt_react = np.concatenate(combine_react_dict[select_event][ch])
title_txt = f'{select_event} - Channel: {ch}'

_ = plt_ERPImage(time_vector, plt_epoch, 
                 sort_idx=plt_vtc,
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react)

#%% ERSP analysis using multi-taper
start_time = time.time()
select_event = "mnt_correct"
ch = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[select_event][ch])
time_vector = plt_epoch.times
(_,multitaper,_) = plt_multitaper(plt_epoch,
                    time_halfbandwidth_product=time_halfbandwidth_product,
                    time_window_duration=time_window_duration,
                    time_window_step=time_window_step)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"ERSP analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

#%% plot ratio of power to baseline
(_,multitaper,_) = plt_multitaper(plt_epoch, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")

#%% compare trials
target_event = "mnt_correct"
ref_event = "city_correct"
ch = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch_target = mne.concatenate_epochs(combine_epoch_dict[target_event][ch])
plt_epoch_ref = mne.concatenate_epochs(combine_epoch_dict[ref_event][ch])
time_vector = plt_epoch_target.times
(_,multitaper,_) = plt_multitaper(plt_epoch_target, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to=plt_epoch_ref)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")

#%% check in-zone vs out-of-zone ratio
check_ch = 'cz'
for select_event in in_out_zone_dict.keys():
    if "_response" not in select_event:
        total_in_zone = 0
        total_out_zone = 0
        print(f"\n{select_event}:")
        for subj_i, in_out_zone in enumerate(in_out_zone_dict[select_event][check_ch]):
            n_in_zone = np.sum(in_out_zone)
            n_out_zone = np.sum(~in_out_zone)
            total_in_zone += n_in_zone
            total_out_zone += n_out_zone
            print(f"  Subject {subj_id_array[subj_i]}: {n_in_zone} in-zone, {n_out_zone} out-of-zone")
        total_trials = total_in_zone + total_out_zone
        if total_trials > 0:
            print(f"  Total: {total_in_zone} in-zone ({total_in_zone/total_trials*100:.1f}%), {total_out_zone} out-of-zone ({total_out_zone/total_trials*100:.1f}%)")
        else:
            print(f"  Total: No trials found")

#%% zone-in vs zone-out
select_event = "mnt_correct"
vis_ch = ["fz","cz","pz","oz"]

# Extract cross-subject ERPs for both conditions
in_zone_erp = dict()
out_zone_erp = dict()
for ch in vis_ch:
    subj_epoch_array = combine_epoch_dict[select_event][ch]
    n_subjects = len(subj_epoch_array)
    subj_in_zone_erp = []
    subj_out_zone_erp = []
    for subj_i, epoch in enumerate(subj_epoch_array):
        # get channel data
        ch_erp = np.squeeze(epoch.get_data())
        # get in-zone/ out-of-zone data
        subj_in_zone_erp.append(np.mean(ch_erp[in_out_zone_dict[select_event][ch][subj_i]],axis=0))
        subj_out_zone_erp.append(np.mean(ch_erp[~in_out_zone_dict[select_event][ch][subj_i]],axis=0))
    in_zone_erp[ch] = np.vstack(subj_in_zone_erp)
    out_zone_erp[ch] = np.vstack(subj_out_zone_erp)

# Plot comparison for each channel
for ch in vis_ch:
    plt_in_zone = in_zone_erp[ch]
    plt_out_zone = out_zone_erp[ch]

    plt.figure(figsize=(10, 6))
    # Calculate mean and SEM across subjects
    mean_in = np.mean(plt_in_zone, axis=0)
    sem_in = np.std(plt_in_zone, axis=0) / np.sqrt(n_subjects)
    upper_in = mean_in + 2 * sem_in
    lower_in = mean_in - 2 * sem_in
    mean_out = np.mean(plt_out_zone, axis=0)
    sem_out = np.std(plt_out_zone, axis=0) / np.sqrt(n_subjects)
    upper_out = mean_out + 2 * sem_out
    lower_out = mean_out - 2 * sem_out

    # Get time vector and convert to milliseconds
    time_vector = combine_epoch_dict[select_event][ch][0].times * 1000

    # Plot
    plt.plot(time_vector, mean_in, color='b', linewidth=2, label='In-zone')
    plt.fill_between(time_vector, lower_in, upper_in, alpha=0.3, color='b')
    plt.plot(time_vector, mean_out, color='r', linewidth=2, label='Out-of-zone')
    plt.fill_between(time_vector, lower_out, upper_out, alpha=0.3, color='r')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{ch.upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# %% In-zone/ out-of-zone ERSP
start_time = time.time()
select_event = "mnt_correct"
ch = 'cz'
time_halfbandwidth_product = 1
time_window_duration = 0.2
time_window_step = 0.1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)

# multitaper
print("In Zone")
_ = plt_multitaper(in_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print("Out of Zone")
_ = plt_multitaper(out_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print("In Zone / Out of Zone")
(_,multitaper,_) = plt_multitaper(in_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to=out_zone_erp)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"ERSP analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


#%% Compare PSD
select_event = "city_correct"
ch = 'cz'
time_halfbandwidth_product = 1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_city,_,_) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
(log_power_out_city,multitaper,_) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)

select_event = "mnt_correct"

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_mnt,_,_) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
(log_power_out_mnt,_,connectivity) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
vis_f_range = [0, 50] # Hz
vis_mask = (connectivity.frequencies>=vis_f_range[0])&(connectivity.frequencies<=vis_f_range[1])
plt.figure()
plt.plot(connectivity.frequencies[vis_mask], log_power_in_city[vis_mask], 'b-', label='City (in zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_out_city[vis_mask], 'b--', label='City (out of zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_in_mnt[vis_mask], 'r-', label='Mnt (in zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_out_mnt[vis_mask], 'r--', label='Mnt (out of zone)')
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Power ($log\,V^2$)")
plt.grid()
plt.legend()


