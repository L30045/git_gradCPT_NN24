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
subj_id_array = [670, 671, 673, 695]
# subj_id_array = [670, 671, 673, 695, 719, 721, 723]
# subj_id_array = [719]
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    select_event = select_event,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit
)

#%% Check if preprocessed EEG exist. If not, preprocess.
subj_EEG_dict = dict()
for subj_id in tqdm(subj_id_array):
    subj_EEG_dict[f"sub-{subj_id}"] = dict()
    # get all the vdhr files in raw folder
    raw_EEG_path = os.path.join(data_path, f'sub-{subj_id}', 'eeg')
    preproc_save_path = os.path.join(data_save_path,f"sub-{subj_id}",'eeg')
    if not os.path.exists(preproc_save_path):
        os.makedirs(preproc_save_path, exist_ok=True)
    filename_list = [os.path.basename(x) for x in glob.glob(os.path.join(raw_EEG_path,"*.vhdr"))]
    # check if subject's EEG has been preprocessed.
    for fname in filename_list:
        preproc_fname = os.path.join(preproc_save_path,fname.split('.')[0]+'_preproc_eeg.fif')
        if not os.path.exists(preproc_fname):
            EEG = fix_and_load_brainvision(os.path.join(raw_EEG_path,fname),subj_id)
            EEG = eeg_preproc_basic(EEG, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                                is_reref=is_reref, reref_ch=reref_ch,
                                is_ica_rmEye=is_ica_rmEye)
            EEG.save(preproc_fname, overwrite=True)
        else:
            # load existed EEG
            EEG = mne.io.read_raw(preproc_fname,preload=True)
        # store into dict
        run_id = fname.split('.')[0][-1]
        if "cpt" in fname.lower():
            key_name = "gradcpt"+run_id
        else:
            key_name = "rest"+run_id
        subj_EEG_dict[f"sub-{subj_id}"][key_name] = EEG


#%% Epoch data
subj_epoch_dict = dict()
subj_vtc_dict = dict()
subj_react_dict = dict()
include_2_analysis = []
# for each subject
for subj_id in tqdm(subj_EEG_dict.keys()):
    subj_epoch_dict[subj_id] = dict()
    subj_vtc_dict[subj_id] = dict()
    subj_react_dict[subj_id] = dict()
    # check if event_file exist
    event_file = os.path.join(data_save_path,f"{subj_id}","eeg",
                            f"{subj_id}_task-gradCPT_run-01_events.tsv")
    if not os.path.exists(event_file):
        gen_EEG_event_tsv(int(subj_id.split('-')[-1]))
    # for each run
    for run_id in np.arange(1,4):
        subj_epoch_dict[subj_id][f"run{run_id:02d}"] = dict()
        subj_vtc_dict[subj_id][f"run{run_id:02d}"] = dict()
        subj_react_dict[subj_id][f"run{run_id:02d}"] = dict()
        EEG = subj_EEG_dict[subj_id][f"gradcpt{run_id}"]
        # load corresponding event file
        event_file = os.path.join(data_save_path,f"{subj_id}","eeg",
                                f"{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        events, event_labels_lookup, vtc_list, reaction_time = tsv_to_events(event_file, EEG.info["sfreq"])
        # for each condition
        for select_event in event_labels_lookup.keys():
            if np.any(events[:,-1]==event_labels_lookup[select_event]):
                ev_vtc = vtc_list[events[:,-1]==event_labels_lookup[select_event]]
                ev_react = reaction_time[events[:,-1]==event_labels_lookup[select_event]]
                event_duration = 1.6 if select_event.split('_')[-1]=='response' else 1.8
                baseline_length = -1.2 if select_event.split('_')[-1]=='response' else -0.2
                try:    
                    epochs = epoch_by_select_event(EEG, events, select_event=select_event,
                                                                baseline_length=baseline_length,
                                                                epoch_reject_crit=dict(eeg=100e-6),
                                                                is_detrend=1,
                                                                event_duration=event_duration,
                                                                verbose=False)
                    include_2_analysis.append((subj_id, f"run{run_id:02d}", select_event))
                    # remove vtc that is dropped
                    ev_vtc = ev_vtc[[len(x)==0 for x in epochs.drop_log]]
                    # remove reaction time that is dropped
                    ev_react = ev_react[[len(x)==0 for x in epochs.drop_log]]
                except:
                    print("="*20)
                    print(f"No clean trial found in {subj_id}_gradCPT{run_id} ({select_event}).")    
                    print("="*20)
                    epochs = []
            else:
                epochs=[]         
                ev_vtc = []         
                ev_react = []                                                  
            # save epochs
            subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event] = epochs
            subj_vtc_dict[subj_id][f"run{run_id:02d}"][select_event] = ev_vtc
            subj_react_dict[subj_id][f"run{run_id:02d}"][select_event] = ev_react

# save processed data for future use
save_data = dict(
    subj_epoch_dict=subj_epoch_dict,
    subj_vtc_dict=subj_vtc_dict,
    subj_react_dict=subj_react_dict,
    include_2_analysis=include_2_analysis
)
with open(os.path.join(data_save_path, f'subj_epochs_dict.pkl'), 'wb') as f:
    pickle.dump(save_data, f)

#%% combine runs for each subject
combine_epoch_dict = dict()
combine_vtc_dict = dict()
combine_react_dict = dict()
# get median of the vtc for each subject
subj_thres_vtc = {subj_id: np.median(np.concatenate([subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]
                                           for run_id in range(1, 4)
                                           for event in event_labels_lookup.keys()
                                           if len(subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]) > 0]))
                  for subj_id in subj_vtc_dict.keys()}
in_out_zone_dict = dict()
for select_event in event_labels_lookup.keys():
    epoch_list = []
    vtc_list = []
    react_list = []
    in_out_zone_list = []
    for subj_id in subj_epoch_dict.keys():
        tmp_epoch_list = []
        tmp_vtc_list = []
        tmp_react_list = []
        tmp_in_out_zone_list = []
        for run_id in np.arange(1,4):
            loc_e = subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_v = subj_vtc_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_r = subj_react_dict[subj_id][f"run{run_id:02d}"][select_event]
            if len(loc_e)>0:
                tmp_epoch_list.append(loc_e)
                tmp_vtc_list.append(loc_v)
                tmp_react_list.append(loc_r)
                tmp_in_out_zone_list.append(loc_v<subj_thres_vtc[subj_id])
        if len(tmp_epoch_list)>0:
            epoch_list.append(mne.concatenate_epochs(tmp_epoch_list,verbose=False))
            vtc_list.append(np.concatenate(tmp_vtc_list))
            react_list.append(np.concatenate(tmp_react_list))
            in_out_zone_list.append(np.concatenate(tmp_in_out_zone_list))
    combine_epoch_dict[select_event] = epoch_list
    combine_vtc_dict[select_event] = vtc_list
    combine_react_dict[select_event] = react_list
    in_out_zone_dict[select_event] = in_out_zone_list

#%% compare city and mountain ERP
is_save_fig = False
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['fz','cz','pz','oz']

# Extract cross-subject ERPs for both conditions
condition_data = {}
for select_event in select_events:
    subj_epoch_array = combine_epoch_dict[select_event]
    n_subjects = len(subj_epoch_array)
    xSubj_erps = []
    for epoch in subj_epoch_array:
        # Get average ERP for this subject
        evoked = epoch.average()
        subject_erps = []
        for ch_i in vis_ch:
            ch_idx = evoked.ch_names.index(ch_i)
            subject_erps.append(evoked.data[ch_idx, :])
        subject_erps = np.vstack(subject_erps)
        xSubj_erps.append(subject_erps)
    condition_data[select_event] = {'erps': xSubj_erps, 'n_subjects': n_subjects}

# Plot comparison for each channel
for ch_i in range(len(vis_ch)):
    plt.figure(figsize=(10, 6))

    for idx, select_event in enumerate(select_events):
        xSubj_erps = condition_data[select_event]['erps']
        n_subjects = condition_data[select_event]['n_subjects']

        plt_erps = np.vstack([x[ch_i,:] for x in xSubj_erps])
        # Calculate mean and SEM across subjects
        mean_erp = np.mean(plt_erps, axis=0)
        sem_erp = np.std(plt_erps, axis=0) / np.sqrt(n_subjects)
        upper_bound = mean_erp + 2 * sem_erp
        lower_bound = mean_erp - 2 * sem_erp

        # Get time vector and convert to milliseconds
        time_vector = combine_epoch_dict[select_event][0].times * 1000

        # Plot
        label = select_event.replace('_', ' ').title()
        plt.plot(time_vector, mean_erp, color=colors[idx], linewidth=2, label=f'{label} Mean')
        plt.fill_between(time_vector, lower_bound, upper_bound, alpha=0.3, color=colors[idx])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{vis_ch[ch_i].upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'mntC_vs_mntIC_{vis_ch[ch_i]}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()

#%% ERP Image
"""
Plot ERP Image and sorted by VTC. Merge all subjects's epochs into one big epoch.
"""
select_event = "mnt_correct"
ch_i = 'cz'
window_size = None  # Number of trials to average. If None, window_size equals to 1% of the data length.
clim = [-10*1e-6, 10*1e-6]
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[select_event])
time_vector = plt_epoch.times
plt_epoch.pick(ch_i)
plt_epoch = np.squeeze(plt_epoch.get_data())
if window_size is None:
    window_size = np.max([4,np.floor(plt_epoch.shape[0]*0.01).astype(int)])
plt_vtc = np.concatenate(combine_vtc_dict[select_event])
plt_react = np.concatenate(combine_react_dict[select_event])
title_txt = f'{select_event} - Channel: {ch_i}'

_ = plt_ERPImage(time_vector, plt_epoch, 
                 sort_idx=plt_vtc,
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react)

#%% ERSP analysis using multi-taper
start_time = time.time()
select_event = "city_correct"
ch_i = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[select_event])
time_vector = plt_epoch.times
plt_epoch.pick(ch_i)
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
ch_i = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch_target = mne.concatenate_epochs(combine_epoch_dict[target_event])
plt_epoch_ref = mne.concatenate_epochs(combine_epoch_dict[ref_event])
time_vector = plt_epoch_target.times
plt_epoch_target.pick(ch_i)
plt_epoch_ref.pick(ch_i)
(_,multitaper,_) = plt_multitaper(plt_epoch_target, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to=plt_epoch_ref)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")

#%% check in-zone vs out-of-zone ratio
for select_event in in_out_zone_dict.keys():
    if "_response" not in select_event:
        total_in_zone = 0
        total_out_zone = 0
        print(f"\n{select_event}:")
        for subj_i, in_out_zone in enumerate(in_out_zone_dict[select_event]):
            n_in_zone = np.sum(in_out_zone)
            n_out_zone = np.sum(~in_out_zone)
            total_in_zone += n_in_zone
            total_out_zone += n_out_zone
            print(f"  Subject {subj_i}: {n_in_zone} in-zone, {n_out_zone} out-of-zone")
        total_trials = total_in_zone + total_out_zone
        if total_trials > 0:
            print(f"  Total: {total_in_zone} in-zone ({total_in_zone/total_trials*100:.1f}%), {total_out_zone} out-of-zone ({total_out_zone/total_trials*100:.1f}%)")
        else:
            print(f"  Total: No trials found")

#%% zone-in vs zone-out
select_event = "mnt_correct"
vis_ch = ["fz","cz","pz","oz"]

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event]
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    subj_in_zone_erp = []
    subj_out_zone_erp = []
    for ch_i in vis_ch:
        # get channel data
        ch_erp = np.squeeze(epoch.get_data(picks=ch_i))
        # get in-zone/ out-of-zone data
        subj_in_zone_erp.append(np.mean(ch_erp[in_out_zone_dict[select_event][subj_i]],axis=0))
        subj_out_zone_erp.append(np.mean(ch_erp[~in_out_zone_dict[select_event][subj_i]],axis=0))
    in_zone_erp.append(np.vstack(subj_in_zone_erp))
    out_zone_erp.append(np.vstack(subj_out_zone_erp))

# Plot comparison for each channel
for ch_i in range(len(vis_ch)):
    plt_in_zone = np.vstack([x[ch_i] for x in in_zone_erp])
    plt_out_zone = np.vstack([x[ch_i] for x in out_zone_erp])

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
    time_vector = combine_epoch_dict[select_event][0].times * 1000

    # Plot
    plt.plot(time_vector, mean_in, color='b', linewidth=2, label='Mean (in-zone)')
    plt.fill_between(time_vector, lower_in, upper_in, alpha=0.3, color='b')
    plt.plot(time_vector, mean_out, color='r', linewidth=2, label='Mean (out-of-zone)')
    plt.fill_between(time_vector, lower_out, upper_out, alpha=0.3, color='r')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{vis_ch[ch_i].upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# %% In-zone/ out-of-zone ERSP
start_time = time.time()
select_event = "city_correct"
ch_i = 'cz'
time_halfbandwidth_product = 1
time_window_duration = 0.2
time_window_step = 0.1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask].copy().pick(ch_i)
    out_zone_epochs = epoch[out_zone_mask].copy().pick(ch_i)

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
ch_i = 'cz'
time_halfbandwidth_product = 1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask].copy().pick(ch_i)
    out_zone_epochs = epoch[out_zone_mask].copy().pick(ch_i)

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_city,_,connectivity) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            is_plot=False)
(log_power_out_city,_,connectivity) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            is_plot=False)

select_event = "mnt_correct"

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask].copy().pick(ch_i)
    out_zone_epochs = epoch[out_zone_mask].copy().pick(ch_i)

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_mnt,_,connectivity) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            is_plot=False)
(log_power_out_mnt,_,connectivity) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            is_plot=False)
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


