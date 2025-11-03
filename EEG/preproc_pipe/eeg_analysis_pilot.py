"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from tqdm import tqdm
import pickle
import glob


#%% preprocessing parameter setting
subj_id_array = [670, 671, 673, 695]
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
        subj_EEG_dict[f"sub-{subj_id}"][fname.split('.')[0].split('_')[-1].lower()] = EEG


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
        gen_EEG_event_tsv(subj_id)
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
                                                                event_duration=event_duration)
                    include_2_analysis.append((subj_id, f"run{run_id:02d}", select_event))
                    # remove vtc that is dropped
                    ev_vtc = ev_vtc[[len(x)==0 for x in epochs.drop_log]]
                    # remove reaction time that is dropped
                    ev_react = ev_react[[len(x)==0 for x in epochs.drop_log]]
                except:
                    print("="*20)
                    print(f"No clean trial found in {subj_id}_gradCPT{run_id} ({select_event}).")    
                    print("="*20)
                    # epochs = epoch_by_select_event(EEG, events, select_event=select_event,
                    #                                             baseline_length=baseline_length,
                    #                                             epoch_reject_crit=None,
                    #                                             is_detrend=1,
                    #                                             event_duration=event_duration)
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

# combine runs for each subject
combine_epoch_dict = dict()
combine_vtc_dict = dict()
combine_react_dict = dict()
for select_event in event_labels_lookup.keys():
    epoch_list = []
    vtc_list = []
    react_list = []
    for subj_id in subj_epoch_dict.keys():
        tmp_epoch_list = []
        tmp_vtc_list = []
        tmp_react_list = []
        for run_id in np.arange(1,4):
            loc_e = subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_v = subj_vtc_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_r = subj_react_dict[subj_id][f"run{run_id:02d}"][select_event]
            if len(loc_e)>0:
                tmp_epoch_list.append(loc_e)
                tmp_vtc_list.append(loc_v)
                tmp_react_list.append(loc_r)
        if len(tmp_epoch_list)>0:
            epoch_list.append(mne.concatenate_epochs(tmp_epoch_list,verbose=False))
            vtc_list.append(np.concatenate(tmp_vtc_list))
            react_list.append(np.concatenate(tmp_react_list))
    combine_epoch_dict[select_event] = epoch_list
    combine_vtc_dict[select_event] = vtc_list
    combine_react_dict[select_event] = react_list

#%% Visualizing
# sanity check with one subject
# plt_epoch = subj_epoch_array[3]
# vis_ch = 'cz'
# plt_center, plt_shade, plt_time = plot_ch_erp(plt_epoch, vis_ch, is_return_data=True)

#%% cross-subjects results
is_save_fig = False
select_event = 'mnt_correct'
subj_epoch_array = combine_epoch_dict[select_event]
# Plot mean and +/- 2 SEM across subjects
vis_ch = ['fz','cz','pz','oz']
# Extract data for the selected channel from all subjects
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

for ch_i in range(len(vis_ch)):
    plt_erps = np.vstack([x[ch_i,:] for x in xSubj_erps])
    # Calculate mean and SEM across subjects
    mean_erp = np.mean(plt_erps, axis=0)
    sem_erp = np.std(plt_erps, axis=0) / np.sqrt(n_subjects)
    upper_bound = mean_erp + 2 * sem_erp
    lower_bound = mean_erp - 2 * sem_erp

    # Get time vector
    time_vector = subj_epoch_array[0].times

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, mean_erp, 'b-', linewidth=2, label='Mean')
    plt.fill_between(time_vector, lower_bound, upper_bound, alpha=0.3, color='b', label='±2 SEM')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'Cross-Subject ERP at {vis_ch[ch_i].upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'xSubject_ERP_{select_event}_{vis_ch[ch_i]}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()

#%% compare city and mountain ERP
is_save_fig = False
select_events = ['city_correct_response', 'mnt_incorrect_response']
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

    colors = ['b', 'r']
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
window_size = 5  # Number of trials to average
clim = [-10*1e-6, 10*1e-6]
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[select_event])
time_vector = plt_epoch.times
plt_epoch.pick(ch_i)
plt_epoch = np.squeeze(plt_epoch.get_data())
plt_vtc = np.concatenate(combine_vtc_dict[select_event])
plt_react = np.concatenate(combine_react_dict[select_event])
title_txt = f'{select_event} - Channel: {ch_i}'

_ = plt_ERPImage(time_vector, plt_epoch, 
                 sort_idx=plt_vtc,
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react)

