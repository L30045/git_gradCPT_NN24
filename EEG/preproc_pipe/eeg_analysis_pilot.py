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


#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata/raw")
project_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24")
subj_id_array = [670, 671, 673, 695]
fig_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/plots/EEG")
data_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/processed_data")


#%% preprocessing parameter setting
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
        preproc_fname = os.path.join(preproc_save_path,fname.split('.')[0]+'_preproc.fif')
        if not os.path.exists(preproc_fname):
            EEG = fix_and_load_brainvision(os.path.join(raw_EEG_path,fname),subj_id)
            EEG = eeg_preproc_basic(EEG, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                                is_reref=is_reref, reref_ch=reref_ch,
                                is_ica_rmEye=is_ica_rmEye)
            EEG.save(preproc_fname)
        else:
            # load existed EEG
            EEG = mne.io.read_raw(preproc_fname,preload=True)
        subj_EEG_dict[f"sub-{subj_id}"][fname.split('.')[0].split('_')[-1].lower()] = EEG


#%% Epoch data
subj_epoch_dict = dict()
include_2_analysis = []
# for each subject
for subj_id in tqdm(subj_EEG_dict.keys()):
    subj_epoch_dict[subj_id] = dict()
    # for each run
    for run_id in np.arange(1,4):
        subj_epoch_dict[subj_id][f"run{run_id:02d}"] = dict()
        EEG = subj_EEG_dict[subj_id][f"gradcpt{run_id}"]
        # load corresponding event file
        event_file = os.path.join(project_path,f"{subj_id}","nirs",
                                f"{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        events, event_labels_lookup = tsv_to_events(event_file, EEG.info["sfreq"])
        # for each condition
        for select_event in event_labels_lookup.keys():
            if np.any(events[:,-1]==event_labels_lookup[select_event]):
                event_duration = 1 if select_event.split('_')[-1]=='response' else 0.8
                baseline_length = -0.5 if select_event.split('_')[-1]=='response' else -0.2
                try:    
                    epochs = epoch_by_select_event(EEG, events, select_event=select_event,
                                                                baseline_length=baseline_length,
                                                                epoch_reject_crit=dict(eeg=100e-6),
                                                                is_detrend=1,
                                                                event_duration=event_duration)
                    include_2_analysis.append((subj_id, f"run{run_id:02d}", select_event))
                except:
                    print("="*20)
                    print(f"No clean trial found in {subj_id}_gradCPT{run_id}.")    
                    print("="*20)
                    epochs = epoch_by_select_event(EEG, events, select_event=select_event,
                                                                baseline_length=baseline_length,
                                                                epoch_reject_crit=None,
                                                                is_detrend=1,
                                                                event_duration=event_duration)
            else:
                epochs=[]                                                                    
            # save epochs
            subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event] = epochs

# save processed data for future use
save_data = dict(
    subj_epoch_dict=subj_epoch_dict,
    include_2_analysis=include_2_analysis
)
with open(os.path.join(data_save_path, f'subj_epochs_dict.pkl'), 'wb') as f:
    pickle.dump(save_data, f)

#%% combine runs for each subject
combine_epoch_dict = dict()
for select_event in event_labels_lookup.keys():
    epoch_list = []
    for subj_id in subj_epoch_dict.keys():
        tmp_epoch_list = []
        for run_id in np.arange(1,4):
            loc_e = subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event]
            if len(loc_e)>0:
                tmp_epoch_list.append(loc_e)
        if len(tmp_epoch_list)>0:
            epoch_list.append(mne.concatenate_epochs(tmp_epoch_list,verbose=False))
    combine_epoch_dict[select_event] = epoch_list

#%% Visualizing
# sanity check with one subject
# plt_epoch = subj_epoch_array[3]
# vis_ch = 'cz'
# plt_center, plt_shade, plt_time = plot_ch_erp(plt_epoch, vis_ch, is_return_data=True)

#%% cross-subjects results
is_save_fig = False
select_event = 'city_correct'
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
is_save_fig = True
select_events = ['city_correct', 'mnt_correct']
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
    plt.title(f'City vs Mountain ERP Comparison at {vis_ch[ch_i].upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'xSubject_ERP_cityC_vs_mntC_{vis_ch[ch_i]}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()



#%% visual check EEG
subj_id = 670
run_id = 2
raw_EEG_path = os.path.join(data_path, 'raw', f'sub-{subj_id}', 'eeg')
run_path = os.path.join(raw_EEG_path, f'sub-{subj_id}_gradCPT{run_id}.vhdr')
EEG = fix_and_load_brainvision(run_path,subj_id)

# Plot 6 channels with 1000 sample points
channels_to_plot = [0, 1, 2, 3, 5, 6]
n_samples = 1000
start_sample = 0

# Get data and time vector
data, times = EEG.get_data(return_times=True)
time_vector = times[start_sample:start_sample + n_samples]

# Create figure with 6 subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, ch_idx in enumerate(channels_to_plot):
    ch_name = EEG.ch_names[ch_idx]
    ch_data = data[ch_idx, start_sample:start_sample + n_samples]
    ch_data_uV = ch_data * 1e6  # Convert from V to µV

    axes[idx].plot(time_vector, ch_data_uV, linewidth=0.5)
    axes[idx].set_title(f'Channel {ch_idx}: {ch_name}')
    axes[idx].set_xlabel('Time (s)')
    axes[idx].set_ylabel('Amplitude (µV)')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


