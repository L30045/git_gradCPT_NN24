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


#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata")
subj_id_array = [670, 671, 673, 695]
fig_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/plots/EEG")
data_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/processed_data")

#%% parameter setting
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

#%% epoch each subject
if not os.path.exists(os.path.join(data_save_path, f'epochs_{select_event}.pkl')):
    subj_epoch_dict = dict()
    exclude_run_dict = dict()
    for subj_id in tqdm(subj_id_array):
        raw_EEG_path = os.path.join(data_path, 'raw', f'sub-{subj_id}', 'eeg')
        subj_epoch_dict[f"sub-{subj_id}"] = []
        exclude_run_dict[f"sub-{subj_id}"] = []
        for run_id in np.arange(1,4):
            # load run 1 as testing
            run_path = os.path.join(raw_EEG_path, f'sub-{subj_id}_gradCPT{run_id}.vhdr')
            EEG = fix_and_load_brainvision(run_path,subj_id)
            EEG = eeg_preproc_basic(EEG, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                                is_reref=is_reref, reref_ch=reref_ch,
                                is_ica_rmEye=is_ica_rmEye)

            # Epoching
            # load corresponding event file
            event_file = os.path.join(data_path,os.pardir,f"sub-{subj_id}","nirs",
                                    f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
            try:    
                epochs = epoch_by_select_event(EEG, event_file, select_event=select_event,baseline_length=baseline_length,
                                                                epoch_reject_crit=dict(eeg=100e-6), is_detrend=1)
            except:
                print("="*20)
                print(f"No clean trial found in sub-{subj_id}_gradCPT{run_id}.")    
                print("="*20)
                exclude_run_dict[f"sub-{subj_id}"].append(run_id)
                epochs = epoch_by_select_event(EEG, event_file, select_event=select_event,baseline_length=baseline_length,
                                                                epoch_reject_crit=None, is_detrend=1)
            # save epochs
            subj_epoch_dict[f"sub-{subj_id}"].append(epochs)

    # save processed data for future use
    save_data = dict(
        subj_epoch_dict=subj_epoch_dict,
        exclude_run_dict=exclude_run_dict,
        preproc_params=preproc_params
    )
    with open(os.path.join(data_save_path, f'epochs_{select_event}.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
else:
    with open(os.path.join(data_save_path, f'epochs_{select_event}.pkl'), 'rb') as f:
        tmp_data = pickle.load(f)
        subj_epoch_dict = tmp_data['subj_epoch_dict']
        exclude_run_dict = tmp_data['exclude_run_dict']
        preproc_params = tmp_data['preproc_params']

#%% Combine epochs for each subject
# Unpack preprocessing parameters
is_bpfilter = preproc_params['is_bpfilter']
bp_f_range = preproc_params['bp_f_range']
is_reref = preproc_params['is_reref']
reref_ch = preproc_params['reref_ch']
is_ica_rmEye = preproc_params['is_ica_rmEye']
select_event = preproc_params['select_event']
baseline_length = preproc_params['baseline_length']
epoch_reject_crit = preproc_params['epoch_reject_crit']

# Concatenate the list of epochs into one epoch object for each subject
subj_epoch_array = []
for subj_key in subj_epoch_dict.keys():
    if len(subj_epoch_dict[subj_key]) > 0:
        # Concatenate all epochs for this subject into a single Epochs object
        combined_epochs = mne.concatenate_epochs(subj_epoch_dict[subj_key])
        subj_epoch_array.append(combined_epochs)

#%% Visualizing
# sanity check with one subject
# plt_epoch = subj_epoch_array[3]
# vis_ch = 'cz'
# plt_center, plt_shade, plt_time = plot_ch_erp(plt_epoch, vis_ch, is_return_data=True)

#%% cross-subjects results
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
    save_filename = f'cross_subject_ERP_{vis_ch[ch_i]}_mean_2SEM.png'
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


