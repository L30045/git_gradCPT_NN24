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


#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata")
subj_id_array = [670, 671, 673, 695]

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

#%% epoch each subject
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
            exclude_run_dict[subj_id].append(run_id)
            epochs = epoch_by_select_event(EEG, event_file, select_event=select_event,baseline_length=baseline_length,
                                                            epoch_reject_crit=None, is_detrend=1)
        # save epochs
        subj_epoch_dict[f"sub-{subj_id}"].append(epochs)
    
#%% Visualizing
# sanity check with one subject
plt_epoch = subj_epoch_array[0]
vis_ch = 'cz'
plt_center, plt_shade, plt_time = plot_ch_erp(plt_epoch, vis_ch, is_return_data=True)

#%% check excluded EEG
# run_path = os.path.join(raw_EEG_path, f'{exclude_run_array[0]}.vhdr')
# EEG = fix_and_load_brainvision(run_path,subj_id)
# EEG = eeg_preproc_basic(EEG, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
#                     is_reref=is_reref, reref_ch=reref_ch,
#                     is_ica_rmEye=is_ica_rmEye)
event_file = os.path.join(data_path,os.pardir,f"sub-{subj_id}","nirs",
                                f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
epochs = epoch_by_select_event(EEG, event_file, select_event=select_event,baseline_length=baseline_length,is_detrend=1)

# visualize cross-subjects results



#%%



