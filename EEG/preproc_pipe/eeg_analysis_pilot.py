"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import pandas as pd
from utils import *


#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata")
subj_id = 670
raw_EEG_path = os.path.join(data_path, 'raw', f'sub-{subj_id}', 'eeg')

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

#%% Load and preprocessing
# load run 1 as testing
run_id = 1
run1_path = os.path.join(raw_EEG_path, f'sub-{subj_id}_gradCPT{run_id}.vhdr')
EEG = fix_and_load_brainvision(run1_path,subj_id)
EEG = eeg_preproc_basic(EEG, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                      is_reref=is_reref, reref_ch=reref_ch,
                      is_ica_rmEye=is_ica_rmEye)

# %% Epoching
# load corresponding event file
event_file = os.path.join(data_path,os.pardir,f"sub-{subj_id}","nirs",
                          f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
events_df = pd.read_csv(event_file,sep='\t')
event_duration = float(events_df["duration"].values[0])
is_event_correct = (events_df["value"].values).astype(int)
is_event_mnt = ((events_df["trial_type"]=="mnt").values).astype(int)
event_ids = is_event_mnt*2+is_event_correct
event_labels_lookup = dict(city_incorrect=0, city_correct=1,
                           mnt_incorrect=2, mnt_correct=3)
# select event to epoch

# create events array (onset, stim_channel_voltage, event_id)
events = np.column_stack(((events_df["onset"]*EEG.info["sfreq"]).astype(int),
                    np.zeros(len(events_df), dtype=int),
                    event_ids))
n_select_ev = np.sum(events[:,-1]==event_labels_lookup[select_event])
print(f"# {select_event}/ # total = {n_select_ev}/{events.shape[0]} ({n_select_ev/events.shape[0]*100:.1f}%)")
# pick only selected event
events = events[events[:,-1]==event_labels_lookup[select_event]]
# epoch by event
epochs = mne.Epochs(EEG, events=events,event_id={select_event:event_labels_lookup[select_event]},preload=True,
                    tmin=baseline_length, tmax=event_duration+baseline_length,
                    reject=epoch_reject_crit,
                    detrend=is_detrend, 
                    )
print(f"# Epochs below PTP threshold ({epoch_reject_crit['eeg']*1e6} uV) = {len(epochs.selection)}")

#%% Visualizing
vis_ch = 'fz'
epoch_data = epochs.get_data()[:,epochs.info["ch_names"].index(vis_ch),:]
plt_center = np.mean(epoch_data,axis=0)
plt_shade = np.std(epoch_data,axis=0)/ np.sqrt(epoch_data.shape[0])
plt_time = epochs.times
plt.figure()
plt.plot(plt_time, plt_center, color='b')
plt.fill_between(plt_time, plt_center - 2*plt_shade, plt_center + 2*plt_shade, color='b', alpha=0.3)
plt.axvline(0,color='k',linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%



