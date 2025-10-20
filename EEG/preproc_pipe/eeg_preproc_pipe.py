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


#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata")
subj_id = 671
raw_EEG_path = os.path.join(data_path, 'raw', f'sub-{subj_id}', 'eeg')
# load run 1 as testing
run_id = 1
run1_path = os.path.join(raw_EEG_path, f'sub-{subj_id}_gradCPT{run_id}.vhdr')
EEG = fix_and_load_brainvision(run1_path,subj_id)

#%% parameter setting
bp_f_range = [0.1, 45] #band pass filter range (Hz)


#%% preprocessing
# band-pass filtering (all channels)
EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks='all')
# re-reference to the average of mastoid
EEG.set_eeg_reference(ref_channels=['tp9h', 'tp10h'])
# ICA and remove potential eye components
ica = mne.preprocessing.ICA(n_components=EEG.info.get_channel_types().count('eeg'),
          method='infomax', random_state=42)
ica.fit(EEG)
eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['hEOG','vEOG'], measure='correlation')
# Remove them
ica.exclude = eog_inds
EEG_clean = ica.apply(EEG.copy())

# %%

