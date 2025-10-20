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

#%% parameter setting
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
is_ica_rmEye = True

#%% Load and preprocessing
# load run 1 as testing
run_id = 1
run1_path = os.path.join(raw_EEG_path, f'sub-{subj_id}_gradCPT{run_id}.vhdr')
EEG = fix_and_load_brainvision(run1_path,subj_id)
EEG = eeg_preproc_basic(is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                      is_reref=is_reref, reref_ch=reref_ch,
                      is_ica_rmEye=is_ica_rmEye)

# %%

