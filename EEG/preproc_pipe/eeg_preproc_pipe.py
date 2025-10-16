"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import re
import tempfile


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


#%% utils functions
def fix_and_load_brainvision(vhdr_path,
                             subj_id,
                             preload=True):
    # assign correct eeg and vmrk filename
    correct_eeg_filename = f'sub-{subj_id}_gradCPT1.eeg'
    correct_vmrk_filename = f'sub-{subj_id}_gradCPT1.vmrk'

    # read textvmrk
    with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Replace DataFile= and MarkerFile= (case-insensitive)
    text_fixed = re.sub(r'(?im)^\s*DataFile\s*=.*$',
                        f"DataFile={correct_eeg_filename}",
                        text)
    text_fixed = re.sub(r'(?im)^\s*MarkerFile\s*=.*$',
                        f"MarkerFile={correct_vmrk_filename}",
                        text_fixed)

    # write to a temp file in same directory (so relative paths inside vhdr still work)
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                        suffix='.vhdr',
                                        dir=os.path.dirname(vhdr_path))
    tmp_path = os.path.abspath(tmp.name)
    tmp.close()
    with open(tmp_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text_fixed)

    # now load with MNE
    raw = mne.io.read_raw_brainvision(tmp_path, preload=preload)

    # remove tmp file
    os.remove(tmp_path)

    return raw
