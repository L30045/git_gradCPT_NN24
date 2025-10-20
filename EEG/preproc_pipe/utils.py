#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import re
import tempfile


#%% utils function
def fix_and_load_brainvision(vhdr_path,
                             subj_id,
                             preload=True):
    """
    Load EEG into mne Raw object using vhdr.
    This function correct subject ID in vhdr by creating a temparary vhdr file.
    """
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

def eeg_preproc_basic(is_bpfilter=True, bp_f_range=[0.1, 45],
                      is_reref=True, reref_ch=['tp9h','tp10h'],
                      is_ica_rmEye=True):
    if is_bpfilter:
        # band-pass filtering (all channels)
        EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks='all')
    if is_reref:
        # re-reference to the average of mastoid (EEG channels only)
        EEG.set_eeg_reference(ref_channels=reref_ch, ch_type='eeg')
    if is_ica_rmEye:
        # ICA on EEG channels only
        ica = mne.preprocessing.ICA(n_components=EEG.info.get_channel_types().count('eeg'),
                method='infomax', random_state=42)
        ica.fit(EEG, picks=['eeg'])
        #  remove potential eye components (if any) using EOG channles
        eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['hEOG','vEOG'], measure='correlation')
        ica.exclude = eog_inds
        EEG = ica.apply(EEG)
        
    return EEG
