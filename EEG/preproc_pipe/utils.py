#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import re
import tempfile


#%%
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