#%% load library
import numpy as np
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
from mne import events_from_annotations
import os
import re
import tempfile
import pandas as pd


#%% utils function
def fix_and_load_brainvision(vhdr_path,
                             subj_id,
                             preload=True):
    """
    Load EEG into mne Raw object using vhdr.
    This function correct subject ID in vhdr by creating a temparary vhdr file.
    """
    # check if subj_id == 695
    if subj_id==695:
        vhdr_path = os.path.join(os.path.dirname(vhdr_path),'G'+os.path.basename(vhdr_path).split('_')[-1][1:])
        # load with MNE
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload)
    else:
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

def eeg_preproc_basic(EEG, is_bpfilter=True, bp_f_range=[0.1, 45],
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


def epoch_by_select_event(EEG, event_file, select_event='mnt_correct',baseline_length=-0.2,epoch_reject_crit=dict(eeg=100e-6), is_detrend=1):
    #check if event_file exists
    if not os.path.exists(event_file):
        event_file = event_file.replace("run-0", "run-")
        if not os.path.exists(event_file):
            raise FileNotFoundError("Event.tsv not found.")
    events_df = pd.read_csv(event_file,sep='\t')
    event_duration = float(events_df["duration"].values[0])
    is_event_correct = (events_df["value"].values).astype(int)
    is_event_mnt = ((events_df["trial_type"]=="mnt").values).astype(int)
    event_ids = is_event_mnt*2+is_event_correct
    event_labels_lookup = dict(city_incorrect=0, city_correct=1,
                            mnt_incorrect=2, mnt_correct=3)
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
                        reject=epoch_reject_crit
                        )
    epochs.drop_bad()
    if is_detrend:
        epochs = epochs.detrend(1)
    print(f"# Epochs below PTP threshold ({epoch_reject_crit['eeg']*1e6} uV) = {len(epochs.selection)}")

    return epochs

def plot_ch_erp(epochs, vis_ch, center_method=np.mean, shaded_method=lambda x: np.std(x,axis=0)/np.sqrt(x.shape[0]), is_return_data=False, is_plot=True):
    epoch_data = epochs.get_data()[:,epochs.info["ch_names"].index(vis_ch),:]
    plt_center = center_method(epoch_data,axis=0)
    center_label = f'Mean (# of trial = {(epoch_data.shape[0])})'
    shade_method = np.std(epoch_data,axis=0)/ np.sqrt(epoch_data.shape[0])
    plt_shade = [plt_center-2*shade_method, plt_center+2*shade_method]
    shaded_label = '+/- 2 SEM'
    # plt_shade = np.quantile(epoch_data, q=[0.25,0.75], axis=0)
    # shaded_label = f'Quantile (25/75)'
    plt_time = epochs.times
    if is_plot:
        fig = plt.figure()
        plt.plot(plt_time, plt_center, color='b', label=center_label)
        plt.fill_between(plt_time, plt_shade[0], plt_shade[1], color='b', alpha=0.3, label=shaded_label)
        plt.axvline(0,color='k',linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    if is_return_data:
        return plt_center, plt_shade, plt_time