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
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose=False)
    else:
        # read textvmrk
        with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Capture the old filename from the text
        match = re.search(r'(?im)^\s*DataFile\s*=\s*(.*)$', text)
        if match:
            old_filename = match.group(1)
            # Replace only the subject ID part
            new_filename = re.sub(r'sub-\d+', f'sub-{subj_id}', old_filename)
            
            text_fixed = re.sub(r'(?im)^\s*DataFile\s*=.*$',
                            f"DataFile={new_filename}",
                            text)
        # Capture the old filename from the text
        match = re.search(r'(?im)^\s*MarkerFile\s*=\s*(.*)$', text)
        if match:
            old_filename = match.group(1)
            # Replace only the subject ID part
            new_filename = re.sub(r'sub-\d+', f'sub-{subj_id}', old_filename)
            
            text_fixed = re.sub(r'(?im)^\s*MarkerFile\s*=.*$',
                            f"MarkerFile={new_filename}",
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
        raw = mne.io.read_raw_brainvision(tmp_path, preload=preload, verbose=False)
        # remove tmp file
        os.remove(tmp_path)
    return raw

def eeg_preproc_basic(EEG, is_bpfilter=True, bp_f_range=[0.1, 45],
                      is_reref=True, reref_ch=['tp9h','tp10h'],
                      is_ica_rmEye=True):
    if is_bpfilter:
        # band-pass filtering (all channels)
        EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks='all',verbose=False)
    if is_reref:
        # re-reference to the average of mastoid (EEG channels only)
        EEG.set_eeg_reference(ref_channels=reref_ch, ch_type='eeg',verbose=False)
    if is_ica_rmEye:
        # ICA on EEG channels only
        ica = mne.preprocessing.ICA(n_components=EEG.info.get_channel_types().count('eeg'),
                method='infomax', random_state=42,verbose=False)
        ica.fit(EEG, picks=['eeg'],verbose=False)
        #  remove potential eye components (if any) using EOG channles
        eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['hEOG','vEOG'], measure='correlation', verbose=False)
        ica.exclude = eog_inds
        EEG = ica.apply(EEG,verbose=False)

    return EEG

def tsv_to_events(event_file):
    #check if event_file exists
    if not os.path.exists(event_file):
        event_file = event_file.replace("run-0", "run-")
        if not os.path.exists(event_file):
            raise FileNotFoundError("Event.tsv not found.")
    events_df = pd.read_csv(event_file,sep='\t')
    event_ids = events_df["response_code"]
    event_labels_lookup = dict(city_incorrect=-1, city_correct=1,
                            mnt_incorrect=-2, mnt_correct=0,
                            city_incorrect_response=-11, city_correct_response=11,
                            mnt_incorrect_response=-12, mnt_correct_response=10)
    
    # create events array (onset, stim_channel_voltage, event_id)
    events_stim_onset = np.column_stack(((events_df["onset"]*EEG.info["sfreq"]).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        event_ids))
    events_response = np.column_stack((((events_df["onset"]+events_df["reaction_time"])*EEG.info["sfreq"]).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        event_ids+10*((event_ids>=0).astype(int)*2-1)))
    # stack together
    events = np.vstack([events_stim_onset,events_response])
    return events, event_labels_lookup

def epoch_by_select_event(EEG, events, select_event='mnt_correct',baseline_length=-0.2,epoch_reject_crit=dict(eeg=100e-6), is_detrend=1, event_duration=0.8):
    
    """
    The event duration varies for each trial. For convenience, I fixed it as 0.8 second for mnt_correct trials and 1.6 for city_correct trials.
    (Chi 10/22/2025)
    """
    event_labels_lookup = dict(city_incorrect=-1, city_correct=1,
                            mnt_incorrect=-2, mnt_correct=0,
                            city_incorrect_response=-11, city_correct_response=11,
                            mnt_incorrect_response=-12, mnt_correct_response=10)
    n_select_ev = np.sum(events[:,-1]==event_labels_lookup[select_event])
    print(f"# {select_event}/ # total = {n_select_ev}/{events.shape[0]} ({n_select_ev/events.shape[0]*100:.1f}%)")
    # pick only selected event
    events = events[events[:,-1]==event_labels_lookup[select_event]]
    
    # Check if we have any events
    if len(events) == 0:
        raise ValueError(f"No events found for {select_event}")
    
    # Filter out events that fall outside valid data range
    sfreq = EEG.info["sfreq"]
    n_samples = len(EEG.times)
    tmax = event_duration + baseline_length
    
    # Calculate required samples before and after event
    samples_before = int(np.abs(baseline_length) * sfreq)
    samples_after = int(tmax * sfreq)
    
    # Filter events that have enough data on both sides
    valid_mask = (events[:, 0] >= samples_before) & (events[:, 0] <= n_samples - samples_after)
    events_filtered = events[valid_mask]
    
    n_excluded = len(events) - len(events_filtered)
    if n_excluded > 0:
        print(f"Warning: {n_excluded} events excluded (outside valid data range)")
    
    if len(events_filtered) == 0:
        raise ValueError(f"No valid events remain after filtering (all {len(events)} events are outside the valid data range)")
    
    # epoch by event
    epochs = mne.Epochs(EEG, events=events_filtered,event_id={select_event:event_labels_lookup[select_event]},preload=True,
                        tmin=baseline_length, tmax=tmax,
                        reject=epoch_reject_crit, detrend=is_detrend, verbose=False
                        )
    epochs.drop_bad(verbose=False)

    if epoch_reject_crit is not None:
        print(f"# Epochs below PTP threshold ({epoch_reject_crit['eeg']*1e6} uV) = {len(epochs.selection)}")
    else:
        print(f"# Epochs (no rejection applied) = {len(epochs.selection)}")

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