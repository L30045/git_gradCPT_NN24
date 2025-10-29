#%% load library
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
from mne import events_from_annotations
import os
import glob
import re
import tempfile
import pandas as pd
from scipy.ndimage import uniform_filter1d
import copy

#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata/raw")
project_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24")
fig_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/plots/EEG")
data_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/processed_data")


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
        EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks=['eeg', 'tp9h', 'tp10h'],verbose=False)
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

def gen_EEG_event_tsv(subj_id, savepath=None):
    # setup savepath
    if savepath is None:
        savepath = os.path.join(data_save_path,f'sub-{subj_id}/eeg')
    gradcpt_path = os.path.join(data_path, f'sub-{subj_id}/gradCPT')
    # get all files with .mat ext in gradcpt_path
    files = [f for f in os.listdir(gradcpt_path) 
             if os.path.isfile(os.path.join(gradcpt_path, f)) 
             and f.endswith('.mat')]
    # get EEG path
    raw_EEG_path = os.path.join(data_path, f'sub-{subj_id}', 'eeg')
    filename_list = [os.path.basename(x) for x in glob.glob(os.path.join(raw_EEG_path,"*.vhdr"))]
    for fname in filename_list:
        # check if fname is a run session
        if "cpt" not in fname.lower():
            continue
        # get run id
        run_id = fname.lower().split("cpt")[-1][0]
        # get EEG trigger
        EEG = fix_and_load_brainvision(os.path.join(raw_EEG_path,fname),subj_id)
        eeg_trigger = EEG.get_data()[4]
        # load corresponding gradCPT
        f_cpt = files[[i for i, x in enumerate(files) if x.split('-0')[1][0]==run_id][0]]
        data_cpt = sp.io.loadmat(os.path.join(gradcpt_path,f_cpt))
        # gradcpt starttime
        starttime_cpt = data_cpt['starttime'][0][0]
        # find when EEG trigger is on (to 0)
        t_eeg_offset_sessions = EEG.times[np.where(eeg_trigger < np.min(eeg_trigger)+(np.max(eeg_trigger)-np.min(eeg_trigger))/2)[0][0]] # sec
        # event onset time
        t_onset = data_cpt['ttt'][:-1,0] - starttime_cpt + t_eeg_offset_sessions # exclude last event since it is a fade-out-only event.
        # check if event onset time exceed EEG recording time
        if t_onset[-1]>EEG.times[-1]:
            raise ValueError("Event onset time exceed EEG recording time.")
        # VTC
        react_time = data_cpt['response'][:-1,4] # sec
        meanRT = np.nanmean(react_time)
        stdRT = np.nanstd(react_time)
        vtc = copy.deepcopy(react_time)
        # fill in no reaction time trial with previous trial's reaction time
        non_zero_idx = np.where(vtc>0)[0]
        for rt_i in range(len(react_time)):
            if vtc[rt_i]==0:
                # assign previous reaction time
                vtc[rt_i] = vtc[non_zero_idx[np.where(non_zero_idx<rt_i)[0][-1]]]
        # calculate VTC
        vtc = np.abs((vtc-meanRT)/stdRT)
        # create DataFrame
        ev_df = pd.DataFrame(columns=[
            'onset',
            'duration', 
            'value',
            'trial_type',
            'exemplar',
            'reaction_time',
            'response_code',
            'VTC'
        ])
        ev_df['onset'] = t_onset
        ev_df['duration'] = np.diff(data_cpt['ttt'][:,0])
        ev_df['value'] = np.ones(t_onset.shape).astype(int) # fix amplitude
        ev_df['trial_type'] = ['city' if x==32 else 'mnt' for x in data_cpt['response'][:-1,1]]
        ev_df['exemplar'] = np.zeros(t_onset.shape).astype(int) # missing stimulus figure id
        ev_df['reaction_time'] = react_time
        ev_df['response_code'] = data_cpt['response'][:-1,6].astype(int) # press or not
        ev_df['VTC'] = vtc
        # save dataframe
        save_filename = os.path.join(savepath, f'sub-{subj_id}_task-gradCPT_run-0{run_id}_events.tsv')
        ev_df.to_csv(save_filename, sep='\t', index=False)

def tsv_to_events(event_file, sfreq):
    #check if event_file exists
    if not os.path.exists(event_file):
        event_file = event_file.replace("run-0", "run-")
        if not os.path.exists(event_file):
            raise FileNotFoundError("Event.tsv not found.")
    events_df = pd.read_csv(event_file,sep='\t')
    event_ids = events_df["response_code"].astype(int)
    event_labels_lookup = dict(city_incorrect=-1, city_correct=1,
                            mnt_incorrect=-2, mnt_correct=0,
                            city_incorrect_response=-11, city_correct_response=11,
                            mnt_incorrect_response=-12, mnt_correct_response=10)

    # create events array (onset, stim_channel_voltage, event_id)
    events_stim_onset = np.column_stack(((events_df["onset"]*sfreq).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        event_ids))
    events_response = np.column_stack((((events_df["onset"]+events_df["reaction_time"])*sfreq).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        event_ids+10*((event_ids>=0).astype(int)*2-1)))
    # stack together
    events = np.vstack([events_stim_onset,events_response])
    # extract VTC
    vtc_list = np.tile(events_df["VTC"],2)
    # extract reaction time
    reaction_time = np.concatenate([events_df["reaction_time"].values, -1*events_df["reaction_time"].values])
    
    return events, event_labels_lookup, vtc_list, reaction_time

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
    print("="*20)
    print(f"# {select_event}/ # total = {n_select_ev}/{int((events.shape[0]/2))} ({n_select_ev/(events.shape[0]/2)*100:.1f}%)")
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
    print("="*20)
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

def plt_ERPImage(time_vector, plt_epoch, sort_idx=None, smooth_window_size=10, clim=[-10*1e-6, 10*1e-6], title_txt=None, ref_onset=None):
    """
    Plot ERPImage for a 2D matrix (trial, times) and the average ERP across all trials.

    Input:
        time_vector: 1d array. Time associates with the ERP.
        plt_epoch: 2d matrix (trial, times). Trials to visualize.
        sort_idx: 1d array. Define how to sort trials.
        smooth_window_size: smooth out the ERPImage along trials. ONLY FOR VISUALIZATION.
        clim: color limit in ERPImage.
        title_txt: title.
        ref_onset: 1d array. The time series relative to the onset (e.g. stim vs response)
    
    Return:
        fig: fig object.
    """
    # sort along y-axis according to sort_idx
    if sort_idx is not None:
        s_idx = np.argsort(sort_idx)
        plt_epoch = plt_epoch[s_idx]
        sort_idx = sort_idx[s_idx]
        plt_vtc_smooth = uniform_filter1d(sort_idx, size=smooth_window_size, mode='nearest')
        if ref_onset is not None:
            ref_onset = ref_onset[s_idx]
            ref_onset_smooth = uniform_filter1d(ref_onset, size=smooth_window_size, mode='nearest')
    elif ref_onset is not None:
        ref_onset_smooth = ref_onset

    # Smooth along y-axis (trials) using sliding window
    plt_epoch_smooth = uniform_filter1d(plt_epoch, size=smooth_window_size, axis=0, mode='nearest')
        
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1], sharex=True, constrained_layout=True)

    # Top subplot: ERP Image
    if sort_idx is not None:
        im = axes[0].imshow(plt_epoch_smooth, aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[time_vector[0], time_vector[-1], plt_vtc_smooth[0], plt_vtc_smooth[-1]],
                            vmin=clim[0], vmax=clim[1]
                            )
    else:
        im = axes[0].imshow(plt_epoch_smooth, aspect='auto', origin='lower', cmap='RdBu_r',
                            extent=[time_vector[0], time_vector[-1], 0, plt_epoch_smooth.shape[0]],
                            vmin=clim[0], vmax=clim[1]
                            )
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Onset')

    # Plot reference onset line if provided
    if ref_onset is not None:
        if sort_idx is not None:
            axes[0].plot(ref_onset_smooth, plt_vtc_smooth, color='black', linewidth=2, label='Ref Onset')
        else:
            y_coords = np.arange(len(ref_onset_smooth))
            axes[0].plot(ref_onset_smooth, y_coords, color='black', linewidth=2, label='Ref Onset')

    plt.colorbar(im, ax=axes[0], label='Amplitude (µV)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('VTC')
    axes[0].set_title(f'ERP Image - {title_txt}')
    axes[0].legend(loc='upper right')

    # Bottom subplot: Average of all trials
    avg_erp = np.mean(plt_epoch, axis=0)
    sem_erp = np.std(plt_epoch, axis=0) / np.sqrt(plt_epoch.shape[0])
    axes[1].plot(time_vector, avg_erp, linewidth=2, color='blue', label='Mean')
    axes[1].fill_between(time_vector, avg_erp - 2*sem_erp, avg_erp + 2*sem_erp,
                        alpha=0.3, color='blue', label='±2 SEM')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Stimulus onset')
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (V)')
    axes[1].set_title(f'Average ERP - {title_txt}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.show()
    return fig

