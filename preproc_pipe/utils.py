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
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import copy
from spectral_connectivity import Multitaper, Connectivity
from tqdm import tqdm

#%% path setting
# Add the parent directory and src directory to sys.path
git_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/sourcedata/raw")
project_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24")
fig_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/plots/EEG")
data_save_path = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg")


#%% utils function
#%% Loading and generating events
def fix_and_load_brainvision(vhdr_path,
                             preload=True):
    """
    Load EEG into mne Raw object using vhdr.
    This function correct subject ID in vhdr by creating a temparary vhdr file.
    """
    # read textvmrk
    with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # filename
    filename = os.path.basename(vhdr_path).split('.')[0]
    # Capture the old filename from the text
    match = re.search(r'(?im)^\s*DataFile\s*=\s*(.*)$', text)
    if match:
        # old_filename = match.group(1)
        # Replace only the subject ID part
        # new_filename = re.sub(r'sub-\d+', f'sub-{subj_id}', old_filename)
        new_filename = f"{filename}.eeg"
        
        text_fixed = re.sub(r'(?im)^\s*DataFile\s*=.*$',
                        f"DataFile={new_filename}",
                        text)
    # Capture the old filename from the text
    match = re.search(r'(?im)^\s*MarkerFile\s*=\s*(.*)$', text)
    if match:
        # old_filename = match.group(1)
        # Replace only the subject ID part
        # new_filename = re.sub(r'sub-\d+', f'sub-{subj_id}', old_filename)
        new_filename = f"{filename}.vmrk"

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

def check_flat_channels(EEG):
    eeg_data = EEG.get_data(picks='eeg')
    eeg_chs = np.array([x["ch_name"] for x in EEG.info["chs"] if x["kind"]==2])
    flat_ch_idx = []
    # check flat channels
    for ch_i in range(eeg_data.shape[0]):
        if np.mean(eeg_data[ch_i]-np.mean(eeg_data[ch_i]))==0:
            print(f"Warning: flat channel detected. ({eeg_chs[ch_i]})")
            flat_ch_idx.append(eeg_chs[ch_i])
    return flat_ch_idx

def check_abnormal_var_channels(EEG, thres_std=3):
    eeg_data = EEG.get_data(picks='eeg')
    eeg_chs = np.array([x["ch_name"] for x in EEG.info["chs"] if x["kind"]==2])
    # calculate variance for each channels and transform to z-score
    eeg_var = np.var(eeg_data,axis=1)
    eeg_var_z = (eeg_var-np.mean(eeg_var))/np.std(eeg_var)
    if np.any(abs(eeg_var_z))>thres_std:
        print(f"Warning: channels with abnormal variance: {eeg_chs[abs(eeg_var_z)>thres_std]}")
    return eeg_chs[abs(eeg_var_z)>thres_std]

def eeg_preproc_basic(EEG, is_bpfilter=True, bp_f_range=[0.1, 45], is_check_flat=True, is_check_ch_var=True,
                      is_reref=True, reref_ch=None,
                      is_ica_rmEye=True):
    rm_ch_list = []
    eeg_trigger = EEG.get_data()[4]
    # Check if Trigger is pressed before and after the experiment. (The duration of two triggers should be longer than 6 mins as experiment design.)
    thres_trigger = (np.max(eeg_trigger)-np.min(eeg_trigger))/2+np.min(eeg_trigger)
    eeg_duration = np.max(np.diff(np.where(eeg_trigger<thres_trigger)[0]))/EEG.info["sfreq"]/60 # mins
    if eeg_duration < 6:
        print("="*20)
        print("Valid recording length is shorter than 6 mins. (Missing triggers or not enough recorrding length.)")
        print("="*20)
    if is_bpfilter:
        # band-pass filtering (all channels)
        EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks='all',verbose=False)
    # check flat channels
    if is_check_flat:
        rm_ch_list.extend(check_flat_channels(EEG))
    # check variance
    if is_check_ch_var:
        rm_ch_list.extend(check_abnormal_var_channels(EEG))
    # drop bad channels
    if len(rm_ch_list)>0:
        EEG.drop_channels(rm_ch_list)
    if is_reref:
        if reref_ch:
            # re-reference to the average of mastoid (EEG channels only)
            EEG.set_eeg_reference(ref_channels=reref_ch, ch_type='eeg',verbose=False)
        else:
            # re-reference to common average
            EEG.set_eeg_reference(ref_channels='average', ch_type='eeg',verbose=False)
    # rm eye-related ICA
    if is_ica_rmEye:
        # ICA on EEG channels only
        ica = mne.preprocessing.ICA(n_components=EEG.info.get_channel_types().count('eeg'),
                method='infomax', random_state=42,verbose=False)
        ica.fit(EEG, picks=['eeg'],verbose=False)
        #  remove potential eye components (if any) using EOG channles
        eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['hEOG','vEOG'], measure='correlation', verbose=False)
        ica.exclude = eog_inds
        EEG = ica.apply(EEG,verbose=False)
    # Restore original Trigger channel data
    EEG._data[4] = eeg_trigger
    return EEG, rm_ch_list

def gen_EEG_event_tsv(subj_id, savepath=None):
    # setup savepath
    if savepath is None:
        savepath = os.path.join(data_save_path,f'sub-{subj_id}')
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
        run_id = fname.lower().split("run-0")[-1][0]
        # get EEG trigger
        EEG = fix_and_load_brainvision(os.path.join(raw_EEG_path,fname))
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
        meanRT = np.nanmean(react_time[react_time > 0])
        stdRT = np.nanstd(react_time[react_time > 0])
        vtc = copy.deepcopy(react_time)
        # fill in no reaction time trial with linear interpolation
        non_zero_idx = np.where(vtc>0)[0]
        zero_idx = np.where(vtc==0)[0]
        if len(zero_idx) > 0 and len(non_zero_idx) > 0:
            # use linear interpolation to fill missing values
            vtc[zero_idx] = np.interp(zero_idx, non_zero_idx, vtc[non_zero_idx])
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
        ev_df['trial_type'] = ['city' if x==2 else 'mnt' for x in data_cpt['response'][:-1,0]]
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
    event_labels_lookup = dict(city_incorrect=-2, city_correct=1,
                            mnt_incorrect=-1, mnt_correct=0,
                            city_incorrect_response=-12, city_correct_response=11,
                            mnt_incorrect_response=-11, mnt_correct_response=10)
    # smooth VTC using Gaussian window (20 trials)
    smoothed_vtc = gaussian_filter1d(events_df["VTC"], sigma=2.5, truncate=4) # kernel size = round(truncate*sigma)*2+1
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
    vtc_list = np.tile(smoothed_vtc,2)
    # extract reaction time
    reaction_time = np.concatenate([events_df["reaction_time"].values, -1*events_df["reaction_time"].values])
    
    return events, event_labels_lookup, vtc_list, reaction_time


#%% epoching, ERPImage, and ERSP using multitaper
def epoch_by_select_event(EEG, events, event_labels_lookup, select_event='mnt_correct',baseline_length=-0.2,epoch_reject_crit=dict(eeg=100e-6), is_detrend=1, event_duration=0.8, verbose=True):
    
    """
    The event duration varies for each trial. For convenience, I fixed it as 0.8 second for mnt_correct trials and 1.6 for city_correct trials.
    (Chi 10/22/2025)
    """
    n_select_ev = np.sum(events[:,-1]==event_labels_lookup[select_event])
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
    
    #TODO: Check for flat channel. Current epoch rejection method only remove epochs with large amplitude.
    # =========================


    # =========================

    # epoch by event
    epochs = mne.Epochs(EEG, events=events_filtered,event_id={select_event:event_labels_lookup[select_event]},preload=True,
                        tmin=baseline_length, tmax=tmax,
                        reject=epoch_reject_crit, detrend=is_detrend, verbose=False
                        )
    len_ori_epoch = len(epochs)
    epochs.drop_bad(verbose=False)
    len_after_drop_epoch = len(epochs)

    if verbose:
        print("="*20)
        print(f"# {select_event}/ # total = {n_select_ev}/{int((events.shape[0]/2))} ({n_select_ev/(events.shape[0]/2)*100:.1f}%)")
        if epoch_reject_crit is not None:
            print(f"# Epochs below PTP threshold ({epoch_reject_crit['eeg']*1e6} uV) = {len(epochs.selection)}/{len(epochs.drop_log)}")
        else:
            print(f"# Epochs (no rejection applied) = {len(epochs.selection)}/{len(epochs.drop_log)}")
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

# visualize multitaper result
def plt_multitaper(plt_epoch,
                    time_halfbandwidth_product=3,
                    time_window_duration=1,
                    time_window_step=None,
                    expectation_type="trials_tapers",
                    ratio_to=None,
                    is_plot=True,
                    vis_f_range=[0, 50]):
    if not time_window_step:
        time_window_step = time_window_duration
    time_vector = plt_epoch.times
    # reshpae epoch data for multitaper
    plt_epoch_data = np.expand_dims(np.squeeze(plt_epoch.get_data()).T,axis=-1)
    # create multitaper
    multitaper = Multitaper(
        plt_epoch_data,
        sampling_frequency=plt_epoch.info["sfreq"],
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
        detrend_type='linear'
    )

    # using connectivity
    connectivity = Connectivity.from_multitaper(multitaper, expectation_type=expectation_type)
    # get time vector from multitaper and shift it by onset time
    multitaper_time = multitaper.time + time_vector[0]
    if isinstance(ratio_to, str) and ratio_to=='baseline':
        # find onset time (time=0)
        onset_idx = np.where(multitaper_time>=0)[0][0]
        # calculate baseline power
        power_baseline = np.mean(np.squeeze(connectivity.power())[:onset_idx,:],axis=0)
        log_power_baseline = np.log10(power_baseline)
    # calculate log power
    log_power = np.log10(np.squeeze(connectivity.power()))
    plt_power = copy.deepcopy(log_power)
    
    # transform log power to ratio. (power over ratio_to)
    avg_ref = None
    if ratio_to is not None:
        if isinstance(ratio_to, str) and ratio_to=='baseline':
            plt_power = log_power - log_power_baseline
        elif isinstance(ratio_to, np.ndarray):
            log_ratio_to = np.log10(ratio_to)
            plt_power = log_power - log_ratio_to
        elif isinstance(ratio_to, mne.EpochsArray):
            (log_power_ref,_,_) = plt_multitaper(ratio_to,
                                            time_halfbandwidth_product=time_halfbandwidth_product,
                                            time_window_duration=time_window_duration,
                                            time_window_step=time_window_step,
                                            is_plot=False)
            plt_power = log_power - log_power_ref
            avg_ref = np.mean(ratio_to.get_data(), axis=0).squeeze()
            

    # visualization
    if is_plot:
        vis_mask = (connectivity.frequencies>=vis_f_range[0])&(connectivity.frequencies<=vis_f_range[1])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True, constrained_layout=True)

        # Plot log power spectrogram
        extent = [multitaper_time[0], multitaper_time[-1], 0, np.sum(vis_mask)]
        vmax = np.abs(plt_power).max()
        im = ax1.imshow(plt_power[:,vis_mask].T, aspect='auto', origin='lower', cmap='RdBu_r', extent=extent, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax1, label='Log Power')
        ax1.set_ylabel('Frequency')
        ax1.set_yticks(np.arange(np.sum(vis_mask)))
        ax1.set_yticklabels(connectivity.frequencies[vis_mask])
        ax1.set_title(f'ERSP')
        ax1.axvline(0, color='white', linestyle='--', linewidth=1)

        # Plot average trial - trim to match multitaper time range
        avg_trial_full = np.mean(plt_epoch.get_data(), axis=0).squeeze()
        # Find indices in time_vector that match multitaper_time range
        time_mask = (time_vector >= multitaper_time[0]) & (time_vector <= multitaper_time[-1])
        avg_trial = avg_trial_full[time_mask]
        trimmed_time_vector = time_vector[time_mask]
        ax2.plot(trimmed_time_vector, avg_trial, 'k', linewidth=1.5, label="Target")
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (V)')
        ax2.set_title('Average Trial')
        ax2.grid(True, alpha=0.3)
        if avg_ref is not None:
            # Find indices in time_vector that match multitaper_time range
            avg_ref = avg_ref[time_mask]
            ax2.plot(trimmed_time_vector, avg_ref, 'r', linewidth=1.5, label="Ref")
        ax2.legend()
        plt.show()

    return (log_power,multitaper,connectivity)

#%% load EEG as epoch 
def load_epoch_dict(subj_id_array, preproc_params):
    subj_EEG_dict = dict()
    rm_ch_dict = dict()
    """
    subj_EEG_dic: dictionary for storing subject EEG. 
                    subj_EEG_dict["sub-{subj_id}"]["gradcpt{run_id}"]
                    subj_EEG_dict["sub-{subj_id}"]["rest{run_id}"]
    rm_ch_dict: dictionary for storing the name of removed channels                
                    rm_ch_dict["sub-{subj_id}"]["gradcpt{run_id}"]
                    rm_ch_dict["sub-{subj_id}"]["rest{run_id}"]
    """
    for subj_id in tqdm(subj_id_array):
        print(f"preprocessing sub-{subj_id}")
        single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
        subj_EEG_dict[f"sub-{subj_id}"] = single_subj_EEG_dict
        rm_ch_dict[f"sub-{subj_id}"] = single_subj_rm_ch_dict
        
    # Epoch data
    subj_epoch_dict = dict()
    subj_vtc_dict = dict()
    subj_react_dict = dict()
    """
    subj_epoch_dict: dictionary for storing subject Epoch.
                        subj_epoch_dict["sub-xxx"]["run0x"][Events]
                        Events include:
                            'city_incorrect': incorrect city trials, time-lock to stimulus-onset (first frame)
                            'city_correct': correct city trials, time-lock to stimulus-onset (first frame)
                            'mnt_incorrect': incorrect mountain trials, time-lock to stimulus-onset (first frame)
                            'mnt_correct': correct mountain trials, time-lock to stimulus-onset (first frame)
                            'city_incorrect_response': incorrect city trials, time-lock to stimulus-onset (first frame)
                            'city_correct_response': correct city trials, time-lock to response (spacebar press)
                            'mnt_incorrect_response': incorrect mountain trials, time-lock to response (spacebar press)
                            'mnt_correct_response': correct mountain trials, time-lock to stimulus-onset (first frame)
    """
    # for each subject
    for key_name in tqdm(subj_EEG_dict.keys()):
        subj_id = int(key_name.split('-')[-1])
        print(f"Epoching {key_name}")
        single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(key_name, subj_EEG_dict[key_name])
        # save epochs
        subj_epoch_dict[key_name] = single_subj_epoch_dict
        subj_vtc_dict[key_name] = single_subj_vtc_dict
        subj_react_dict[key_name] = single_subj_react_dict

    # combine runs for each subject
    combine_epoch_dict = dict()
    combine_vtc_dict = dict()
    combine_react_dict = dict()
    in_out_zone_dict = dict()
    """
    combine_epoch_dict: dictionary for combined epochs from each run for subject.
                        combine_epoch_dict["select_event"]["ch"]: list of epoch of selected event and channel. (length equals to number of subjects)
    """
    # get median of the vtc for each subject
    subj_thres_vtc = {subj_id: np.median(np.concatenate([subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]
                                            for run_id in range(1, 4)
                                            for event in event_labels_lookup.keys()
                                            if len(subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]) > 0]))
                    for subj_id in subj_vtc_dict.keys()}
    for select_event in event_labels_lookup.keys():
        epoch_dict = dict()
        vtc_dict = dict()
        react_dict = dict()
        ch_in_out_zone_dict = dict()
        # initialize epoch_dict
        for ch in ch_names:
            epoch_dict[ch] = []
            vtc_dict[ch] = []
            react_dict[ch] = []
            ch_in_out_zone_dict[ch] = []
        for subj_id in subj_epoch_dict.keys():
            tmp_epoch_list = []
            tmp_vtc_list = []
            tmp_react_list = []
            tmp_in_out_zone_list = []
            for run_id in np.arange(1,4):
                loc_e = subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event]
                loc_v = subj_vtc_dict[subj_id][f"run{run_id:02d}"][select_event]
                loc_r = subj_react_dict[subj_id][f"run{run_id:02d}"][select_event]
                if len(loc_e)>0:
                    tmp_epoch_list.append(loc_e)
                    tmp_vtc_list.append(loc_v)
                    tmp_react_list.append(loc_r)
                    tmp_in_out_zone_list.append(loc_v<subj_thres_vtc[subj_id])
            if len(tmp_epoch_list)>0:
                # for each channel, create an epoch
                for ch in ch_names:
                    ch_picked_epoch = [x.copy().pick(ch) for x in tmp_epoch_list if ch in x.ch_names]
                    epoch_dict[ch].append(mne.concatenate_epochs(ch_picked_epoch,verbose=False))
                    vtc_dict[ch].append(np.concatenate([x for x,y in zip(tmp_vtc_list,tmp_epoch_list) if ch in y.ch_names]))
                    react_dict[ch].append(np.concatenate([x for x,y in zip(tmp_react_list,tmp_epoch_list) if ch in y.ch_names]))
                    ch_in_out_zone_dict[ch].append(np.concatenate([x for x,y in zip(tmp_in_out_zone_list,tmp_epoch_list) if ch in y.ch_names]))
        combine_epoch_dict[select_event] = epoch_dict
        combine_vtc_dict[select_event] = vtc_dict
        combine_react_dict[select_event] = react_dict
        in_out_zone_dict[select_event] = ch_in_out_zone_dict

    return combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict, (subj_EEG_dict, subj_epoch_dict, subj_vtc_dict, subj_react_dict)

def remove_subject_by_nb_epochs_preserved(subj_id_array, combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict):
    check_ch = 'cz'
    epoch_count = []
    for ev in combine_epoch_dict.keys():
        tmp_count = [len(x) for x in combine_epoch_dict[ev][check_ch]]
        if len(tmp_count)>0:
            epoch_count.append(tmp_count)
    epoch_count = np.sum(np.vstack(epoch_count),axis=0)
    print("Target number of epoch = 2700 (450 epochs* 3 runs* 2 time-lock)")
    print(f"Number after removal = {epoch_count}")
    print("Remove subjects with preserved trial less than half of the target.")
    rm_subj_idx = epoch_count<0.5*2700
    print(f"Remove subjects: {np.array(subj_id_array)[rm_subj_idx]}")

    # Remove subjects by rm_subj_idx for all items in all dictionaries
    keep_subj_idx = ~rm_subj_idx
    for ev in combine_epoch_dict.keys():
        for ch in combine_epoch_dict[ev].keys():
            combine_epoch_dict[ev][ch] = [x for i, x in enumerate(combine_epoch_dict[ev][ch]) if keep_subj_idx[i]]

    for ev in combine_vtc_dict.keys():
        for ch in combine_vtc_dict[ev].keys():
            combine_vtc_dict[ev][ch] = [x for i, x in enumerate(combine_vtc_dict[ev][ch]) if keep_subj_idx[i]]

    for ev in combine_react_dict.keys():
        for ch in combine_react_dict[ev].keys():
            combine_react_dict[ev][ch] = [x for i, x in enumerate(combine_react_dict[ev][ch]) if keep_subj_idx[i]]

    for ev in in_out_zone_dict.keys():
        for ch in in_out_zone_dict[ev].keys():
            in_out_zone_dict[ev][ch] = [x for i, x in enumerate(in_out_zone_dict[ev][ch]) if keep_subj_idx[i]]

    return combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict


def eeg_preproc_subj_level(subj_id, preproc_params):
    # unpacked preproc_params
    is_bpfilter = preproc_params['is_bpfilter']
    bp_f_range = preproc_params['bp_f_range']
    is_reref = preproc_params['is_reref']
    reref_ch = preproc_params['reref_ch']
    is_ica_rmEye = preproc_params['is_ica_rmEye']
    baseline_length = preproc_params['baseline_length']
    epoch_reject_crit = preproc_params['epoch_reject_crit']
    is_detrend = preproc_params['is_detrend']
    ch_names = preproc_params['ch_names']
    is_overwrite = preproc_params['is_overwrite']
    # load EEG
    subj_EEG_dict = dict()
    rm_ch_dict = dict()
    # get all the vdhr files in raw folder
    raw_EEG_path = os.path.join(data_path, f'sub-{subj_id}', 'eeg')
    preproc_save_path = os.path.join(data_save_path,f"sub-{subj_id}")
    if not os.path.exists(preproc_save_path):
        os.makedirs(preproc_save_path, exist_ok=True)
    filename_list = [os.path.basename(x) for x in glob.glob(os.path.join(raw_EEG_path,"*.vhdr"))]
    # check if subject's EEG has been preprocessed.
    for fname in filename_list:
        # get run id
        run_id = fname.split('.')[0][-1]
        if "cpt" in fname.lower():
            key_name = "gradcpt"+run_id
        else:
            key_name = "rest"+run_id
        # define savepath
        preproc_fname = os.path.join(preproc_save_path,fname.split('.')[0]+'_preproc_eeg.fif')
        EEG_raw = fix_and_load_brainvision(os.path.join(raw_EEG_path,fname))
        if not os.path.exists(preproc_fname) or is_overwrite:
            print(f"Start preprocessing {preproc_fname}")
            EEG, rm_ch_list = eeg_preproc_basic(EEG_raw, is_bpfilter=is_bpfilter, bp_f_range=bp_f_range,
                                is_reref=is_reref, reref_ch=reref_ch,
                                is_ica_rmEye=is_ica_rmEye)
            EEG.save(preproc_fname, overwrite=True)
        else:
            # load existed EEG
            EEG = mne.io.read_raw(preproc_fname,preload=True)
            # reconstruct missing channels
            rm_ch_list = list(set(EEG_raw.ch_names) - set(EEG.ch_names))
        subj_EEG_dict[key_name] = EEG    
        rm_ch_dict[key_name] = rm_ch_list

    return subj_EEG_dict, rm_ch_dict

def eeg_epoch_subj_level(key_name, subj_EEG_dict):
    subj_epoch_dict = dict()
    subj_vtc_dict = dict()
    subj_react_dict = dict()
    # check if event_file exist
    event_file = os.path.join(data_save_path,f"{key_name}",
                            f"{key_name}_task-gradCPT_run-01_events.tsv")
    if not os.path.exists(event_file):
        gen_EEG_event_tsv(int(key_name.split('-')[-1]))
    # for each run
    for run_id in np.arange(1,4):
        subj_epoch_dict[f"run{run_id:02d}"] = dict()
        subj_vtc_dict[f"run{run_id:02d}"] = dict()
        subj_react_dict[f"run{run_id:02d}"] = dict()
        EEG = subj_EEG_dict[f"gradcpt{run_id}"]
        # load corresponding event file
        event_file = os.path.join(data_save_path,f"{key_name}",
                                f"{key_name}_task-gradCPT_run-{run_id:02d}_events.tsv")
        events, event_labels_lookup, vtc_list, reaction_time = tsv_to_events(event_file, EEG.info["sfreq"])
        # for each condition
        for select_event in event_labels_lookup.keys():
            if np.any(events[:,-1]==event_labels_lookup[select_event]):
                ev_vtc = vtc_list[events[:,-1]==event_labels_lookup[select_event]]
                ev_react = reaction_time[events[:,-1]==event_labels_lookup[select_event]]
                event_duration = 1.6 if select_event.split('_')[-1]=='response' else 1.8
                baseline_length = -1.2 if select_event.split('_')[-1]=='response' else -0.2
                try:    
                    epochs = epoch_by_select_event(EEG, events, event_labels_lookup,
                                                                select_event=select_event,
                                                                baseline_length=baseline_length,
                                                                epoch_reject_crit=epoch_reject_crit,
                                                                is_detrend=is_detrend,
                                                                event_duration=event_duration,
                                                                verbose=False)
                    # remove vtc that is dropped
                    ev_vtc = ev_vtc[[len(x)==0 for x in epochs.drop_log]]
                    # remove reaction time that is dropped
                    ev_react = ev_react[[len(x)==0 for x in epochs.drop_log]]
                except:
                    print("="*20)
                    print(f"No clean trial found in {key_name}_gradCPT{run_id} ({select_event}).")    
                    print("="*20)
                    epochs = []
            else:
                epochs=[]         
                ev_vtc = []         
                ev_react = []                                                  
            # save epochs
            subj_epoch_dict[f"run{run_id:02d}"][select_event] = epochs
            subj_vtc_dict[f"run{run_id:02d}"][select_event] = ev_vtc
            subj_react_dict[f"run{run_id:02d}"][select_event] = ev_react

    return subj_epoch_dict, subj_vtc_dict, subj_react_dict, event_labels_lookup
