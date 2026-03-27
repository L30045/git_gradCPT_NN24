#%% load library
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import copy
import pandas as pd
from tqdm import tqdm
import pickle
import glob
import time
import sys
# from spectral_connectivity import Multitaper, Connectivity
# from spectral_connectivity.transforms import prepare_time_series


filepath = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/sourcedata/raw/sub-SPARK-testing-002")
# read replace opto location
opt_loc_csv = pd.read_csv(os.path.join(filepath, 'Replace_optodes_list.csv'))
nearest_1020 = opt_loc_csv['Nearest 10-20 system '].copy()
nearest_1020.iloc[4] = 'FCz'

#%% functions
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

def smoothing_VTC_gaussian_array(vtc, sigma=None, alpha=2.5, L=None, radius=None, truncate=4, savepath=None):
    """
    Smooth VTC using gaussian_filter1d.
    ---------------------------------------------------
    Input:
        vtc_dict: VTC Dict.
        sigma: stddev used in gaussian filter.
        alpha: default parameter for determining sigma in Matlab gausswin function
        L: designed window size. use this parameter to determine radius. If None, define window size by sigma and truncate, or radius.
        radius: radius of gaussian filter. Ignored if L is given.
        truncate: number of stddev away from center will be truncated
        savepath: where to save smoothed VTC
    Output:
        smooth_vtc_dict: smoothed VTC Dict.
    """
    # define  from L and alpha if sigma is not given
    if not sigma:
        if not L:
            sigma = 12
        else:
            sigma = (L-1)/(2*alpha) # default formula for stddev in Matlab gausswin function
    # define radius if L is given
    if L:
        radius = np.ceil((L-1)/2).astype(int) # radius of gaussian filter
    # smooth VTC
    smooth_vtc = sp.ndimage.gaussian_filter1d(vtc, sigma=sigma, radius=radius, truncate=4)
    return smooth_vtc

def tsv_to_events(event_file, sfreq):
    #check if event_file exists
    if not os.path.exists(event_file):
        event_file = event_file.replace("run-0", "run-")
        if not os.path.exists(event_file):
            raise FileNotFoundError("Event.tsv not found.")
    events_df = pd.read_csv(event_file,sep='\t')
    event_ids = np.ones(len(events_df))*np.nan
    # mnt-correct
    event_ids[(events_df['trial_type']=='mnt')&(events_df['response_code']==0)] = 0
    # mnt-incorrect
    event_ids[(events_df['trial_type']=='mnt')&(events_df['response_code']!=0)] = -2
    # city-correct
    event_ids[(events_df['trial_type']=='city')&(events_df['response_code']!=0)] = 1
    # city-incorrect
    event_ids[(events_df['trial_type']=='city')&(events_df['response_code']==0)] = -1
    
    event_labels_lookup = dict(city_incorrect=-1, city_correct=1,
                            mnt_incorrect=-2, mnt_correct=0,
                            city_incorrect_response=-11, city_correct_response=11,
                            mnt_incorrect_response=-12, mnt_correct_response=10)
    # check if smooth VTC in eventfiles
    if 'VTC_smoothed' in events_df.columns:
        smoothed_vtc = events_df["VTC_smoothed"]
    else:
        smoothed_vtc = smoothing_VTC_gaussian_array(events_df["VTC"], L=20)
    # create events array (onset, stim_channel_voltage, event_id)
    events_stim_onset = np.column_stack(((events_df["onset"]*sfreq).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        event_ids.astype(int)))
    events_response = np.column_stack((((events_df["onset"]+events_df["reaction_time"])*sfreq).astype(int),
                        np.zeros(len(events_df), dtype=int),
                        (event_ids+10*((event_ids>=0).astype(int)*2-1)).astype(int)))
    # stack together
    events = np.vstack([events_stim_onset,events_response])
    # extract VTC
    vtc_list = np.tile(smoothed_vtc,2)
    # extract reaction time
    reaction_time = np.concatenate([events_df["reaction_time"].values, -1*events_df["reaction_time"].values])
    
    return events, event_labels_lookup, vtc_list, reaction_time

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

#%% load EEG and preprocessing
bp_f_range = [0.1, 45] #band pass filter range (Hz)
reref_ch = 'average' # None as No reref since system has reref to A1 already
# reref_ch = None
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=500e-6 #unit:V
                        ) # epoch with amp more than this threshold in any channels will be rejected
is_detrend = 1 # 0:constant, 1:linear, None
is_overwrite = False # Force to re run preprocessing if it is True

# load EEG
subj_EEG_dict = dict()
subj_ICA_dict = dict()
rm_ch_dict = dict()
# get all the csv files in raw folder
filename_list = [os.path.basename(x) for x in glob.glob(os.path.join(filepath,"*.csv"))]
# check if subject's EEG has been preprocessed.
for fname in filename_list:
    # get run id
    run_id = fname.split('.')[0][-1]
    if "cpt" in fname.lower():
        key_name = "gradcpt"+run_id
    else:
        continue
    # define savepath
    preproc_fname = os.path.join(filepath,fname.split('.')[0]+'_preproc_eeg.fif')
    ica_fname = os.path.join(filepath,fname.split('.')[0]+'_ica.fif')
    print(f"Start preprocessing {preproc_fname}")
    # =================================================
    # EEG preprocessing
    # check if preproc file exist
    if os.path.exists(preproc_fname):
        print(f"Preproc file exists, skipping preprocessing: {preproc_fname}")
        EEG = mne.io.read_raw_fif(preproc_fname, preload=True, verbose=False)
        rm_ch_list = [ch for ch in nearest_1020 if ch not in EEG.ch_names]
    else:
        # load EEG raw
        eeg_csv = pd.read_csv(os.path.join(filepath,fname), skiprows=12, index_col=False)
        ch_cols = [c for c in eeg_csv.columns if c.strip().startswith('CH')]
        eeg_csv = eeg_csv[['Time(s)', 'TRIGGER(DIGITAL)'] + ch_cols]
        # truncate EEG by trigger: keep rows between first and last button press (trigger ~ 0)
        trig = eeg_csv['TRIGGER(DIGITAL)']
        pressed = trig[trig == 0].index
        eeg_csv = eeg_csv.loc[pressed[0]:pressed[-1]].reset_index(drop=True)
        # put eeg_csv into mne raw object
        sfreq = np.round(np.median(1/np.diff(eeg_csv['Time(s)'])))  # Hz
        ch_names = nearest_1020.tolist()
        ch_types = ['eeg'] * len(ch_cols)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = eeg_csv[ch_cols].values.T * 1e-6  # convert uV to V
        EEG = mne.io.RawArray(data, info)
        # assign 10-20 system locations
        montage = mne.channels.make_standard_montage('standard_1020')
        EEG.set_montage(montage, on_missing='warn')
        rm_ch_list = []
        # band-pass filtering (all channels)
        EEG.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1],picks='all',verbose=False)
        # check flat channels
        rm_ch_list.extend(check_flat_channels(EEG))
        # check variance
        rm_ch_list.extend(check_abnormal_var_channels(EEG))
        # drop bad channels
        if len(rm_ch_list)>0:
            EEG.drop_channels(rm_ch_list)
        # re-reference to common average
        if reref_ch=='average':
            EEG.set_eeg_reference(ref_channels='average', ch_type='eeg',verbose=False)
        elif reref_ch:
            EEG.set_eeg_reference(ref_channels=reref_ch, ch_type='eeg',verbose=False)
        EEG.save(preproc_fname, overwrite=True)
    # ICA
    if os.path.exists(ica_fname):
        print(f"ICA file exists, skipping ICA: {ica_fname}")
        ica = mne.preprocessing.read_ica(ica_fname)
    else:
        ica = mne.preprocessing.ICA(n_components=0.99,
                method='infomax', random_state=42,verbose=False)
        ica.fit(EEG, picks=['eeg'],verbose=False)
        ica.save(ica_fname, overwrite=True)
    # =================================================
    subj_EEG_dict[key_name] = EEG    
    subj_ICA_dict[key_name] = ica
    rm_ch_dict[key_name] = rm_ch_list

#%% generate event files
# get all files with .mat ext in gradcpt_path
files = [f for f in os.listdir(filepath) 
            if os.path.isfile(os.path.join(filepath, f)) 
            and f.endswith('.mat')]
# get EEG path
filename_list = [os.path.basename(x) for x in glob.glob(os.path.join(filepath,"*.csv"))]
# load VTC information
vtc_dict = None
smooth_vtc_dict= None

for fname in filename_list:
    # check if fname is a run session
    if "cpt" not in fname.lower():
        continue
    # get run id
    run_id = fname.lower().split("run-0")[-1][0]
    # load corresponding gradCPT
    f_cpt = files[[i for i, x in enumerate(files) if x.split('run-0')[1][0]==run_id][0]]
    data_cpt = sp.io.loadmat(os.path.join(filepath,f_cpt))
    # get reaction time. Remove last trial since it is a fade-out only trial.
    react_time = data_cpt['response'][:-1,4] # sec
    # assign correct response code
    # correct/incorrect city trial in data_cpt is ('city', 1) and ('city', 0).
    # correct/incorrect mnt trial in data_cpt is ('mnt', 0) and ('mnt', -1).
    # assigned response code: city_correct=1, city_incorrect=-1, mnt_correct=0, mnt_incorrect=-2
    response_code = data_cpt['response'][:-1,6].astype(int)
    # assign mnt_incorrect trials
    response_code[response_code==-1] = -2
    # assign city_incorrect trials
    response_code[(response_code==0)&(data_cpt['response'][:-1,0]==2)] = -1
    # gradcpt starttime
    starttime_cpt = data_cpt['starttime'][0][0]
    # event onset time
    t_onset = data_cpt['ttt'][:-1,0] - starttime_cpt # exclude last event since it is a fade-out-only event.
    # check if event onset time exceed EEG recording time
    if t_onset[-1]>EEG.times[-1]:
        raise ValueError("Event onset time exceed EEG recording time.")
    # VTC
    if not vtc_dict:
        # get mean and std RT for trials with non-zero RT
        meanRT = np.nanmean(react_time[react_time > 0])
        stdRT = np.nanstd(react_time[react_time > 0])
        original_vtc = copy.deepcopy(react_time)
        # fill in no reaction time trial with linear interpolation
        non_zero_idx = np.where(original_vtc>0)[0]
        zero_idx = np.where(original_vtc==0)[0]
        if len(zero_idx) > 0 and len(non_zero_idx) > 0:
            # use linear interpolation to fill missing values
            original_vtc[zero_idx] = np.interp(zero_idx, non_zero_idx, original_vtc[non_zero_idx])
        # calculate VTC
        original_vtc = np.abs((original_vtc-meanRT)/stdRT)
        # smooth VTC
        smoothed_vtc = smoothing_VTC_gaussian_array(original_vtc, L=20)
    else:
        original_vtc = vtc_dict["run-0"+run_id]
        smoothed_vtc = smooth_vtc_dict["run-0"+run_id]
    # create DataFrame
    ev_df = pd.DataFrame(columns=[
        'onset',
        'duration', 
        'value',
        'trial_type',
        'exemplar',
        'reaction_time',
        'response_code',
        'VTC',
        'VTC_smoothed'
    ])
    ev_df['onset'] = t_onset
    ev_df['duration'] = np.diff(data_cpt['ttt'][:,0])
    ev_df['value'] = np.ones(t_onset.shape).astype(int) # fix amplitude
    ev_df['trial_type'] = ['city' if x==2 else 'mnt' for x in data_cpt['response'][:-1,0]]
    ev_df['exemplar'] = np.zeros(t_onset.shape).astype(int) # missing stimulus figure id
    ev_df['reaction_time'] = react_time
    ev_df['response_code'] = response_code
    ev_df['VTC'] = original_vtc
    ev_df['VTC_smoothed'] = smoothed_vtc
    # save dataframe
    save_filename = os.path.join(filepath, f'{fname.split('.')[0]}_events.tsv')
    ev_df.to_csv(save_filename, sep='\t', index=False)

#%% epoch by events
# create dictionary to store data
subj_epoch_dict = dict()
subj_vtc_dict = dict()
subj_react_dict = dict()
# for each run
for run_id in np.arange(1,4):
    subj_epoch_dict[f"run{run_id:02d}"] = dict()
    subj_vtc_dict[f"run{run_id:02d}"] = dict()
    subj_react_dict[f"run{run_id:02d}"] = dict()
    EEG = subj_EEG_dict[f"gradcpt{run_id}"]
    # load corresponding event file
    event_file = glob.glob(os.path.join(filepath, f'*run-0{run_id}_events.tsv'))[0]
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

#%% combine runs
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
thres_vtc = np.median(np.concatenate([subj_vtc_dict[f"run{run_id:02d}"][event]
                                        for run_id in range(1, 4)
                                        for event in event_labels_lookup.keys()
                                        if not event.endswith("_response") and len(subj_vtc_dict[f"run{run_id:02d}"][event]) > 0]))
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
    
    tmp_epoch_list = []
    tmp_vtc_list = []
    tmp_react_list = []
    tmp_in_out_zone_list = []
    for run_id in np.arange(1,4):
        loc_e = subj_epoch_dict[f"run{run_id:02d}"][select_event]
        loc_v = subj_vtc_dict[f"run{run_id:02d}"][select_event]
        loc_r = subj_react_dict[f"run{run_id:02d}"][select_event]
        if len(loc_e)>0:
            tmp_epoch_list.append(loc_e)
            tmp_vtc_list.append(loc_v)
            tmp_react_list.append(loc_r)
            tmp_in_out_zone_list.append(loc_v<thres_vtc)
    # initialize append values
    concat_epoch = []
    concat_vtc = []
    concat_react = []
    concat_ch_in_out_zone = []
    # for each channel, create an epoch
    for ch in ch_names:
        # initialize append values
        concat_epoch = []
        concat_vtc = []
        concat_react = []
        concat_ch_in_out_zone = []
        if len(tmp_epoch_list)>0:
            ch_picked_epoch = [x.copy().pick(ch) for x in tmp_epoch_list if ch in x.ch_names]
            if len(ch_picked_epoch)>0:
                concat_epoch = mne.concatenate_epochs(ch_picked_epoch,verbose=False)
                concat_vtc = np.concatenate([x for x,y in zip(tmp_vtc_list,tmp_epoch_list) if ch in y.ch_names])
                concat_react = np.concatenate([x for x,y in zip(tmp_react_list,tmp_epoch_list) if ch in y.ch_names])
                concat_ch_in_out_zone = np.concatenate([x for x,y in zip(tmp_in_out_zone_list,tmp_epoch_list) if ch in y.ch_names])
        epoch_dict[ch].append(concat_epoch)
        vtc_dict[ch].append(concat_vtc)
        react_dict[ch].append(concat_react)
        ch_in_out_zone_dict[ch].append(concat_ch_in_out_zone)

    combine_epoch_dict[select_event] = epoch_dict
    combine_vtc_dict[select_event] = vtc_dict
    combine_react_dict[select_event] = react_dict
    in_out_zone_dict[select_event] = ch_in_out_zone_dict


#%% visualization
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['Fz','Cz','Pz','Oz']

# Extract cross-subject ERPs for both conditions
condition_data = {}
for select_event in select_events:
    condition_data[select_event] = dict()
    for ch in vis_ch:
        subj_epoch_array = combine_epoch_dict[select_event][ch]
        xSubj_erps = []
        for epoch in subj_epoch_array:
            # If subject epoch exist, Get average ERP for this subject
            if len(epoch)>0:
                xSubj_erps.append(np.squeeze(epoch.get_data()))
        xSubj_erps = np.vstack(xSubj_erps)
        condition_data[select_event][ch] = {'erps': xSubj_erps, 'n_subjects': xSubj_erps.shape[0]}

# Plot comparison for each channel
for ch in vis_ch:
    plt.figure(figsize=(10, 6))

    for idx, select_event in enumerate(select_events):
        # xSubj_erps = condition_data[select_event][ch]['erps']
        n_subjects = condition_data[select_event][ch]['n_subjects']
        plt_erps = condition_data[select_event][ch]['erps']
        # plt_erps = np.vstack([x[ch_i,:] for x in xSubj_erps])
        # Calculate mean and SEM across subjects
        mean_erp = np.mean(plt_erps, axis=0)
        sem_erp = np.std(plt_erps, axis=0) / np.sqrt(n_subjects)
        upper_bound = mean_erp + 2 * sem_erp
        lower_bound = mean_erp - 2 * sem_erp

        # Get time vector and convert to milliseconds
        time_vector = combine_epoch_dict[select_event][ch][0].times * 1000

        # Plot
        # label = select_event.replace('_', ' ').title()
        label = 'City' if select_event.split('_')[0]=='city' else 'Mountain'
        plt.plot(time_vector, mean_erp, color=colors[idx], linewidth=2, label=f'{label} ({n_subjects})')
        plt.fill_between(time_vector, lower_bound, upper_bound, alpha=0.3, color=colors[idx])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{ch.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
