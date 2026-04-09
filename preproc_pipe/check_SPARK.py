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
import asrpy
# from spectral_connectivity import Multitaper, Connectivity
# from spectral_connectivity.transforms import prepare_time_series


filepath = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/sourcedata/raw/sub-SPARK-testing-003")
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
        ch_names = nearest_1020.tolist()
    else:
        # load EEG raw
        eeg_csv = pd.read_csv(os.path.join(filepath,fname), skiprows=12, index_col=False)
        ch_cols = [c for c in eeg_csv.columns if c.strip().startswith('CH')]
        eeg_csv = eeg_csv[['Time(s)', 'TRIGGER(DIGITAL)'] + ch_cols]
        # truncate EEG by trigger: keep rows between first and last button press (trigger ~ 0)
        trig = eeg_csv['TRIGGER(DIGITAL)']
        pressed = trig[trig == 0].index
        # check if there is no trigger pressed, skip this file
        if len(pressed)<1:
            continue
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
        # ASR
        asr = asrpy.ASR(sfreq=EEG.info["sfreq"], cutoff=15)
        asr.fit(EEG)
        EEG = asr.transform(EEG)
        # save preprocessing results
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
    #  remove potential eye components (if any) using Fz channles
    eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['Fz'], measure='correlation', verbose=False)
    ica.exclude = eog_inds
    EEG = ica.apply(EEG,verbose=False)
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
for key_name in subj_EEG_dict.keys():
    run_id = int(key_name.split('gradcpt')[-1])
    subj_epoch_dict[f"run{run_id:02d}"] = dict()
    subj_vtc_dict[f"run{run_id:02d}"] = dict()
    subj_react_dict[f"run{run_id:02d}"] = dict()
    EEG = subj_EEG_dict[key_name]
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
                print(f"No clean trial found in gradCPT{run_id} ({select_event}).")    
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
thres_vtc = np.median(np.concatenate([subj_vtc_dict[key_name][event]
                                        for key_name in subj_vtc_dict.keys()
                                        for event in event_labels_lookup.keys()
                                        if not event.endswith("_response") and len(subj_vtc_dict[key_name][event]) > 0]))
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
    for key_name in subj_epoch_dict.keys():
        loc_e = subj_epoch_dict[key_name][select_event]
        loc_v = subj_vtc_dict[key_name][select_event]
        loc_r = subj_react_dict[key_name][select_event]
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

# %% Cross-subject ERP plot (001, 002, 003)
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['Fz', 'Cz', 'Pz', 'Oz']

base_raw_dir = os.path.dirname(filepath)
subj_ids = ['001', '002', '003']

# Collect per-subject mean ERPs: {event: {ch: [mean_erp_subj1, ...]}}
xsubj_erp_dict = {ev: {ch: [] for ch in vis_ch} for ev in select_events}
time_vec = None

for subj_id in subj_ids:
    subj_dir = os.path.join(base_raw_dir, f'sub-SPARK-testing-{subj_id}')
    preproc_files = sorted(glob.glob(os.path.join(subj_dir, '*preproc_eeg.fif')))
    subj_all_epochs = {ev: [] for ev in select_events}

    for preproc_f in preproc_files:
        run_tag = preproc_f.split('run-')[1][:2]  # e.g. '01'
        ica_fname = preproc_f.split('_preproc_eeg')[0]+'_ica.fif'
        event_files = glob.glob(os.path.join(subj_dir, f'*run-{run_tag}*events.tsv'))
        if len(event_files) == 0:
            continue
        EEG = mne.io.read_raw_fif(preproc_f, preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(ica_fname)
        #  remove potential eye components (if any) using Fz channles
        eog_inds, eog_scores = ica.find_bads_eog(EEG, ch_name=['Fz'], measure='correlation', verbose=False)
        ica.exclude = eog_inds
        EEG = ica.apply(EEG,verbose=False)
        events_arr, ev_labels, _, _ = tsv_to_events(event_files[0], EEG.info['sfreq'])

        for select_event in select_events:
            if not np.any(events_arr[:, -1] == ev_labels[select_event]):
                continue
            try:
                ev_epochs = epoch_by_select_event(EEG, events_arr, ev_labels,
                                                  select_event=select_event,
                                                  baseline_length=-0.2,
                                                  epoch_reject_crit=epoch_reject_crit,
                                                  is_detrend=is_detrend,
                                                  event_duration=1.8,
                                                  verbose=False)
                subj_all_epochs[select_event].append(ev_epochs)
            except Exception:
                pass

    # Compute subject-level mean ERP per condition per channel
    for select_event in select_events:
        if len(subj_all_epochs[select_event]) == 0:
            continue
        # Pick only vis_ch (channels present in all runs) before concatenating
        available_vis_ch = [ch for ch in vis_ch if all(ch in ep.ch_names for ep in subj_all_epochs[select_event])]
        aligned = [ep.copy().pick(available_vis_ch) for ep in subj_all_epochs[select_event]]
        concat = mne.concatenate_epochs(aligned, verbose=False)
        if time_vec is None:
            time_vec = concat.times * 1000
        for ch in vis_ch:
            if ch not in concat.ch_names:
                continue
            ch_data = concat.copy().pick(ch).get_data()  # (n_epochs, 1, n_times)
            mean_erp = np.mean(ch_data[:, 0, :], axis=0)  # (n_times,)
            xsubj_erp_dict[select_event][ch].append(mean_erp)

# Plot
fig, axes = plt.subplots(len(vis_ch), 1, figsize=(10, 3 * len(vis_ch)), sharex=True)
for ax, ch in zip(axes, vis_ch):
    for idx, select_event in enumerate(select_events):
        erps = xsubj_erp_dict[select_event][ch]
        if len(erps) == 0:
            continue
        erp_arr = np.vstack(erps)  # (n_subjects, n_times)
        n_subj = erp_arr.shape[0]
        mean_erp = np.mean(erp_arr, axis=0)
        sem_erp = np.std(erp_arr, axis=0) / np.sqrt(n_subj)
        label = 'City' if select_event.split('_')[0] == 'city' else 'Mountain'
        ax.plot(time_vec, mean_erp, color=colors[idx], linewidth=2, label=f'{label} (n={n_subj})')
        ax.fill_between(time_vec, mean_erp - sem_erp, mean_erp + sem_erp, alpha=0.3, color=colors[idx])
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(ch.upper())
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Time (ms)')
plt.suptitle('Cross-Subject ERP (sub-001, 002, 003)', fontsize=14)
plt.tight_layout()
plt.show()

#%% Dipole fitting
# Fit a single equivalent current dipole (ECD) per subject per condition
# using a sphere head model (no MRI available).
# Strategy:
#   1. Load preprocessed EEG + apply ICA for each subject/run
#   2. Epoch by event, compute evoked (ERP average)
#   3. Compute noise covariance from pre-stimulus baseline
#   4. Fit dipole at the time of the ERP peak amplitude (Pz)
#   5. Store dipole fit results (position, orientation, GOF) per subject

dipole_select_events = ['mnt_correct']
dipole_base_raw_dir = os.path.dirname(filepath)
dipole_subj_ids = ['001', '002', '003']
dipole_peak_ch = 'Pz'          # channel used to identify ERP peak time
dipole_fit_twindow = (0.25, 0.45)  # time window (s) around the peak to fit dipole

# Results: {event: [{subj, pos, ori, gof, peak_time, amplitude}]}
dipole_results = {ev: [] for ev in dipole_select_events}

for subj_id in dipole_subj_ids:
    subj_dir = os.path.join(dipole_base_raw_dir, f'sub-SPARK-testing-{subj_id}')
    preproc_files = sorted(glob.glob(os.path.join(subj_dir, '*preproc_eeg.fif')))

    subj_epochs_per_event = {ev: [] for ev in dipole_select_events}

    for preproc_f in preproc_files:
        run_tag = preproc_f.split('run-')[1][:2]
        ica_fname = preproc_f.split('_preproc_eeg')[0] + '_ica.fif'
        event_files = glob.glob(os.path.join(subj_dir, f'*run-{run_tag}*events.tsv'))
        if len(event_files) == 0:
            continue
        EEG = mne.io.read_raw_fif(preproc_f, preload=True, verbose=False)
        if os.path.exists(ica_fname):
            ica = mne.preprocessing.read_ica(ica_fname)
            eog_inds, _ = ica.find_bads_eog(EEG, ch_name=['Fz'], measure='correlation', verbose=False)
            ica.exclude = eog_inds
            EEG = ica.apply(EEG, verbose=False)
        events_arr, ev_labels, _, _ = tsv_to_events(event_files[0], EEG.info['sfreq'])

        for select_event in dipole_select_events:
            if not np.any(events_arr[:, -1] == ev_labels[select_event]):
                continue
            try:
                ep = epoch_by_select_event(EEG, events_arr, ev_labels,
                                           select_event=select_event,
                                           baseline_length=-0.2,
                                           epoch_reject_crit=epoch_reject_crit,
                                           is_detrend=is_detrend,
                                           event_duration=1.8,
                                           verbose=False)
                subj_epochs_per_event[select_event].append(ep)
            except Exception as e:
                print(f"sub-{subj_id} {select_event} run-{run_tag}: skipped ({e})")

    # Fit dipole per condition for this subject
    for select_event in dipole_select_events:
        ep_list = subj_epochs_per_event[select_event]
        if len(ep_list) == 0:
            print(f"sub-{subj_id} {select_event}: no epochs, skipping dipole fit.")
            continue

        # Align channels across runs and concatenate
        common_chs = list(set.intersection(*[set(ep.ch_names) for ep in ep_list]))
        ep_list_aligned = [ep.copy().pick(common_chs) for ep in ep_list]
        all_epochs = mne.concatenate_epochs(ep_list_aligned, verbose=False)

        # Baseline-corrected evoked
        evoked = all_epochs.average()
        evoked.apply_baseline((-0.2, 0))

        # Noise covariance from pre-stimulus baseline
        noise_cov = mne.compute_covariance(all_epochs, tmin=-0.2, tmax=0.0,
                                            method='auto', verbose=False)

        # Sphere head model
        sphere = mne.make_sphere_model(r0='auto', head_radius='auto',
                                       info=evoked.info, verbose=False)

        # Find peak time on Pz (or first available channel)
        peak_ch = dipole_peak_ch if dipole_peak_ch in evoked.ch_names else evoked.ch_names[0]
        peak_ch_idx = evoked.ch_names.index(peak_ch)
        t_mask = (evoked.times >= dipole_fit_twindow[0]) & (evoked.times <= dipole_fit_twindow[1])
        if not np.any(t_mask):
            t_mask = np.ones(len(evoked.times), dtype=bool)
        peak_rel_idx = np.argmax(np.abs(evoked.data[peak_ch_idx, t_mask]))
        peak_time = evoked.times[t_mask][peak_rel_idx]

        # Crop evoked to a narrow window around the peak for stable fitting
        half_win = 0.025  # ±25 ms
        evoked_crop = evoked.copy().crop(peak_time - half_win, peak_time + half_win)

        # Fit dipole
        dip, _ = mne.fit_dipole(evoked_crop, noise_cov, sphere, verbose=False)

        # Filter dipoles: keep only those with GOF >= 80%
        gof_mask = dip.gof >= 80.0
        if not np.any(gof_mask):
            print(f"sub-{subj_id} {select_event}: no dipoles with GOF>=80%, skipping.")
            continue

        # Among remaining dipoles, pick the one with the largest peak amplitude
        best_idx = np.where(gof_mask)[0][np.argmax(np.abs(dip.amplitude[gof_mask]))]
        result = dict(
            subj=subj_id,
            pos=dip.pos[best_idx],           # (x,y,z) in meters
            ori=dip.ori[best_idx],           # unit orientation vector
            gof=dip.gof[best_idx],           # goodness of fit (%)
            amplitude=dip.amplitude[best_idx],  # nAm
            peak_time=peak_time,
        )
        dipole_results[select_event].append(result)
        print(f"sub-{subj_id} {select_event}: GOF={result['gof']:.1f}%, "
              f"pos={np.round(result['pos']*1000, 1)} mm, "
              f"peak={peak_time*1000:.0f} ms")

# Print summary table
print("\n" + "="*60)
print("Dipole fitting summary")
print("="*60)
for ev in dipole_select_events:
    print(f"\nCondition: {ev}")
    print(f"{'Subj':<8} {'GOF%':>6} {'Peak(ms)':>10} {'Amp(nAm)':>10} {'pos_x(mm)':>10} {'pos_y(mm)':>10} {'pos_z(mm)':>10}")
    for r in dipole_results[ev]:
        print(f"{r['subj']:<8} {r['gof']:>6.1f} {r['peak_time']*1000:>10.0f} "
              f"{r['amplitude']*1e9:>10.2f} "
              f"{r['pos'][0]*1000:>10.1f} {r['pos'][1]*1000:>10.1f} {r['pos'][2]*1000:>10.1f}")

#%% Plot IC activities for dipole-fitted subjects
# For each subject that passed dipole fitting:
#   - Reload raw + ICA, exclude EOG components as before
#   - Get IC source epochs (all non-excluded ICs)
#   - Average across epochs (ERP of each IC)
#   - Plot: topomap + time course for each IC, one figure per subject

for select_event in dipole_select_events:
    subj_ids_with_dipole = [r['subj'] for r in dipole_results[select_event]]

    for subj_id in subj_ids_with_dipole:
        subj_dir = os.path.join(dipole_base_raw_dir, f'sub-SPARK-testing-{subj_id}')
        preproc_files = sorted(glob.glob(os.path.join(subj_dir, '*preproc_eeg.fif')))

        ic_epochs_list = []  # collect IC source epochs across runs

        for preproc_f in preproc_files:
            run_tag = preproc_f.split('run-')[1][:2]
            ica_fname = preproc_f.split('_preproc_eeg')[0] + '_ica.fif'
            event_files = glob.glob(os.path.join(subj_dir, f'*run-{run_tag}*events.tsv'))
            if len(event_files) == 0 or not os.path.exists(ica_fname):
                continue

            EEG = mne.io.read_raw_fif(preproc_f, preload=True, verbose=False)
            ica = mne.preprocessing.read_ica(ica_fname)
            eog_inds, _ = ica.find_bads_eog(EEG, ch_name=['Fz'], measure='correlation', verbose=False)
            ica.exclude = eog_inds

            events_arr, ev_labels, _, _ = tsv_to_events(event_files[0], EEG.info['sfreq'])
            if not np.any(events_arr[:, -1] == ev_labels[select_event]):
                continue

            try:
                ep = epoch_by_select_event(EEG, events_arr, ev_labels,
                                           select_event=select_event,
                                           baseline_length=-0.2,
                                           epoch_reject_crit=epoch_reject_crit,
                                           is_detrend=is_detrend,
                                           event_duration=1.8,
                                           verbose=False)
            except Exception as e:
                print(f"IC plot sub-{subj_id} run-{run_tag}: skipped ({e})")
                continue

            # Project epochs into IC source space (excluded ICs zeroed out)
            ic_ep = ica.get_sources(ep)  # shape: (n_epochs, n_ics, n_times)
            ic_epochs_list.append((ica, ic_ep))

        if len(ic_epochs_list) == 0:
            print(f"sub-{subj_id}: no IC epochs available for plotting.")
            continue

        # Use ICA from first run for topomaps
        ica_ref, ic_ep_ref = ic_epochs_list[0]
        times_ms = ic_ep_ref.times * 1000
        active_ics = [i for i in range(len(ica_ref.ch_names)) if i not in ica_ref.exclude]

        # Average IC ERPs across runs: each run contributes its own mean (n_ics may differ
        # across runs if ICA was fit separately, so handle per-IC averaging independently)
        ic_erp_sum = np.zeros((len(active_ics), len(ic_ep_ref.times)))
        ic_erp_count = np.zeros(len(active_ics))
        for ica_run, ic_ep in ic_epochs_list:
            run_active = [i for i in range(len(ica_run.ch_names)) if i not in ica_run.exclude]
            run_data = ic_ep.get_data()  # (n_epochs, n_ics_run, n_times)
            run_erp = np.mean(run_data, axis=0)  # (n_ics_run, n_times)
            for out_row, ic_idx in enumerate(active_ics):
                if ic_idx < len(run_active) and ic_idx in run_active:
                    ic_erp_sum[out_row] += run_erp[ic_idx]
                    ic_erp_count[out_row] += 1
        # Avoid division by zero
        ic_erp_count[ic_erp_count == 0] = np.nan
        ic_erp_avg = ic_erp_sum / ic_erp_count[:, np.newaxis]

        n_active = len(active_ics)

        # One figure per subject: rows=ICs, cols=[topomap, time course]
        fig, axes = plt.subplots(n_active, 2,
                                 figsize=(8, max(2 * n_active, 4)),
                                 gridspec_kw={'width_ratios': [1, 3]})
        if n_active == 1:
            axes = axes[np.newaxis, :]  # ensure 2-D indexing

        fig.suptitle(f'IC activations — sub-SPARK-testing-{subj_id} ({select_event})', fontsize=11)

        for row, ic_idx in enumerate(active_ics):
            ax_topo = axes[row, 0]
            ax_ts   = axes[row, 1]

            # Topomap
            mne.viz.plot_ica_components(ica_ref, picks=[ic_idx], axes=ax_topo,
                                        show=False, colorbar=False)
            ax_topo.set_title(f'IC {ic_idx}', fontsize=8)

            # Time course (ERP averaged across runs)
            ic_data = ic_erp_avg[row]
            ax_ts.plot(times_ms, ic_data, lw=1)
            ax_ts.axvline(0, color='k', lw=0.8, ls='--')
            ax_ts.axhline(0, color='k', lw=0.5, ls=':')
            ax_ts.set_xlabel('Time (ms)', fontsize=7)
            ax_ts.set_ylabel('Amplitude (a.u.)', fontsize=7)
            ax_ts.tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()
