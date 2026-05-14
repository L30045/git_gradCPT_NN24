"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from tqdm import tqdm
import pickle
import gzip
import glob
import time
import sys
from spectral_connectivity import Multitaper, Connectivity
from spectral_connectivity.transforms import prepare_time_series

#%% preprocessing parameter setting
# subj_id_array = [670, 695, 721, 723, 726, 730]
subj_id_array = [670, 671, 673, 695, 719, 721, 723, 726, 727, 730, 733, 746, 751, 755]

ch_names = ['fz','cz','pz','oz']
split_zone_crit = 'vtc'
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
# reref_ch = None # reref to average
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None
is_overwrite = False # Force to re run preprocessing if it is True

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    select_event = select_event,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names,
    is_overwrite = is_overwrite
)

#%% load epoch for each condition.
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
    gz_path = os.path.join(data_save_path, f"sub-{subj_id}", f"sub-{subj_id}_preprocessed_dict.pkl.gz")
    if not os.path.exists(gz_path):
        print(f"sub-{subj_id}: preprocessed dict not found at {gz_path}, skipping.")
        continue
    with gzip.open(gz_path, 'rb') as f:
        _payload = pickle.load(f)
    subj_EEG_dict[f"sub-{subj_id}"] = _payload["EEG"]
    rm_ch_dict[f"sub-{subj_id}"]    = _payload["rm_ch"]
    print(f"sub-{subj_id}: loaded runs {list(_payload['EEG'].keys())}")
    
#%% Epoch data
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
    single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(key_name, subj_EEG_dict[key_name], preproc_params,interp_rt=True)
    # save epochs
    subj_epoch_dict[key_name] = single_subj_epoch_dict
    subj_vtc_dict[key_name] = single_subj_vtc_dict
    subj_react_dict[key_name] = single_subj_react_dict

#%% get pupil size for each trial
import sys
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from utils_eyetracking import preprocess_pupil, get_pupil_epoch
from pupil_labs import neon_recording as nr

# condition boolean index (stimulus-locked) from events_df columns
_cond_base = {
    'mnt_correct':   lambda df: (df['trial_type']=='mnt')  & (df['response_code']==0),
    'mnt_incorrect': lambda df: (df['trial_type']=='mnt')  & (df['response_code']!=0),
    'city_correct':  lambda df: (df['trial_type']=='city') & (df['response_code']>0),
    'city_incorrect':lambda df: (df['trial_type']=='city') & (df['response_code']<0),
}

subj_pupil_dict = dict()
for key_name in tqdm(subj_EEG_dict.keys()):
    subj_id = key_name.split('-')[-1]
    subj_nirs_dir = os.path.join(project_path, f"sub-{subj_id}", 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', f"sub-{subj_id}", 'eye_tracking')
    neon_dirs_subj = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)]) \
                     if os.path.isdir(subj_neon_dir) else []
    snirf_files = sorted([f for f in os.listdir(subj_nirs_dir) if f.endswith('.snirf')]) \
                  if os.path.isdir(subj_nirs_dir) else []
    subj_pupil_dict[key_name] = {}
    for run_name in subj_EEG_dict[key_name].keys():
        if 'gradcpt' not in run_name:
            continue
        run_id  = int(run_name.split('cpt')[-1])
        run_key = f"run{run_id:02d}"
        # load physio file (check in priority order: 20260423 → corrected-idx → plain)
        physio_file = os.path.join(subj_nirs_dir,
                f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260423.tsv")
        if not os.path.isfile(physio_file):
            physio_file = os.path.join(subj_nirs_dir,
            f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv")
        if not os.path.isfile(physio_file):
            physio_file = os.path.join(subj_nirs_dir,
                f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
        if not os.path.isfile(physio_file):
            subj_pupil_dict[key_name][run_key] = {
                ev: np.full((len(subj_epoch_dict[key_name][run_key][ev]), 1), np.nan)
                for ev in event_labels_lookup
            }
            continue
        neon_data = pd.read_csv(physio_file, sep='\t')
        # neon recording object (for blink detection)
        snirf_name = f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_nirs.snirf"
        if neon_dirs_subj and snirf_name in snirf_files:
            neon_idx = snirf_files.index(snirf_name)
            rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[neon_idx])) \
                  if neon_idx < len(neon_dirs_subj) else None
        else:
            rec = None
        # load event file
        event_file = os.path.join(subj_nirs_dir,
            f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if not os.path.isfile(event_file):
            subj_pupil_dict[key_name][run_key] = {
                ev: np.full((len(subj_epoch_dict[key_name][run_key][ev]), 1), np.nan)
                for ev in event_labels_lookup
            }
            continue
        events_df = pd.read_csv(event_file, sep='\t')
        # preprocess pupil (detrend + lowpass)
        t_neon, pupil_d = preprocess_pupil(neon_data, rec=rec, detrend_order=2,
                                           is_rm_phasic=False, events_df=events_df)
        subj_pupil_dict[key_name][run_key] = {}
        if t_neon is None:
            for ev in event_labels_lookup:
                subj_pupil_dict[key_name][run_key][ev] = np.full((len(subj_epoch_dict[key_name][run_key][ev]), 1), np.nan)
            continue
        for ev in event_labels_lookup:
            epochs = subj_epoch_dict[key_name][run_key][ev]
            if len(epochs) == 0:
                subj_pupil_dict[key_name][run_key][ev] = np.full((0, 1), np.nan)
                continue
            base_ev = ev.replace('_response', '')
            if base_ev not in _cond_base:
                subj_pupil_dict[key_name][run_key][ev] = np.full((len(epochs), 1), np.nan)
                continue
            # build per-event dataframe (shift onset for response-locked events)
            ev_df = events_df[_cond_base[base_ev](events_df)].copy()
            if len(ev_df) != 0:
                if ev.endswith('_response'):
                    ev_df['onset'] = ev_df['onset'] + ev_df['reaction_time']
                ev_sel = pd.Series([True] * len(ev_df), index=ev_df.index)
                pupil_epochs = get_pupil_epoch(ev_df, ev_sel, t_neon, pupil_d,
                                            baseline_length=0.2, epoch_length=1.6)
                # align to EEG drop_log: keep only trials not rejected by EEG
                kept = [len(x) == 0 for x in epochs.drop_log]
                subj_pupil_dict[key_name][run_key][ev] = np.array([
                    e for e, k in zip(pupil_epochs, kept) if k
                ])
            else:
                if len(epochs)!=0:
                    print(f"{key_name} - {run_name}: EEG length = {len(epochs)}")
                    raise ValueError("Number of epoch don't match between BIDS and EEG events file.")
                subj_pupil_dict[key_name][run_key][ev] = np.full((len(epochs), 1), np.nan)

#%% Combined runs. Epoch from each run is combined for each subject.
combine_epoch_dict = dict()
combine_vtc_dict = dict()
combine_react_dict = dict()
combine_pupil_dict = dict()
in_out_zone_dict = dict()
"""
combine_epoch_dict: dictionary for combined epochs from each run for subject.
                    combine_epoch_dict["select_event"]["ch"]: list of epoch of selected event and channel. (length equals to number of subjects)
"""
# get median of the vtc for each subject
match split_zone_crit:
    case 'vtc':
        subj_thres_zone = {subj_id: np.median(np.concatenate([subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]
                                                for run_id in range(1, 4)
                                                for event in event_labels_lookup.keys()
                                                if not event.endswith("_response") and len(subj_vtc_dict[subj_id][f"run{run_id:02d}"][event]) > 0]))
                        for subj_id in subj_vtc_dict.keys()}
    case 'react':
        subj_thres_zone = {subj_id: np.median(np.concatenate([subj_react_dict[subj_id][f"run{run_id:02d}"][event]
                                                for run_id in range(1, 4)
                                                for event in event_labels_lookup.keys()
                                                if not event.endswith("_response") and len(subj_react_dict[subj_id][f"run{run_id:02d}"][event]) > 0]))
                        for subj_id in subj_react_dict.keys()}
    case 'pupil':
        # TODO: remove subjects with no eye tracker data (all-nan slice)
        pupil_d_output = [(subj_id, np.concatenate([np.mean(subj_pupil_dict[subj_id][f"run{run_id:02d}"][event],axis=1)
                                                for run_id in range(1, 4)
                                                for event in event_labels_lookup.keys()
                                                if not event.endswith("_response") and len(subj_pupil_dict[subj_id][f"run{run_id:02d}"][event]) > 0]))
                        for subj_id in subj_pupil_dict.keys() if not all(np.isnan(np.concatenate([np.mean(subj_pupil_dict[subj_id][f"run{run_id:02d}"][event],axis=1)
                                                for run_id in range(1, 4)
                                                for event in event_labels_lookup.keys()
                                                if not event.endswith("_response") and len(subj_pupil_dict[subj_id][f"run{run_id:02d}"][event]) > 0])))]
        preserved_subj_id = [x[0] for x in pupil_d_output]
        mean_pupil_d_trial = [x[1] for x in pupil_d_output]
        # calculate the mean pupil size across the trial
        subj_thres_zone = {subj_id: np.nanmedian(x)
                        for subj_id,x in pupil_d_output}
for select_event in event_labels_lookup.keys():
    epoch_dict = dict()
    vtc_dict = dict()
    react_dict = dict()
    pupil_dict = dict()
    ch_in_out_zone_dict = dict()
    # initialize epoch_dict
    for ch in preproc_params['ch_names']:
        epoch_dict[ch] = []
        vtc_dict[ch] = []
        react_dict[ch] = []
        pupil_dict[ch] = []
        ch_in_out_zone_dict[ch] = []
    for subj_id in subj_epoch_dict.keys():
        tmp_epoch_list = []
        tmp_vtc_list = []
        tmp_react_list = []
        tmp_pupil_list = []
        tmp_in_out_zone_list = []
        for run_id in np.arange(1,4):
            loc_e = subj_epoch_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_v = subj_vtc_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_r = subj_react_dict[subj_id][f"run{run_id:02d}"][select_event]
            loc_p = subj_pupil_dict[subj_id][f"run{run_id:02d}"][select_event]
            if len(loc_e)>0:
                tmp_epoch_list.append(loc_e)
                tmp_vtc_list.append(loc_v)
                tmp_react_list.append(loc_r)
                tmp_pupil_list.append(loc_p)
                match split_zone_crit:
                    case 'vtc':
                        tmp_in_out_zone_list.append(loc_v < subj_thres_zone[subj_id])
                    case 'react':
                        tmp_in_out_zone_list.append(loc_r < subj_thres_zone[subj_id])
                    case 'pupil':
                        if subj_id in preserved_subj_id:
                            tmp_in_out_zone_list.append(np.mean(loc_p,axis=1) < subj_thres_zone[subj_id])
                        else:
                            tmp_in_out_zone_list.append(np.full(loc_p.shape[0],np.nan))
        # initialize append values
        concat_epoch = []
        concat_vtc = []
        concat_react = []
        concat_pupil = []
        concat_ch_in_out_zone = []
        # for each channel, create an epoch
        for ch in preproc_params['ch_names']:
            # initialize append values
            concat_epoch = []
            concat_vtc = []
            concat_react = []
            concat_pupil = []
            concat_ch_in_out_zone = []
            if len(tmp_epoch_list)>0:
                ch_picked_epoch = [x.copy().pick(ch) for x in tmp_epoch_list if ch in x.ch_names]
                if len(ch_picked_epoch)>0:
                    concat_epoch = mne.concatenate_epochs(ch_picked_epoch,verbose=False)
                    concat_vtc = np.concatenate([x for x,y in zip(tmp_vtc_list,tmp_epoch_list) if ch in y.ch_names])
                    concat_react = np.concatenate([x for x,y in zip(tmp_react_list,tmp_epoch_list) if ch in y.ch_names])
                    _p_list = [x.mean(axis=1) for x,y in zip(tmp_pupil_list,tmp_epoch_list) if ch in y.ch_names]
                    concat_pupil = np.concatenate(_p_list)
                    concat_ch_in_out_zone = np.concatenate([x for x,y in zip(tmp_in_out_zone_list,tmp_epoch_list) if ch in y.ch_names])
            epoch_dict[ch].append(concat_epoch)
            vtc_dict[ch].append(concat_vtc)
            react_dict[ch].append(concat_react)
            pupil_dict[ch].append(concat_pupil)
            ch_in_out_zone_dict[ch].append(concat_ch_in_out_zone)

    combine_epoch_dict[select_event] = epoch_dict
    combine_vtc_dict[select_event] = vtc_dict
    combine_react_dict[select_event] = react_dict
    combine_pupil_dict[select_event] = pupil_dict
    in_out_zone_dict[select_event] = ch_in_out_zone_dict

#%% remove subjects with number of epoch less than half of the target number of epoch (2700/2)
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict, combine_pupil_dict, preserved_subj_array, removed_subj_array = remove_subject_by_nb_epochs_preserved(subj_id_array, combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict, combine_pupil_dict)
print(f"Preserved subjects: {preserved_subj_array}")
print(f"Removed subjects:   {removed_subj_array}")

#%% If select pupil as in/out zone criteria, further remove epochs with combine_pupil_dict == nan
if split_zone_crit == 'pupil':
    for select_event in event_labels_lookup.keys():
        for ch in preproc_params['ch_names']:
            keep_indices = []
            for subj_idx in range(len(combine_pupil_dict[select_event][ch])):
                pupil_arr = combine_pupil_dict[select_event][ch][subj_idx]
                if len(pupil_arr) == 0:
                    continue  # drop subject with no epochs
                valid_mask = ~np.isnan(pupil_arr)
                if not np.any(valid_mask):
                    continue  # drop subject with all-NaN pupil (no eye tracker)
                if not np.all(valid_mask):
                    epoch_obj = combine_epoch_dict[select_event][ch][subj_idx]
                    if len(epoch_obj) > 0:
                        combine_epoch_dict[select_event][ch][subj_idx] = epoch_obj[valid_mask]
                    combine_vtc_dict[select_event][ch][subj_idx]   = combine_vtc_dict[select_event][ch][subj_idx][valid_mask]
                    combine_react_dict[select_event][ch][subj_idx] = combine_react_dict[select_event][ch][subj_idx][valid_mask]
                    combine_pupil_dict[select_event][ch][subj_idx] = pupil_arr[valid_mask]
                    in_out_zone_dict[select_event][ch][subj_idx]   = in_out_zone_dict[select_event][ch][subj_idx][valid_mask]
                keep_indices.append(subj_idx)
            combine_epoch_dict[select_event][ch]  = [combine_epoch_dict[select_event][ch][i]  for i in keep_indices]
            combine_vtc_dict[select_event][ch]    = [combine_vtc_dict[select_event][ch][i]    for i in keep_indices]
            combine_react_dict[select_event][ch]  = [combine_react_dict[select_event][ch][i]  for i in keep_indices]
            combine_pupil_dict[select_event][ch]  = [combine_pupil_dict[select_event][ch][i]  for i in keep_indices]
            in_out_zone_dict[select_event][ch]    = [in_out_zone_dict[select_event][ch][i]    for i in keep_indices]

#%% Compare in-zone/out-of-zone reaction time
check_ch = 'cz'
in_zone_RT = [x[y] for x,y in zip(combine_react_dict['city_correct'][check_ch],in_out_zone_dict['city_correct'][check_ch])]
out_zone_RT = [x[~y] for x,y in zip(combine_react_dict['city_correct'][check_ch],in_out_zone_dict['city_correct'][check_ch])]
print(f'RT diff (in/out zone) = {np.mean([np.mean(x)-np.mean(y) for x,y in zip(in_zone_RT,out_zone_RT)])*1000:.2f} ms')

# Plot distribution of RT differences
rt_diff_dist = [1000*(np.mean(x)-np.mean(y)) for x,y in zip(in_zone_RT,out_zone_RT)]
plt.figure(figsize=(8, 5))
plt.hist(rt_diff_dist, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(rt_diff_dist), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(rt_diff_dist):.2f} ms')
plt.xlabel('RT Difference (in-zone - out-zone) [ms]')
plt.ylabel('Frequency')
plt.title(f'Distribution of RT Differences ({check_ch.upper()})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Check VTC and Reaction time
plt_vtc = subj_vtc_dict['sub-733']['run02']['city_correct']
plt_react = subj_react_dict['sub-733']['run02']['city_correct']

# Calculate medians
median_vtc = np.median(plt_vtc)
median_react = np.median(plt_react)

# Create masks for VTC
above_median_vtc = plt_vtc >= median_vtc
below_median_vtc = plt_vtc < median_vtc

# Create arrays with NaN for segments we don't want to plot
vtc_above = np.where(above_median_vtc, plt_vtc, np.nan)
vtc_below = np.where(below_median_vtc, plt_vtc, np.nan)
react_above = np.where(above_median_vtc, plt_react, np.nan)
react_below = np.where(below_median_vtc, plt_react, np.nan)

plt.figure(figsize=(15,8))
# Plot VTC segments
plt.plot(vtc_above, '-o', color='red', label='VTC Out zone')
plt.plot(vtc_below, '-o', color='blue', label='VTC In zone')

# Plot React segments
plt.plot(react_above, color='orange', label=f'RT Out {np.mean(plt_react[above_median_vtc]):.3f}')
plt.plot(react_below, color='green', label=f'RT In {np.mean(plt_react[below_median_vtc]):.3f}')

# Add median lines
plt.axhline(y=median_vtc, color='purple', linestyle='--', alpha=0.7, label=f'VTC Median: {median_vtc:.2f}')
plt.axhline(y=median_react, color='brown', linestyle='--', alpha=0.7, label=f'React Median: {median_react:.2f}')

plt.legend()
plt.show()

#%% compare city and mountain ERP
is_save_fig = False
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['fz','cz','pz','oz']

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
                evoked = epoch.average()
                xSubj_erps.append(evoked.data)
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
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'mntC_vs_mntIC_{ch}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()

#%% ERP Image
"""
Plot ERP Image and sorted by VTC. Merge all subjects's epochs into one big epoch.
"""
select_event = "city_correct"
ch = 'cz'
window_size = None  # Number of trials to average. If None, window_size equals to 1% of the data length.
clim = [-10*1e-6, 10*1e-6]
plt_epoch = mne.concatenate_epochs([x for x in combine_epoch_dict[select_event][ch] if len(x)>0])
time_vector = plt_epoch.times
plt_epoch = np.squeeze(plt_epoch.get_data())
if window_size is None:
    window_size = np.max([4,np.floor(plt_epoch.shape[0]*0.01).astype(int)])
plt_vtc = np.concatenate(combine_vtc_dict[select_event][ch])
plt_react = np.concatenate(combine_react_dict[select_event][ch])
# get the
plt_pupil = np.concatenate([x for x in combine_pupil_dict[select_event][ch] if len(x)>0])
title_txt = f'{select_event} - Channel: {ch}'

print("ERP sorted by VTC")
_ = plt_ERPImage(time_vector, plt_epoch, 
                 sort_idx=plt_vtc,
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react)

print("ERP sorted by RT")
_ = plt_ERPImage(time_vector, plt_epoch,
                 sort_idx=plt_react,
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react)

print("ERP sorted by Pupil diameter")
_valid = ~np.isnan(plt_pupil)
_ = plt_ERPImage(time_vector, plt_epoch[_valid],
                 sort_idx=plt_pupil[_valid],
                 smooth_window_size=window_size,
                 clim=[-10*1e-6, 10*1e-6],
                 title_txt=title_txt,
                 ref_onset=plt_react[_valid])

#%% ERSP analysis using multi-taper
start_time = time.time()
select_event = "mnt_correct"
ch = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[select_event][ch])
time_vector = plt_epoch.times
(_,multitaper,_) = plt_multitaper(plt_epoch,
                    time_halfbandwidth_product=time_halfbandwidth_product,
                    time_window_duration=time_window_duration,
                    time_window_step=time_window_step)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"ERSP analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

#%% plot ratio of power to baseline
(_,multitaper,_) = plt_multitaper(plt_epoch, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")

#%% compare trials
target_event = "mnt_correct"
ref_event = "city_correct"
ch = 'cz'
time_halfbandwidth_product = 1 
time_window_duration = 0.2 # sec
time_window_step = 0.05
plt_epoch_target = mne.concatenate_epochs(combine_epoch_dict[target_event][ch])
plt_epoch_ref = mne.concatenate_epochs(combine_epoch_dict[ref_event][ch])
time_vector = plt_epoch_target.times
(_,multitaper,_) = plt_multitaper(plt_epoch_target, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to=plt_epoch_ref)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")

#%% check in-zone vs out-of-zone ratio
check_ch = 'cz'
for select_event in in_out_zone_dict.keys():
    if "_response" not in select_event:
        total_in_zone = 0
        total_out_zone = 0
        print(f"\n{select_event}:")
        for subj_i, in_out_zone in enumerate(in_out_zone_dict[select_event][check_ch]):
            n_in_zone = np.sum(in_out_zone)
            n_out_zone = np.sum(~in_out_zone)
            total_in_zone += n_in_zone
            total_out_zone += n_out_zone
            print(f"  Subject {subj_id_array[subj_i]}: {n_in_zone} in-zone, {n_out_zone} out-of-zone")
        total_trials = total_in_zone + total_out_zone
        if total_trials > 0:
            print(f"  Total: {total_in_zone} in-zone ({total_in_zone/total_trials*100:.1f}%), {total_out_zone} out-of-zone ({total_out_zone/total_trials*100:.1f}%)")
        else:
            print(f"  Total: No trials found")

#%% zone-in vs zone-out
select_event = "mnt_correct"
vis_ch = ["fz","cz","pz","oz"]

# Extract cross-subject ERPs for both conditions
in_zone_erp = dict()
out_zone_erp = dict()
for ch in vis_ch:
    subj_epoch_array = combine_epoch_dict[select_event][ch]
    n_subjects = len(subj_epoch_array)
    subj_in_zone_erp = []
    subj_out_zone_erp = []
    for subj_i, epoch in enumerate(subj_epoch_array):
        # get channel data
        ch_erp = np.squeeze(epoch.get_data())
        # get in-zone/ out-of-zone data
        subj_in_zone_erp.append(np.mean(ch_erp[in_out_zone_dict[select_event][ch][subj_i]],axis=0))
        subj_out_zone_erp.append(np.mean(ch_erp[~in_out_zone_dict[select_event][ch][subj_i]],axis=0))
    in_zone_erp[ch] = np.vstack(subj_in_zone_erp)
    out_zone_erp[ch] = np.vstack(subj_out_zone_erp)

# Plot comparison for each channel
for ch in vis_ch:
    plt_in_zone = in_zone_erp[ch]
    plt_out_zone = out_zone_erp[ch]

    plt.figure(figsize=(10, 6))
    # Calculate mean and SEM across subjects
    mean_in = np.mean(plt_in_zone, axis=0)
    sem_in = np.std(plt_in_zone, axis=0) / np.sqrt(n_subjects)
    upper_in = mean_in + 2 * sem_in
    lower_in = mean_in - 2 * sem_in
    mean_out = np.mean(plt_out_zone, axis=0)
    sem_out = np.std(plt_out_zone, axis=0) / np.sqrt(n_subjects)
    upper_out = mean_out + 2 * sem_out
    lower_out = mean_out - 2 * sem_out

    # Get time vector and convert to milliseconds
    time_vector = combine_epoch_dict[select_event][ch][0].times * 1000

    # Plot
    plt.plot(time_vector, mean_in, color='b', linewidth=2, label='In-zone')
    plt.fill_between(time_vector, lower_in, upper_in, alpha=0.3, color='b')
    plt.plot(time_vector, mean_out, color='r', linewidth=2, label='Out-of-zone')
    plt.fill_between(time_vector, lower_out, upper_out, alpha=0.3, color='r')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{ch.upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# %% In-zone/ out-of-zone ERSP
start_time = time.time()
select_event = "city_correct_response"
ch = 'cz'
time_halfbandwidth_product = 1
time_window_duration = 0.5
time_window_step = 0.1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)

# multitaper
print("In Zone")
_ = plt_multitaper(in_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print("Out of Zone")
_ = plt_multitaper(out_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to="baseline")
print("In Zone / Out of Zone")
(_,multitaper,_) = plt_multitaper(in_zone_erp, 
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   time_window_duration=time_window_duration,
                   time_window_step=time_window_step,
                   ratio_to=out_zone_erp)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"ERSP analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


#%% Compare PSD
select_event = "city_correct"
ch = 'cz'
time_halfbandwidth_product = 1

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_city,_,_) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
(log_power_out_city,multitaper,_) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)

select_event = "mnt_correct"

# Extract cross-subject ERPs for both conditions
subj_epoch_array = combine_epoch_dict[select_event][ch]
time_vector = subj_epoch_array[0].times
n_subjects = len(subj_epoch_array)
in_zone_erp = []
out_zone_erp = []
for subj_i, epoch in enumerate(subj_epoch_array):
    # select trials based on in-zone/out-of-zone condition
    in_zone_mask = in_out_zone_dict[select_event][ch][subj_i]
    out_zone_mask = ~in_out_zone_dict[select_event][ch][subj_i]

    # get in-zone and out-of-zone epochs for specific channel
    in_zone_epochs = epoch[in_zone_mask]
    out_zone_epochs = epoch[out_zone_mask]

    in_zone_erp.append(in_zone_epochs)
    out_zone_erp.append(out_zone_epochs)

in_zone_erp = mne.concatenate_epochs(in_zone_erp)
out_zone_erp = mne.concatenate_epochs(out_zone_erp)
(log_power_in_mnt,_,_) = plt_multitaper(in_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
(log_power_out_mnt,_,connectivity) = plt_multitaper(out_zone_erp,
                            time_halfbandwidth_product=time_halfbandwidth_product,
                            time_window_duration=None,
                            is_plot=False)
print(f"Freq. resolution = {multitaper.frequency_resolution:.2F} Hz")
vis_f_range = [0, 50] # Hz
vis_mask = (connectivity.frequencies>=vis_f_range[0])&(connectivity.frequencies<=vis_f_range[1])
plt.figure()
plt.plot(connectivity.frequencies[vis_mask], log_power_in_city[vis_mask], 'b-', label='City (in zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_out_city[vis_mask], 'b--', label='City (out of zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_in_mnt[vis_mask], 'r-', label='Mnt (in zone)')
plt.plot(connectivity.frequencies[vis_mask], log_power_out_mnt[vis_mask], 'r--', label='Mnt (out of zone)')
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Power ($log\,V^2$)")
plt.grid()
plt.legend()

#%% Time-frequency analysis using wavelet transform
"""
For city_correct trials, apply wavelet transform for frequency range same as bandpass filter range during preprocessing.
1. split city_correct trials into in-the-zone/ out-of-the-zone by 3 criteria: VTC, interpret RT, and pupil diameter.
2. Plot the power difference between in-the-zone and out-of-the-zone.
"""
select_event = "city_correct"
ch           = 'cz'
# logspaced frequencies within bp_f_range; start at 2 Hz (0.1 Hz needs >10 s epochs)
freqs    = np.logspace(np.log10(2), np.log10(bp_f_range[1]), 30)
n_cycles = freqs / 2.0   # standard Morlet: half-cycle bandwidth

# --- define per-subject in/out masks for each criterion ---
criteria = {
    'VTC':       lambda si: (combine_vtc_dict[select_event][ch][si],   None),
    'Interp RT': lambda si: (combine_react_dict[select_event][ch][si], None),
    'Pupil':     lambda si: (combine_pupil_dict[select_event][ch][si], 'nanmask'),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for ax, (crit_name, crit_fn) in zip(axes, criteria.items()):
    in_list, out_list = [], []
    for subj_i, epoch in enumerate(combine_epoch_dict[select_event][ch]):
        if len(epoch) == 0:
            continue
        crit_arr, flag = crit_fn(subj_i)
        crit_arr = np.asarray(crit_arr, dtype=float)
        if flag == 'nanmask':
            valid = ~np.isnan(crit_arr)
            if np.sum(valid) < 2:
                continue
            med      = np.median(crit_arr[valid])
            in_mask  = valid & (crit_arr <  med)
            out_mask = valid & (crit_arr >= med)
        else:
            med      = np.median(crit_arr)
            in_mask  = crit_arr <  med
            out_mask = crit_arr >= med
        if not np.any(in_mask) or not np.any(out_mask):
            continue
        in_list.append(epoch[in_mask])
        out_list.append(epoch[out_mask])

    if not in_list:
        ax.set_title(f'{crit_name}: no data')
        continue

    in_epochs  = mne.concatenate_epochs(in_list,  verbose=False)
    out_epochs = mne.concatenate_epochs(out_list, verbose=False)

    # compute Morlet TFR (average across trials)
    tfr_in  = mne.time_frequency.compute_tfr(in_epochs, method='morlet', freqs=freqs, n_cycles=n_cycles,
                                             average=True, return_itc=False, verbose=False)
    tfr_out = mne.time_frequency.compute_tfr(out_epochs, method='morlet', freqs=freqs, n_cycles=n_cycles,
                                             average=True, return_itc=False, verbose=False)

    # log power ratio: in-zone vs out-of-zone (dB)
    power_diff = 10 * np.log10(tfr_in.data[0] / tfr_out.data[0])   # (n_freqs, n_times)
    times_ms   = tfr_in.times * 1000

    im = ax.pcolormesh(times_ms, freqs, power_diff, cmap='RdBu_r',
                       vmin=-1, vmax=1, shading='auto')
    ax.set_yscale('log')
    ax.set_yticks([2, 4, 8, 16, 32, 45])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axvline(0,   color='k', linestyle='--', linewidth=1.5)
    ax.axvline(800, color='k', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'{crit_name}\n(n_in={len(in_epochs)}, n_out={len(out_epochs)})')
    plt.colorbar(im, ax=ax, label='Power diff (dB)')

fig.suptitle(f'{select_event} — {ch.upper()} — TFR: In vs Out of zone')
plt.show()

#%% Calculate the correlation between EEG band power and VTC/RT.
"""
1. Use a 1 second sliding window with step size of 1 second to calculate EEG spectrum for the entire run.
2. Calculate the correlation between EEG band power and VTC/RT with different time delay.
"""
import scipy.signal as sp_sig

ch          = 'cz'
win_sec     = 1.0   # window length (s)
step_sec    = 1.0   # step size (s) — non-overlapping
max_lag_sec = 10    # ± lag range (s)

bands = {
    'delta': (0.5,  4),
    'theta': (4,    8),
    'alpha': (8,   13),
    'beta':  (13,  30),
    'gamma': (30,  45),
}
n_lags   = 2 * max_lag_sec + 1
lag_axis = np.arange(-max_lag_sec, max_lag_sec + 1)   # seconds

def _zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else np.zeros_like(x)

def _run_xcorr(pwr_z, ts_z, n_wins):
    """Return cross-correlation vector (Pearson r at each lag)."""
    corr = np.correlate(pwr_z, ts_z, mode='full')   # length 2*n_wins - 1
    mid  = n_wins - 1
    out  = np.zeros(n_lags)
    for li, lag in enumerate(lag_axis):
        idx = mid + lag
        if 0 <= idx < len(corr):
            out[li] = corr[idx] / max(n_wins - abs(lag), 1)
    return out

# per-subject lists of cross-correlation vectors (one entry per subject)
subj_xcorr_vtc = {b: [] for b in bands}
subj_xcorr_rt  = {b: [] for b in bands}
n_subjects     = 0

for key_name in [f"sub-{x}" for x in preserved_subj_array]:
    # accumulators for this subject's runs
    run_xcorr_vtc = {b: [] for b in bands}
    run_xcorr_rt  = {b: [] for b in bands}

    for run_name in subj_EEG_dict[key_name].keys():
        if 'gradcpt' not in run_name:
            continue
        run_id = int(run_name.split('cpt')[-1])
        EEG    = subj_EEG_dict[key_name][run_name]
        sfreq  = EEG.info['sfreq']

        if ch not in EEG.ch_names:
            continue
        eeg_data  = EEG.get_data(picks=[ch]).squeeze()
        win_samp  = int(win_sec  * sfreq)
        step_samp = int(step_sec * sfreq)
        n_wins    = (len(eeg_data) - win_samp) // step_samp + 1
        if n_wins < max_lag_sec * 2 + 5:
            continue

        # sliding-window band power
        band_power = {b: np.zeros(n_wins) for b in bands}
        for wi in range(n_wins):
            seg = eeg_data[wi*step_samp : wi*step_samp + win_samp]
            f, psd = sp_sig.welch(seg, fs=sfreq, nperseg=win_samp)
            for b, (flo, fhi) in bands.items():
                band_power[b][wi] = np.mean(psd[(f >= flo) & (f <= fhi)])

        # VTC and interp RT at 1 Hz grid
        event_file = os.path.join(data_save_path, key_name,
                                  f"{key_name}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if not os.path.isfile(event_file):
            continue
        ev_df   = pd.read_csv(event_file, sep='\t')
        vtc_col = 'VTC_smoothed' if 'VTC_smoothed' in ev_df.columns else 'VTC'

        vtc_ts = np.full(n_wins, np.nan)
        rt_ts  = np.full(n_wins, np.nan)
        for _, row in ev_df.iterrows():
            wi = int(row['onset'] // step_sec)
            if 0 <= wi < n_wins:
                vtc_ts[wi] = row[vtc_col]
                rt_ts[wi]  = row['reaction_time']

        # fill NaN windows from neighbours
        for ts in (vtc_ts, rt_ts):
            valid = np.where(~np.isnan(ts))[0]
            if len(valid) < 2:
                break
            ts[np.isnan(ts)] = np.interp(np.where(np.isnan(ts))[0], valid, ts[valid])
        else:
            # interpolate zero RT from non-zero neighbours
            nz, zi = np.where(rt_ts > 0)[0], np.where(rt_ts == 0)[0]
            if len(nz) > 1 and len(zi) > 0:
                rt_ts[zi] = np.interp(zi, nz, rt_ts[nz])

            vtc_z = _zscore(vtc_ts)
            rt_z  = _zscore(rt_ts)
            for b in bands:
                pwr_z = _zscore(band_power[b])
                run_xcorr_vtc[b].append(_run_xcorr(pwr_z, vtc_z, n_wins))
                run_xcorr_rt[b].append( _run_xcorr(pwr_z, rt_z,  n_wins))

    # average across valid runs within this subject
    if all(len(run_xcorr_vtc[b]) > 0 for b in bands):
        for b in bands:
            subj_xcorr_vtc[b].append(np.mean(run_xcorr_vtc[b], axis=0))
            subj_xcorr_rt[b].append( np.mean(run_xcorr_rt[b],  axis=0))
        n_subjects += 1

# --- cross-subject mean ± SEM ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(bands)))

for ax, (acc, lbl) in zip(axes, [(subj_xcorr_vtc, 'VTC'), (subj_xcorr_rt, 'Interp RT')]):
    for (b, (flo, fhi)), col in zip(bands.items(), colors):
        mat  = np.vstack(acc[b])            # (n_subjects, n_lags)
        mean = mat.mean(axis=0)
        sem  = mat.std(axis=0) / np.sqrt(n_subjects)
        ax.plot(lag_axis, mean, label=f'{b} ({flo}–{fhi} Hz)', color=col, linewidth=2)
        ax.fill_between(lag_axis, mean - sem, mean + sem, color=col, alpha=0.2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='-',  linewidth=1)
    ax.set_xlabel('Lag (s)  [+ = EEG power leads]')
    ax.set_ylabel('Pearson r')
    ax.set_title(f'EEG band power ~ {lbl}  ({ch.upper()})')
    ax.legend(title='Band')
    ax.grid(True, alpha=0.3)
fig.suptitle(f'Cross-correlation: EEG band power vs VTC / Interp RT  (n={n_subjects} subjects)')
plt.show()