#%% load library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from model import *
from tqdm import tqdm
import pickle
import glob
import time
import sys
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
from scipy.optimize import fmin
from sklearn.decomposition import PCA

#%% Test statsmodel
df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df = df.dropna()
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())

#%% preprocessing parameter setting
# subj_id_array = [670, 671, 673, 695]
subj_id_array = [670, 671, 673, 695, 719, 721, 723]
# subj_id_array = [719]
ch_names = ['fz','cz','pz','oz']
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
is_ica_rmEye = True
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names,
    is_overwrite = False
)


#%% Load HRF
example_file = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-695/sub-695_conc_o_results_for_chi.pkl"
with open(example_file,'rb') as f:
    hrf_data = pickle.load(f)
#%% visualize HbO in DorsAttnB
"""
TODO: get the name of parcels of interest. Targeting DorsAttnB for now.
"""
target_parcel = "DorsAttnB"
t_hrf_data = hrf_data['HRF_only'].time.values
hbo_only_all_parcel = hrf_data['HRF_only'].data.magnitude[0,:,:]
roi_index = [x.startswith(target_parcel) for x in hrf_data['HRF_only'].parcel.values]
unique_roi_name = np.unique(['_'.join(x.split('_')[:2]) for x in hrf_data['HRF_only'].parcel.values[roi_index]])

# Plot mean HbO for each unique ROI name
plt.figure(figsize=(12, 6))
for roi_name in unique_roi_name:
    roi_mask = ['_'.join(x.split('_')[:2]) == roi_name for x in hrf_data['HRF_only'].parcel.values[roi_index]]
    hbo_roi = hbo_only_all_parcel[roi_index,:][roi_mask,:]
    mean_hbo = np.mean(hbo_roi, axis=0)
    plt.plot(t_hrf_data, mean_hbo, label=roi_name, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('HbO')
plt.title(f'Mean HbO for each ROI in {target_parcel}')
plt.legend()
plt.grid()
plt.show()

#%% Check continuous time course
hbo_cont_all_parcel = hrf_data["full_timeseries"].data[0,:,:]
print(f"HbO recording length = {hbo_cont_all_parcel.shape[1]/(1/np.diff(t_hrf_data)[0])} second")
print("Time doesn't match with EEG. Need to check time label")
#%% load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(695, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level("sub-695", single_subj_EEG_dict, preproc_params)

#%% combine mnt_correct trials across runs
mnt_correct_all_eeg_chs = []
for key_name in single_subj_epoch_dict.keys():
    mnt_correct_all_eeg_chs.append(single_subj_epoch_dict[key_name]['mnt_correct'].get_data()[:,:4,:])
mnt_correct_all_eeg_chs = np.concatenate(mnt_correct_all_eeg_chs,axis=0)

# Reshape from (trials, channels, times) to (channels, trials * times)
n_trials, n_channels, n_times = mnt_correct_all_eeg_chs.shape
mnt_correct_2d = mnt_correct_all_eeg_chs.transpose(1, 0, 2).reshape(n_channels, n_trials * n_times)

# Calculate PCA along axis 1 (trials * times)
pca = PCA()
pca.fit(mnt_correct_2d.T)  # Transpose so samples are along axis 0
pca_components = pca.components_  # Shape: (n_components, n_channels)
explained_variance = pca.explained_variance_ratio_

# Transform data to get PCA time series: (PCA, trials * times)
pca_timeseries_2d = pca.transform(mnt_correct_2d.T).T  # Shape: (n_components, trials * times)

# Reshape back to 3D: (trials, PCA, times)
n_pca_components = pca_timeseries_2d.shape[0]
pca_timeseries_3d = pca_timeseries_2d.reshape(n_pca_components, n_trials, n_times).transpose(1, 0, 2)

fig, ax = plt.subplots(2,1,figsize=(10,8))
for i in range(4):
    ax[0].plot(np.mean(pca_timeseries_3d,axis=0)[i],label=f"Explained VAR = {explained_variance[i]:.2f}")
ax[0].legend()
ax[0].set_title("Averaged PCA across trials")
pca_90 = np.sum(np.mean(pca_timeseries_3d,axis=0)[:np.where(np.cumsum(explained_variance)>0.9)[0][0]+1],axis=0)
ax[1].plot(pca_90,
            label=f"Sum of 90% var. explained PCA comp. {np.sum(np.cumsum(explained_variance)<=0.9)+1} comp.")
for ch_i, ch in enumerate(ch_names):
    ax[1].plot(np.mean(mnt_correct_all_eeg_chs[:,ch_i,:],axis=0), label=f"mean across trials ({ch})")
ax[1].legend()
ax[1].set_title("Comp. between PC and Ch")
for ax_i in range(2):
    ax[ax_i].grid()
plt.show()

#%% Extract EEG feature for IRF
# Analyze pca_90 for N2 and P3
# Get time vector from the epochs
time_vector_pca = single_subj_epoch_dict[list(single_subj_epoch_dict.keys())[0]]['mnt_correct'].times

# Extract N2 and P3 features
n2_p3_features = extract_n2_p3_features(pca_90, time_vector_pca)

# Visualize results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_vector_pca, pca_90, 'k-', linewidth=2, label='PCA 90% variance')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.axvline(0, color='gray', linestyle='--', linewidth=1)

# Mark zero crossings
for zt in n2_p3_features['zero_crossing_times']:
    ax.axvline(zt, color='lightblue', linestyle=':', alpha=0.5, linewidth=1)

# Mark N2
ax.plot(n2_p3_features['n2_peak_time'], n2_p3_features['n2_peak_amp'],
        'rv', markersize=10, label=f"N2 (t={n2_p3_features['n2_peak_time']*1000:.1f}ms)")
if not np.isnan(n2_p3_features['n2_area']) and n2_p3_features['n2_start_idx'] is not None:
    n2_region = slice(n2_p3_features['n2_start_idx'], n2_p3_features['n2_end_idx']+1)
    ax.fill_between(time_vector_pca[n2_region], 0, pca_90[n2_region],
                    alpha=0.3, color='red', label=f"N2 area={n2_p3_features['n2_area']:.2e}")

# Mark P3
ax.plot(n2_p3_features['p3_peak_time'], n2_p3_features['p3_peak_amp'],
        'b^', markersize=10, label=f"P3 (t={n2_p3_features['p3_peak_time']*1000:.1f}ms)")
if not np.isnan(n2_p3_features['p3_area']) and n2_p3_features['p3_start_idx'] is not None:
    p3_region = slice(n2_p3_features['p3_start_idx'], n2_p3_features['p3_end_idx']+1)
    ax.fill_between(time_vector_pca[p3_region], 0, pca_90[p3_region],
                    alpha=0.3, color='blue', label=f"P3 area={n2_p3_features['p3_area']:.2e}")

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('N2 and P3 Component Analysis')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary
print("="*50)
print("N2 and P3 Component Analysis Summary")
print("="*50)
print(f"Zero crossing times (s): {n2_p3_features['zero_crossing_times']}")
print(f"\nN2 Component:")
print(f"  Peak time: {n2_p3_features['n2_peak_time']*1000:.2f} ms")
print(f"  Peak amplitude: {n2_p3_features['n2_peak_amp']:.4e}")
print(f"  Area: {n2_p3_features['n2_area']:.4e}")
print(f"\nP3 Component:")
print(f"  Peak time: {n2_p3_features['p3_peak_time']*1000:.2f} ms")
print(f"  Peak amplitude: {n2_p3_features['p3_peak_amp']:.4e}")
print(f"  Area: {n2_p3_features['p3_area']:.4e}")
print("="*50)

#%% Create IRF for N2/P3/N2+P3




#%% GLM HbO DorsAttnB = conv(h_HbO, N2/P3/N2+P3)


#%% load epoch for each condition. Epoch from each run is combined for each subject.
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict, (subj_EEG_dict, subj_epoch_dict, subj_vtc_dict, subj_react_dict) = load_epoch_dict(subj_id_array, preproc_params)
# remove subjects with number of epoch less than half of the target number of epoch (2700/2)
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict = remove_subject_by_nb_epochs_preserved(subj_id_array, combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict)

#%% investigate if EEG characteristics still preserve after downsampling to 8Hz
down_sampling_freq = 8 # Hz
select_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ['fz','cz','pz','oz']

# Extract cross-subject ERPs for both conditions
condition_data = {}
for select_event in select_events:
    condition_data[select_event] = dict()
    for ch in vis_ch:
        subj_epoch_array = combine_epoch_dict[select_event][ch]
        n_subjects = len(subj_epoch_array)
        xSubj_erps = []
        for epoch in subj_epoch_array:
            # down sample epoch
            epoch.resample(down_sampling_freq)
            # Get average ERP for this subject
            evoked = epoch.average()
            xSubj_erps.append(evoked.data)
        xSubj_erps = np.vstack(xSubj_erps)
        condition_data[select_event][ch] = {'erps': xSubj_erps, 'n_subjects': n_subjects}

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
        label = select_event.replace('_', ' ').title()
        plt.plot(time_vector, mean_erp, color=colors[idx], linewidth=2, label=f'{label} Mean')
        plt.fill_between(time_vector, lower_bound, upper_bound, alpha=0.3, color=colors[idx])

    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'{ch.upper()} (n={n_subjects})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save figure to fig_save_path
    if is_save_fig:
        save_filename = f'mntC_vs_mntIC_{ch}_mean_2SEM.png'
        plt.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
    plt.show()


#%% project EEG from sensor space to source space
"""
A^+ = (A^T A)^-1 A^T

X_eeg = F dot S_eeg.
X in sensor space. S in source space. F is forward_matrix.
X_eeg (sensors, time points), F (sensors, sources), S_eeg (sources, time points)
S_eeg = (F^T F)^-1 F^T X_eeg.
(F^T F) will be rank deficient so S_eeg will be an estimation that minimize norm2 (F dot S_eeg)

We can reduce the number of sources from all voxel to only voxel in ROI or even the center of ROI.
    case 1 (All voxel): a lot.
    case 2 (voxel in ROI): voxel in 17 ROI.
    case 3 (center of ROI): 17 only.

Thoughts:
1. I feel fitting in sensor domain is weird. HbT = X_eeg dot IRF. 
   This means that each channels has its own IRF. However, these IRFs will be dependent and hard to interpret their physiological meaning.
   Moreover, we cannot superimpose them to get HbT due to their dependency.
2. For source spance, even though we have multiple voxels corresponding to one HbT, I feel it is more reasonable to assume the sources in the same ROI share the same IRF.
   And their effect can be superimosed.
3. Or we just assume each voxel has its own IRF. I am not sure if the rank deficient S_eeg will give us any error.
   However, I assume nearby IRFs should look alike.
"""



#%% Train test split
"""
We only have 6 good subjects for now. Should we do 6-fold cross validation instead?
"""



#%% Design matrix
"""
We are going to have a huge design matrix.
    Solution 1: Huge memory. (Not scalable if we recruit more and more subjects.)
    Solution 2: Update beta online. (Beta might jumps between local minimum.)
Question:
1. How do define the window length of IRF? (In paper, it is 7 second. In our case, we can only do no more than 1.6 seconds.)
   With 1.6 seconds, there will be 800 time points in IRF.
2. How to fit IRF from formula?
"""
tmp_X_eeg = np.concatenate([combine_epoch_dict['city_correct'][key][0].get_data() for key in combine_epoch_dict['city_correct'].keys()], axis=1)
# get pseudo source
pseudo_S_eeg = tmp_X_eeg[0,1,:][-800:]
design_matrix = make_design_matrix(pseudo_S_eeg, 800)
# visual check 
plt.figure()
plt.plot(design_matrix[:,0],color=[0,0,1],label='t=0')
plt.plot(design_matrix[:,50],color=[0,0.5,1],label='t=50')
plt.plot(design_matrix[:,100],color=[0,1,1],label='t=100')
plt.legend()
plt.grid()
plt.show()

#%% constraint IRF
"""
IRF = A * ((t-t0)/tau_D)^3 * exp((t-t0)/tau_D)
    + B * ((t-t0)/tau_C)^3 * exp((t-t0)/tau_C)
"""
time_vector = np.linspace(0, 1.6, int(1.6*500+1))
def make_IRF(params, t):
    """
    params = [A, B, tau_D, tau_C] (I don't think we should fit t0.)
    set t0 = 0
    """
    irf = params[0] * (t/params[2])**3 * np.exp(-t/params[2])\
        + params[1] * (t/params[3])**3 * np.exp(-t/params[3])
    return irf
params = [1, -0.5, 0.1, 0.15]
irf = make_IRF(params, time_vector)
plt.figure()
plt.plot(time_vector, irf)
plt.grid()


#%% Unconstraint IRF
"""
Fit IRF directly without given equation.
"""

#%% Evaluation 


