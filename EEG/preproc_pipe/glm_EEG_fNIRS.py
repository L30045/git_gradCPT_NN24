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
select_event = "mnt_correct"
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
    select_event = select_event,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names
)

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

#%% IRF
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
params = [1e-4, 1e-4, 0.1, 0.1]
irf = make_IRF(params, time_vector)
plt.figure()
plt.plot(time_vector, irf)


#%% Evaluation 


