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

#%% remove subjects with number of epoch less than half of the target number of epoch (2700/2)
combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict = remove_subject_by_nb_epochs_preserved(subj_id_array, combine_epoch_dict, combine_vtc_dict, combine_react_dict, in_out_zone_dict)

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



#%% Design matrix and fit model
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
# assume IRF is 1.6 seconds long
design_matrix = []
for t_i in range(800):
    shift_S = np.concatenate([np.zeros(t_i), pseudo_S_eeg[:len(pseudo_S_eeg)-t_i]])
    design_matrix.append(shift_S)
design_matrix = np.stack(design_matrix,axis=1)
# plot 3 columns to verify design_matrix is correct
plt.figure()
plt.plot(design_matrix[:,0],label='t=0')
plt.plot(design_matrix[:,50],label='t=50')
plt.plot(design_matrix[:,100],label='t=100')
plt.legend()
plt.grid()


#%% Evaluation 


