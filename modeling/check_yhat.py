#%% load library
import numpy as np
import pickle
import copy
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re

#%% settings
subj_id = 695
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")

model_types = ['full_noEEG_rejected_ttest', 'reduced', 'onlyEEG']
model_to_dmkey = {
    'full_noEEG_rejected_ttest': 'full',
    'reduced': 'onlyStim',
    'onlyEEG': 'onlyEEG',
}

#%% load design matrices and dependent variable (Y_all, as in model_EEG_inform_single_subject.py)
with open(os.path.join(save_file_path, 'dm_dict.pkl'), 'rb') as f:
    dm_dict = pickle.load(f)

Y_all = dm_dict['Y_all']  # dims: chromo, channel, time
y_true = Y_all.sel(channel=[debug_channel])

#%% retrain stim and EEG only
ar_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_full_noEEG_rejected_ttest_20260706.pkl')
with open(ar_path, 'rb') as f:
    old_full_result = pickle.load(f)
    autoReg_dict = old_full_result['autoReg_dict']

beta_dict = dict()

#%% full
full_result, full_ar = model.my_fit(y_true, dm_dict['full'], autoReg=None)
beta_dict['full_noEEG_rejected_ttest'] = full_result['betas']

#%% stim only model
stim_results, stim_ar = model.my_fit(y_true, dm_dict[model_to_dmkey['reduced']], autoReg=None)
beta_dict['reduced'] = stim_results.sm.params
# EEG only model
eeg_results, eeg_ar = model.my_fit(y_true, dm_dict[model_to_dmkey['onlyEEG']], autoReg=None)
beta_dict['onlyEEG'] = eeg_results.sm.params

#%% calculate y_hat
for model_type in model_types:
    betas = beta_dict[model_type]  # dims: channel, chromo, regressor
    dm = dm_dict[model_to_dmkey[model_type]]  # .common dims: time, regressor, chromo

    # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
    y_hat = (dm.common * betas).sum('regressor')
    y_hat_dict[model_type] = y_hat.transpose('chromo', 'channel', 'time')

#%% compare stim-only, EEG-only, and full model y_hat for a single channel
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)
for model_type in model_types:
    y_pred = y_hat_dict[model_type].sel(channel=debug_channel, chromo=debug_chromo).values
    plt.plot(Y_all.time.values, y_pred, label=model_type, alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
