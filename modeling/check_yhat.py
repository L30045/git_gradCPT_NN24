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
import statsmodels.api as sm
import scipy
import cedalion
import statsmodels.api as sm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.xrutils as xrutils

#%% settings
subj_id = 695
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")

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

#%% open my_fit
ts = y_true
design_matrix = dm_dict['onlyStim']
autoReg = None
ar_order = 30
verbose = False

ts = ts.pint.dequantify()

dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


reg_results = xr.DataArray(
    np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
    dims=("channel", dim3_name),
    coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
)
autoReg_dict = dict()

for (
    dim3,
    group_channels,
    group_design_matrix,
) in design_matrix.iter_computational_groups(ts):
    group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
        "time", "channel"
    )

    # pass x as a DataFrame to statsmodel to make it aware of regressor names
    x = pd.DataFrame(
        group_design_matrix.values, columns=group_design_matrix.regressor.values
    )

    for chan in tqdm(group_y.channel.values, disable=not verbose):
        result = cedalion.math.ar_irls.ar_irls_GLM(group_y.loc[:, chan], x, pmax=ar_order)
        reg_results.loc[chan, dim3] = result

description='AR_IRLS' # FIXME
reg_results.attrs["description"] = description
betas = reg_results.sm.params

 # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
y_hat_ori_stim = (design_matrix.common * betas).sum('regressor')
y_hat_ori_stim = y_hat_ori_stim.transpose('chromo', 'channel', 'time')

#%% retrain stim-only model using original cedalion ar_irls_GLM instead of my_ar_irls_GLM
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)
y_pred = y_hat_ori_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Cedalion AR-IRLS', alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% train model using my_ar_irls_GLM
# stim only model
stim_results, stim_ar = model.my_fit(y_true, dm_dict['onlyStim'], autoReg=None)
betas = stim_results.sm.params
y_hat = (dm_dict['onlyStim'].common * betas).sum('regressor')
y_hat_my_stim = y_hat.transpose('chromo', 'channel', 'time')

#%% Using AR from Full
full_results, full_ar = model.my_fit(y_true, dm_dict['full'], autoReg=None)
betas = full_results.sm.params
y_hat = (dm_dict['full'].common * betas).sum('regressor')
y_hat_full = y_hat.transpose('chromo', 'channel', 'time')

stim_results, _ = model.my_fit(y_true, dm_dict['onlyStim'], autoReg=full_ar)
betas = stim_results.sm.params
y_hat = (dm_dict['onlyStim'].common * betas).sum('regressor')
y_hat_my_stim_full_ar = y_hat.transpose('chromo', 'channel', 'time')

#%% compare stim-only, EEG-only, and full model y_hat for a single channel
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)

y_pred = y_hat_my_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='My Stim', alpha=0.7)

y_pred = y_hat_ori_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Cedalion Stim', alpha=0.7)

y_pred = y_hat_full.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='My Full', alpha=0.7)


y_pred = y_hat_my_stim_full_ar.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label='Full-AR Stim', alpha=0.7)


plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
