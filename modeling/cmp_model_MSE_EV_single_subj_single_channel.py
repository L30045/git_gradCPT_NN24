#%% load library
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from params_setting import *

#%% settings
subj_id = 695
model_types = ['full_noEEG_rejected_ttest', 'reduced', 'onlyEEG', 'basis']
model_to_dmkey = {
    'full_noEEG_rejected_ttest': 'full',
    'reduced': 'onlyStim',
    'onlyEEG': 'onlyEEG',
    'basis': 'basis',
}

#%% load design matrices and dependent variable (Y_all, as in run_model_EEG_inform.py)
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")
with open(os.path.join(save_file_path, 'dm_dict.pkl'), 'rb') as f:
    dm_dict = pickle.load(f)

Y_all = dm_dict['Y_all']  # dims: chromo, channel, time

#%% for each model, load betas and reconstruct the full trial estimation (Y_hat)
y_hat_dict = dict()
for model_type in model_types:
    if model_type == 'full_noEEG_rejected_ttest':
        pkl_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_{model_type}_20260706.pkl')
    else:
        pkl_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_{model_type}.pkl')
    with open(pkl_path, 'rb') as f:
        result_dict = pickle.load(f)

    betas = result_dict['betas']  # dims: channel, chromo, regressor
    dm = dm_dict[model_to_dmkey[model_type]]  # .common dims: time, regressor, chromo

    # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
    y_hat = (dm.common * betas).sum('regressor')
    y_hat_dict[model_type] = y_hat.transpose('chromo', 'channel', 'time')

#%% compare models for a single channel: MSE and explained variance (EV)
channel = Y_all.channel.values[0]
chromo = 'HbO'

y_true = Y_all.sel(channel=channel, chromo=chromo).values
ss_tot = np.sum((y_true - y_true.mean()) ** 2)

print(f"sub-{subj_id}, channel={channel}, chromo={chromo}")
for model_type in model_types:
    y_pred = y_hat_dict[model_type].sel(channel=channel, chromo=chromo).values
    resid = y_true - y_pred
    mse = np.mean(resid ** 2)
    ev = 1 - np.sum(resid ** 2) / ss_tot
    print(f"  {model_type:30s}  MSE={mse:.6e}  EV={ev:.4f}")

#%% plot true vs predicted time courses for the selected channel
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=1)
for model_type in model_types:
    y_pred = y_hat_dict[model_type].sel(channel=channel, chromo=chromo).values
    ax.plot(Y_all.time.values, y_pred, label=model_type, alpha=0.7)
ax.set_xlabel('time (s)')
ax.set_ylabel(f'{chromo} concentration')
ax.set_title(f'sub-{subj_id} channel={channel}')
ax.legend()
plt.tight_layout()
plt.show()
