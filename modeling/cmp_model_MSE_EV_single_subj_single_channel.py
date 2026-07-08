#%% load library
import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import model
from params_setting import *

#%% settings
subj_id = 695
model_types = ['full_cedalion', 'onlyStim_cedalion', 'onlyEEG_cedalion', 'basis_cedalion']
model_to_dmkey = {
    'full_cedalion': 'full',
    'onlyStim_cedalion': 'onlyStim',
    'onlyEEG_cedalion': 'onlyEEG',
    'basis_cedalion': 'basis',
}

#%% load design matrices and dependent variable (Y_all, as in run_model_EEG_inform.py)
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")
with open(os.path.join(save_file_path, 'dm_dict.pkl'), 'rb') as f:
    dm_dict = pickle.load(f)

Y_all = dm_dict['Y_all']  # dims: chromo, channel, time

#%% for each model, load betas and reconstruct the full trial estimation (Y_hat)
y_hat_dict = dict()
for model_type in model_types:
    pkl_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_{model_type}.pkl')
    with open(pkl_path, 'rb') as f:
        result_dict = pickle.load(f)

    betas = result_dict['betas']  # dims: channel, chromo, regressor
    dm = dm_dict[model_to_dmkey[model_type]]  # .common dims: time, regressor, chromo

    # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
    y_hat = (dm.common * betas).sum('regressor')
    y_hat_dict[model_type] = y_hat.transpose('chromo', 'channel', 'time')

#%% load geo3d / probe montage for scalp plot
chromo = 'HbO'
hbo_file = os.path.join(project_path, f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    preproc_results = pickle.load(f)
all_runs = preproc_results['runs']
geo3d = preproc_results['geo3d']

#%% compute per-channel MSE for each model
mse_dict = dict()
for model_type in model_types:
    resid = Y_all.sel(chromo=chromo).values - y_hat_dict[model_type].sel(chromo=chromo).values
    mse_dict[model_type] = np.mean(resid ** 2, axis=1)  # dims: channel

#%% Topoplot of MSE comparison
model_pairs = [
    ('full_cedalion', 'basis_cedalion'),
    ('onlyStim_cedalion', 'basis_cedalion'),
    ('onlyEEG_cedalion', 'basis_cedalion'),
    ('full_cedalion', 'onlyStim_cedalion'),
    ('onlyEEG_cedalion', 'onlyStim_cedalion'),
    ('full_cedalion', 'onlyEEG_cedalion'),
]

f, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()
for i, (model_a, model_b) in enumerate(model_pairs):
    mse_diff = mse_dict[model_a] / mse_dict[model_b]
    vlim = np.nanpercentile(np.abs(mse_diff), 99)  # robust to outlier channels
    model.scalp_plot(
        all_runs[0]['conc_o'],
        geo3d,
        mse_diff,
        ax=axs[i],
        cmap='RdBu_r',
        vmin=-vlim,
        vmax=vlim,
        optode_labels=False,
        optode_size=6,
        title=f"{model_a} - {model_b}",
    )
plt.tight_layout()
plt.show()


#%% compare models for a single channel: MSE and explained variance (EV)
# select the channel with minimum MSE_Full / MSE_basis, excluding outlier channels (outside 5th-95th percentile)
mse_ratio_full_basis = mse_dict['full_cedalion'] / mse_dict['basis_cedalion']
ratio_low, ratio_high = np.nanpercentile(mse_ratio_full_basis, [5, 95])
mse_ratio_full_basis_clean = np.where(
    (mse_ratio_full_basis < ratio_low) | (mse_ratio_full_basis > ratio_high),
    np.nan,
    mse_ratio_full_basis,
)
# channel = Y_all.channel.values[np.nanargmin(mse_ratio_full_basis_clean)]
channel = 'S10D127'
chromo = 'HbO'

y_true = Y_all.sel(channel=channel, chromo=chromo).values
ss_tot = np.sum((y_true - y_true.mean()) ** 2)

print(f"sub-{subj_id}, channel={channel}, chromo={chromo}")
mse_vals = []
ev_vals = []
for model_type in model_types:
    y_pred = y_hat_dict[model_type].sel(channel=channel, chromo=chromo).values
    resid = y_true - y_pred
    mse = np.mean(resid ** 2)
    ev = 1 - np.sum(resid ** 2) / ss_tot
    mse_vals.append(mse)
    ev_vals.append(ev)
    print(f"  {model_type:30s}  MSE={mse:.6e}  EV={ev:.4f}")

#%% bar plots comparing MSE and EV across models for the selected channel
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

axs[0].bar(model_types, mse_vals, color=colors)
axs[0].set_ylabel('MSE')
axs[0].set_title('Mean Squared Error')
axs[0].tick_params(axis='x', rotation=30)

axs[1].bar(model_types, ev_vals, color=colors)
axs[1].set_ylabel('EV')
axs[1].set_title('Explained Variance')
axs[1].tick_params(axis='x', rotation=30)

fig.suptitle(f'sub-{subj_id} channel={channel} ({chromo})')
plt.tight_layout()
plt.show()


#%% scalp plot showing location of the selected channel
plt_scalp = np.where(Y_all.channel.values == channel, 1, np.nan)

fig, ax = plt.subplots(figsize=(5, 5))
model.scalp_plot(
    all_runs[0]['conc_o'],
    geo3d,
    plt_scalp,
    ax=ax,
    cmap='RdBu_r',
    vmin=0,
    vmax=1,
    optode_labels=False,
    optode_size=6,
    title=f"sub-{subj_id} selected channel={channel}",
)
plt.tight_layout()
plt.show()

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

# %%
