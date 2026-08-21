#%% load library
import numpy as np
import pickle
import glob
import os
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from params_setting import *

#%% select model type
eeg_reg_type = 'cont_EEG_cz'
is_hp_fNIRS = False # If True, highpass fNIRS by 1/len_delay (Hz)
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'

#load betas for all subjects
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')))

subj_betas = dict()
for f in betas_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if f'sub-{m.group(1)}' in excluded_subj:
        continue
    with open(f, 'rb') as fh:
        betas_dict = pickle.load(fh)
        len_delay = len(betas_dict['betas_bspline']['component'])  # Delay time in HRF (sec); must match run_model_cont_EEG_fNIRS.py
        subj_betas[subject] = betas_dict['betas_eeg']

# group parcels by network (first '_'-delimited token in the parcel name), excluding the medial-wall background label
parcel_names = [p for p in next(iter(subj_betas.values())).parcel.values if not p.startswith('Background+FreeSurfer')]
networks = sorted(set(p.split('_')[0] for p in parcel_names))

# for each network, average across parcels within a subject, then summarize across subjects (mean and 95% CI)
n_cols = 3
n_rows = int(np.ceil(len(networks) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
axs = np.atleast_1d(axs).flatten()
for ax, net in zip(axs, networks):
    net_parcels = [p for p in parcel_names if p.split('_')[0] == net]
    # subjects x delay
    subj_HRF_net = np.stack([
        betas.sel(parcel=net_parcels).mean('parcel').values.flatten()
        for betas in subj_betas.values()
    ])

    n_subj = subj_HRF_net.shape[0]
    mean_HRF = subj_HRF_net.mean(axis=0)
    sem_HRF = stats.sem(subj_HRF_net, axis=0)
    ci95 = sem_HRF * stats.t.ppf(0.975, n_subj - 1)

    x = np.arange(len(mean_HRF)) * (len_delay / len(mean_HRF))
    ax.plot(x, mean_HRF)
    ax.fill_between(x, mean_HRF - ci95, mean_HRF + ci95, alpha=0.3)
    ax.set_title(net)
    ax.grid()
for ax in axs[len(networks):]:
    ax.set_visible(False)
fig.supxlabel('Time (s)')
fig.supylabel('Beta (HRF estimate)')
fig.suptitle(f'Average HRF per network across subjects (n={len(subj_betas)})')
plt.tight_layout()
plt.show()

# %%
