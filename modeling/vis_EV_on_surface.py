#%% load library
import numpy as np
import pickle
import gzip
import glob
import os
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import explained_variance_score
from params_setting import *
import xarray as xr
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

#%% select model type
eeg_reg_type = 'cont_EEG_cz_add_VTC'
is_hp_fNIRS = False # If True, highpass fNIRS by 1/len_delay (Hz)
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'
plot_dir = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/EV_surf'

#%% for each subject, load betas + (Y_all, dm_all) and compute per-parcel explained variance
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')))

subj_ev = dict()
for f in betas_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if subject in excluded_subj:
        continue

    Y_all_path = f.replace('_betas.pkl', '_Y_all.pkl.gz')
    dm_all_path = f.replace('_betas.pkl', '_dm_all.pkl.gz')
    if not (os.path.exists(Y_all_path) and os.path.exists(dm_all_path)):
        print(f'{subject}: missing Y_all/dm_all files, skipping (rerun run_model_cont_EEG_fNIRS_add_VTC.py with is_overwrite=True).')
        continue

    with open(f, 'rb') as fh:
        betas_all = pickle.load(fh)['betas']  # dims: parcel, chromo, regressor

    with gzip.open(Y_all_path, 'rb') as fh:
        Y_all = pickle.load(fh)  # dims: time, parcel, chromo

    with gzip.open(dm_all_path, 'rb') as fh:
        dm_all = pickle.load(fh)  # .common dims: time, regressor, chromo

    y_hat = xr.dot(dm_all.common, betas_all, dims='regressor')  # dims: time, chromo, parcel

    y_true = Y_all.sel(chromo=select_chromo).transpose('time', 'parcel').values
    y_pred = y_hat.sel(chromo=select_chromo).transpose('time', 'parcel').values
    ev_vals = explained_variance_score(y_true, y_pred, multioutput='raw_values')

    subj_ev[subject] = xr.DataArray(ev_vals, dims='parcel', coords={'parcel': Y_all.parcel.values})

# group parcels by network (first '_'-delimited token in the parcel name), excluding the medial-wall background label
parcel_names = [p for p in next(iter(subj_ev.values())).parcel.values if not p.startswith('Background+FreeSurfer')]
networks = sorted(set(p.split('_')[0] for p in parcel_names))

#%% network summary bar plot: mean EV per network across subjects (error bar = 95% CI across subjects)
os.makedirs(plot_dir, exist_ok=True)
net_mean_by_subj = np.stack([
    [ev.sel(parcel=[p for p in parcel_names if p.split('_')[0] == net]).mean('parcel').item() for net in networks]
    for ev in subj_ev.values()
])  # subjects x networks

n_subj = net_mean_by_subj.shape[0]
net_mean = net_mean_by_subj.mean(axis=0)
net_sem = stats.sem(net_mean_by_subj, axis=0)
net_ci95 = net_sem * stats.t.ppf(0.975, n_subj - 1)

fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(networks)), 5))
ax.bar(networks, net_mean, yerr=net_ci95, capsize=3)
ax.set_ylabel('Explained variance')
ax.set_title(f'Group-average EV per network (n={n_subj})')
ax.tick_params(axis='x', rotation=90)
ax.grid(axis='y')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'group_EV_by_network.png'))
plt.show()

#%% network summary bar plot per subject (error bar = 95% CI across parcels within network)
for subject, ev in subj_ev.items():
    net_mean_subj = []
    net_ci95_subj = []
    for net in networks:
        net_parcels = [p for p in parcel_names if p.split('_')[0] == net]
        parcel_vals = ev.sel(parcel=net_parcels).values
        n_parcel = len(parcel_vals)
        net_mean_subj.append(parcel_vals.mean())
        sem_val = stats.sem(parcel_vals)
        net_ci95_subj.append(sem_val * stats.t.ppf(0.975, n_parcel - 1))

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(networks)), 5))
    ax.bar(networks, net_mean_subj, yerr=net_ci95_subj, capsize=3)
    ax.set_ylabel('Explained variance')
    ax.set_title(f'{subject}: EV per network (across parcels)')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y')
    plt.tight_layout()
    subj_ev_dir = os.path.join(plot_dir, subject, eeg_reg_type)
    os.makedirs(subj_ev_dir, exist_ok=True)
    fig.savefig(os.path.join(subj_ev_dir, f'{subject}_EV_by_network.png'))
    plt.close(fig)

#%% function to render one parcel-wise scalar map (e.g. EV) on the brain surface
def plot_scalar_on_surf(parcel_vals, parcel_values, label, out_path, clim=None,
                         head=head, vertex_parcel=vertex_parcel, n_vertex=n_vertex):
    """Render a single brain-surface snapshot of one scalar value per parcel.

    Args:
        parcel_vals: array of scalar values (e.g. explained variance), aligned
            with parcel_values.
        parcel_values: parcel labels aligned with parcel_vals.
        label: used in the plot title.
        out_path: output filename (without extension) for the saved PNG.
        clim: (vmin, vmax) color limits; defaults to (0, 99th percentile).
    """
    beta_by_parcel = dict(zip(parcel_values, parcel_vals))
    vertex_vals = np.array([beta_by_parcel.get(p, np.nan) for p in vertex_parcel])

    if clim is None:
        clim = (0, np.nanpercentile(vertex_vals, 99))

    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )

    image_recon_multi_view(
        X_ts=X_surf, head=head, cmap='viridis', clim=clim,
        view_type='hbo_brain',
        title_str=f'{label} explained variance',
        SAVE=True, filename=out_path,
        wdw_size=(1600, 800),
    )

#%% group-average EV on the brain surface
group_ev_dir = os.path.join(plot_dir, 'group', eeg_reg_type)
os.makedirs(group_ev_dir, exist_ok=True)

group_ev = np.stack([ev.values for ev in subj_ev.values()]).mean(axis=0)  # parcel
group_parcel_values = next(iter(subj_ev.values())).parcel.values
plot_scalar_on_surf(group_ev, group_parcel_values, f'group (n={n_subj})', os.path.join(group_ev_dir, 'group_EV'))

#%% per-subject EV on the brain surface
for subject, ev in subj_ev.items():
    subj_ev_dir = os.path.join(plot_dir, subject, eeg_reg_type)
    os.makedirs(subj_ev_dir, exist_ok=True)
    plot_scalar_on_surf(ev.values, ev.parcel.values, subject, os.path.join(subj_ev_dir, f'{subject}_EV'))
