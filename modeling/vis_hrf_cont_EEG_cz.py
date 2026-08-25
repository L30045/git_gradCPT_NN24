#%% load library
import numpy as np
import pickle
import glob
import os
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
# from params_setting import *
import xarray as xr
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices


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

#%% for each network, average across parcels within a subject, then summarize across subjects (mean and 95% CI)
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

#%% visualize indivisual subject's HRF
# same layout as above, but one figure per subject: solid line is the average
# across parcels within a network, shaded area is the 95% CI across those parcels
for subject, betas in subj_betas.items():
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).flatten()
    for ax, net in zip(axs, networks):
        net_parcels = [p for p in parcel_names if p.split('_')[0] == net]
        # parcels x delay
        parcel_HRF_net = betas.sel(parcel=net_parcels).values.reshape(len(net_parcels), -1)

        n_parcel = parcel_HRF_net.shape[0]
        mean_HRF = parcel_HRF_net.mean(axis=0)
        sem_HRF = stats.sem(parcel_HRF_net, axis=0)
        ci95 = sem_HRF * stats.t.ppf(0.975, n_parcel - 1)

        x = np.arange(len(mean_HRF)) * (len_delay / len(mean_HRF))
        ax.plot(x, mean_HRF)
        ax.fill_between(x, mean_HRF - ci95, mean_HRF + ci95, alpha=0.3)
        ax.set_title(net)
        ax.grid()
    for ax in axs[len(networks):]:
        ax.set_visible(False)
    fig.supxlabel('Time (s)')
    fig.supylabel('Beta (HRF estimate)')
    fig.suptitle(f'{subject}: average HRF per network across parcels')
    plt.tight_layout()
    plt.show()

#%% gather only parcel with a maximum peak around 5 seconds
# same layout/summary as line 36 (mean + 95% CI across subjects), but restricted to
# parcels whose peak beta falls within 1s of t=5s; each subject's own mean (within
# their qualifying parcels) is overlaid and labeled 'sub-xxx (n parcels)'
peak_time = 5.0
peak_tol = 2

fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
axs = np.atleast_1d(axs).flatten()
for ax, net in zip(axs, networks):
    net_parcels = [p for p in parcel_names if p.split('_')[0] == net]

    subj_HRF_net = []
    for subject, betas in subj_betas.items():
        parcel_HRF_net = betas.sel(parcel=net_parcels).values.reshape(len(net_parcels), -1)
        x = np.arange(parcel_HRF_net.shape[1]) * (len_delay / parcel_HRF_net.shape[1])

        peak_t = x[np.argmax(abs(parcel_HRF_net), axis=1)]
        is_peak_near_5s = np.abs(peak_t - peak_time) <= peak_tol
        if not np.any(is_peak_near_5s):
            continue

        subj_mean_HRF = parcel_HRF_net[is_peak_near_5s].mean(axis=0)
        subj_HRF_net.append(subj_mean_HRF)
        ax.plot(x, subj_mean_HRF, alpha=0.5, linewidth=1, label=f'{subject} ({is_peak_near_5s.sum()}/{len(net_parcels)})')

    subj_HRF_net = np.stack(subj_HRF_net)
    n_subj = subj_HRF_net.shape[0]
    mean_HRF = subj_HRF_net.mean(axis=0)
    sem_HRF = stats.sem(subj_HRF_net, axis=0)
    ci95 = sem_HRF * stats.t.ppf(0.975, n_subj - 1)

    ax.plot(x, mean_HRF, color='k', linewidth=2)
    ax.fill_between(x, mean_HRF - ci95, mean_HRF + ci95, color='k', alpha=0.3)
    ax.set_title(net)
    ax.legend(fontsize=6)
    ax.grid()
for ax in axs[len(networks):]:
    ax.set_visible(False)
fig.supxlabel('Time (s)')
fig.supylabel('Beta (HRF estimate)')
fig.suptitle(f'Average HRF per network across subjects, parcels peaking near {peak_time:g}s')
plt.tight_layout()
plt.show()

#%% visualize group-average parcel HRF on the brain surface
# broadcast each parcel's group-average beta at a chosen delay time onto the
# ICBM152 brain surface vertices and render with cedalion's image reconstruction plots
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

# group-average beta per parcel (subjects x parcel x delay -> mean over subjects)
subj_betas_parcel = np.stack([betas.sel(chromo='HbO').values for betas in subj_betas.values()])  # subj x parcel x delay
mean_betas_parcel = subj_betas_parcel.mean(axis=0)  # parcel x delay
parcel_values = next(iter(subj_betas.values())).parcel.values
delay_x = np.arange(mean_betas_parcel.shape[1]) * (len_delay / mean_betas_parcel.shape[1])

plot_time = 5.0  # seconds; delay time point to display
delay_idx = np.argmin(np.abs(delay_x - plot_time))

beta_by_parcel = dict(zip(parcel_values, mean_betas_parcel[:, delay_idx]))
vertex_vals = np.array([beta_by_parcel.get(p, np.nan) for p in vertex_parcel])

X_surf = xr.DataArray(
    np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)

clim_max = np.nanmax(np.abs(vertex_vals))
group_plot_params = dict(
    X_ts=X_surf, cmap='seismic', clim=(-clim_max, clim_max),
    view_type='hbo_brain',
    title_str=f'Group-average HRF beta at t={plot_time:g}s (n={len(subj_betas)})',
    SAVE=False, wdw_size=(1600, 800),
)
image_recon_multi_view(head=head, **group_plot_params)

#%% same plot, but for sub-723 only
select_subj = 'sub-723'
beta_file = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-723/sub-723_cont_EEG_cz_ar_irls_noHp_betas.pkl'
# beta_file = 'sub-723_cont_EEG_cz_ar_irls_noHp_betas.pkl'
with open(beta_file, 'rb') as fh:
    betas_dict = pickle.load(fh)
    len_delay = len(betas_dict['betas_bspline']['component'])  # Delay time in HRF (sec); must match run_model_cont_EEG_fNIRS.py
    subj_beta = betas_dict['betas_eeg']

delay_x = np.arange(mean_betas_parcel.shape[1]) * (len_delay / mean_betas_parcel.shape[1])
plot_time = 5.0  # seconds; delay time point to display
delay_idx = np.argmin(np.abs(delay_x - plot_time))
parcel_values = subj_beta.parcel.values

subj_beta = subj_beta.sel(chromo='HbO').values  # parcel x delay
beta_by_parcel_1 = dict(zip(parcel_values, subj_beta[:, delay_idx]))
vertex_vals_1 = np.array([beta_by_parcel_1.get(p, np.nan) for p in vertex_parcel])

X_surf_1 = xr.DataArray(
    np.stack([vertex_vals_1, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)

clim_max_1 = np.nanmax(np.abs(vertex_vals_1))
subj_plot_params = dict(
    X_ts=X_surf_1, cmap='seismic', clim=(-clim_max_1, clim_max_1),
    view_type='hbo_brain',
    title_str=f'{select_subj} HRF beta at t={plot_time:g}s',
    SAVE=False, wdw_size=(1600, 800),
)
image_recon_multi_view(head=head, **subj_plot_params)

#%% save the plot parameters
# `head` is shared across plots and stored once to avoid duplicating the full mesh
xsubj_dir = os.path.join(project_path, 'derivatives', 'eeg', 'xSubj_results')
out_path = os.path.join(xsubj_dir, f'{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_brain_surface_plot_params.pkl')
with open(out_path, 'wb') as fh:
    pickle.dump({
        'head': head,
        'plots': {'group': group_plot_params, select_subj: subj_plot_params},
    }, fh)
print(f'Saved plot parameters to {out_path}')

#%% Vis HRF on brain surface
image_recon_multi_view(
        X_ts = foo_img_v,
        head = head,
        cmap = 'jet',
        # clim = [-6,6],
        view_type = 'hbo_brain',
        title_str = 'HbO T-stat: in-out',
        filename = None,
        SAVE = False,
        wdw_size = (1300, 768)
    )

