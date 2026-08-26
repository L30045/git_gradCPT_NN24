#%% load library
import numpy as np
import pickle
import glob
import os
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from params_setting import *
import xarray as xr
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices
from matplotlib.colors import ListedColormap

#%% select model type
eeg_reg_type = 'cont_EEG_city_mnt_cz'
is_hp_fNIRS = False # If True, highpass fNIRS by 1/len_delay (Hz)
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
plot_dir = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/HRF_surf'
len_delay = 15  # Delay time in HRF (sec); must match run_model_cont_EEG_city_mnt.py
trial_type_prefixes = ['city', 'mnt']  # must match run_model_cont_EEG_city_mnt.py

#load betas for all subjects
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')))

# subj_betas_by_type[trial_type][subject] = betas_eeg (parcel x chromo x delay)
subj_betas_by_type = {tt: dict() for tt in trial_type_prefixes}
for f in betas_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if f'sub-{m.group(1)}' in excluded_subj:
        continue
    with open(f, 'rb') as fh:
        betas_dict = pickle.load(fh)
        for tt in trial_type_prefixes:
            subj_betas_by_type[tt][subject] = betas_dict['betas_eeg_per_type'][tt]

# group parcels by network (first '_'-delimited token in the parcel name), excluding the medial-wall background label
parcel_names = [p for p in next(iter(subj_betas_by_type[trial_type_prefixes[0]].values())).parcel.values if not p.startswith('Background+FreeSurfer')]
networks = sorted(set(p.split('_')[0] for p in parcel_names))

#%% for each network, average across parcels within a subject, then summarize across subjects (mean and 95% CI)
# one figure per trial type
n_cols = 3
n_rows = int(np.ceil(len(networks) / n_cols))
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
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
    fig.suptitle(f'{tt}: average HRF per network across subjects (n={len(subj_betas)})')
    plt.tight_layout()
    plt.show()

#%% visualize indivisual subject's HRF
# same layout as above, but one figure per subject per trial type: solid line is the
# average across parcels within a network, shaded area is the 95% CI across those parcels
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
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
        fig.suptitle(f'{subject} ({tt}): average HRF per network across parcels')
        plt.tight_layout()
        plt.show()

#%% function to snapshot parcel HRF activity on the brain surface at given time points
def snapshot_HRF_surf(betas_parcel, parcel_values, delay_x, snap_times, label, out_dir,
                       head=head, vertex_parcel=vertex_parcel, n_vertex=n_vertex):
    """Render one brain-surface snapshot at each requested time point.

    Args:
        betas_parcel: array (parcel x delay) of HbO beta values, e.g. a
            group-average or single-subject slice of betas_eeg.
        parcel_values: parcel labels aligned with betas_parcel's parcel axis.
        delay_x: delay time (s) for each column of betas_parcel.
        snap_times: time points (s) to snapshot; each is matched to the nearest
            column of betas_parcel via delay_x.
        label: used in the plot title and output filenames.
        out_dir: directory where per-second PNGs are saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt_vertex_vals = dict()
    clim_max = -np.inf
    for t in snap_times:
        delay_idx = np.argmin(np.abs(delay_x - t))
        beta_by_parcel = dict(zip(parcel_values, betas_parcel[:, delay_idx]))
        vertex_vals = np.array([beta_by_parcel.get(p, np.nan) for p in vertex_parcel])
        plt_vertex_vals[t] = vertex_vals
        loc_clim_max = np.nanmax(vertex_vals)
        clim_max = np.max([clim_max,loc_clim_max])

    clim_min = -clim_max
    for t in snap_times:
        X_surf = xr.DataArray(
            np.stack([plt_vertex_vals[t], np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )

        image_recon_multi_view(
            X_ts=X_surf, head=head, cmap='seismic', clim=(clim_min, clim_max),
            view_type='hbo_brain',
            title_str=f'{label} HRF beta at t={t:g}s',
            SAVE=True, filename=os.path.join(out_dir, f'{label}_t{t:g}s'),
            wdw_size=(1600, 800),
        )

#%% visualize group-average parcel HRF on the brain surface, snapshotted every second
# broadcast each parcel's group-average beta at each delay second onto the
# ICBM152 brain surface vertices and render with cedalion's image reconstruction plots
# (one set of snapshots per trial type)
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
    surf_plot_dir = os.path.join(plot_dir, 'group', eeg_reg_type, tt)

    # group-average beta per parcel (subjects x parcel x delay -> mean over subjects)
    subj_betas_parcel = np.stack([betas.sel(chromo='HbO').values for betas in subj_betas.values()])  # subj x parcel x delay
    mean_betas_parcel = subj_betas_parcel.mean(axis=0)  # parcel x delay
    parcel_values = next(iter(subj_betas.values())).parcel.values
    delay_x = np.arange(mean_betas_parcel.shape[1]) * (len_delay / mean_betas_parcel.shape[1])
    snap_times = np.arange(np.ceil(delay_x[0]), np.floor(delay_x[-1]) + 1)

    snapshot_HRF_surf(mean_betas_parcel, parcel_values, delay_x, snap_times, f'group_{tt}', surf_plot_dir)

#%% same snapshots, but for each subject individually (one set of snapshots per trial type)
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
    for select_subj, betas in subj_betas.items():
        surf_plot_dir = os.path.join(plot_dir, select_subj, eeg_reg_type, tt)

        parcel_values = betas.parcel.values
        subj_beta = betas.sel(chromo='HbO').values  # parcel x delay
        delay_x = np.arange(subj_beta.shape[-1]) * (len_delay / subj_beta.shape[-1])
        snap_times = np.arange(np.ceil(delay_x[0]), np.floor(delay_x[-1]) + 1)

        snapshot_HRF_surf(subj_beta, parcel_values, delay_x, snap_times, f'{select_subj}_{tt}', surf_plot_dir)

#%% find the parcels nearest the Cz landmark (shared across subjects: fixed atlas parcellation)
n_near_cz = 5
cz_coord = head.landmarks.sel(label='Cz').pint.dequantify().values
brain_vert_coords = head.brain.vertices.pint.dequantify().values

unique_parcels_no_bg = np.array([p for p in np.unique(vertex_parcel) if not p.startswith('Background+FreeSurfer')])
parcel_centroids = {
    p: brain_vert_coords[vertex_parcel == p].mean(axis=0) for p in unique_parcels_no_bg
}
parcel_dist_to_cz = {p: np.linalg.norm(c - cz_coord) for p, c in parcel_centroids.items()}
near_cz_parcels = sorted(parcel_dist_to_cz, key=parcel_dist_to_cz.get)[:n_near_cz]
print('Parcels nearest Cz:', near_cz_parcels)

#%% for each subject, plot the location and HRF of each near-Cz parcel (per trial type)
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
    for select_subj, betas in subj_betas.items():
        near_cz_dir = os.path.join(plot_dir, select_subj, eeg_reg_type, tt, 'near_Cz')
        os.makedirs(near_cz_dir, exist_ok=True)

        subj_beta = betas.sel(chromo='HbO').values  # parcel x delay
        parcel_values = betas.parcel.values
        delay_x = np.arange(subj_beta.shape[-1]) * (len_delay / subj_beta.shape[-1])
        beta_by_parcel = dict(zip(parcel_values, subj_beta))

        for parcel in near_cz_parcels:
            # location plot: highlight this one parcel on the brain surface
            vertex_mask = (vertex_parcel == parcel).astype(float)
            vertex_mask[vertex_mask == 0] = np.nan
            X_surf_loc = xr.DataArray(
                np.stack([vertex_mask, np.zeros(n_vertex)], axis=-1),
                dims=['vertex', 'chromo'],
                coords={'chromo': ['HbO', 'HbR'],
                        'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
            )
            image_recon_multi_view(
                X_ts=X_surf_loc, head=head, cmap=ListedColormap(['red']), clim=(0, 1),
                view_type='hbo_brain',
                title_str=f'{parcel}',
                SAVE=True, filename=os.path.join(near_cz_dir, f'{parcel}_location'),
                wdw_size=(1600, 800),
            )

            # HRF plot: this parcel's beta timecourse for this subject
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(delay_x, beta_by_parcel[parcel])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Beta (HRF estimate)')
            ax.set_title(f'{select_subj} ({tt}): {parcel}')
            ax.grid()
            plt.tight_layout()
            fig.savefig(os.path.join(near_cz_dir, f'{parcel}_HRF.png'))
            plt.close(fig)

#%% same location and HRF plots for the near-Cz parcels, using the group average (per trial type)
for tt in trial_type_prefixes:
    subj_betas = subj_betas_by_type[tt]
    near_cz_dir = os.path.join(plot_dir, 'group', eeg_reg_type, tt, 'near_Cz')
    os.makedirs(near_cz_dir, exist_ok=True)

    group_betas_parcel = np.stack([betas.sel(chromo='HbO').values for betas in subj_betas.values()])  # subj x parcel x delay
    mean_betas_parcel = group_betas_parcel.mean(axis=0)  # parcel x delay
    parcel_values = next(iter(subj_betas.values())).parcel.values
    delay_x = np.arange(mean_betas_parcel.shape[1]) * (len_delay / mean_betas_parcel.shape[1])
    beta_by_parcel = dict(zip(parcel_values, mean_betas_parcel))

    for parcel in near_cz_parcels:
        # location plot: highlight this one parcel on the brain surface
        vertex_mask = (vertex_parcel == parcel).astype(float)
        vertex_mask[vertex_mask == 0] = np.nan
        X_surf_loc = xr.DataArray(
            np.stack([vertex_mask, np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )
        image_recon_multi_view(
            X_ts=X_surf_loc, head=head, cmap=ListedColormap(['red']), clim=(0, 1),
            view_type='hbo_brain',
            title_str=f'{parcel}',
            SAVE=True, filename=os.path.join(near_cz_dir, f'{parcel}_location'),
            wdw_size=(1600, 800),
        )

        # HRF plot: group-average beta timecourse for this parcel
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(delay_x, beta_by_parcel[parcel])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Beta (HRF estimate)')
        ax.set_title(f'group ({tt}, n={len(subj_betas)}): {parcel}')
        ax.grid()
        plt.tight_layout()
        fig.savefig(os.path.join(near_cz_dir, f'{parcel}_HRF.png'))
        plt.close(fig)
