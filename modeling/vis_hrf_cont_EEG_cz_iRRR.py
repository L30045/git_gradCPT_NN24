#%% load library
# For each subject fit by run_model_cont_EEG_fNIRS_iRRR.py: plot Y_partial vs Y_hat_eeg for one
# parcel, and the top shared HRF components (SVD of the delay x parcel HRF matrix) next to
# their per-parcel weights on the brain surface. Same plots as check_iRRR.py's visualization cell.
import numpy as np
import pickle
import gzip
import glob
import os
import re
import matplotlib.pyplot as plt
import xarray as xr
from params_setting import *
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

#%% select model type
eeg_reg_type = 'cont_EEG_cz_3-stage'  # must match run_model_cont_EEG_fNIRS_iRRR.py
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 (Hz)
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'
select_parcel = 'DefaultA_PFCd_1_LH'
len_delay = 15 # Delay time in HRF (sec); must match run_model_cont_EEG_fNIRS_iRRR.py
n_comp = 4 # number of shared HRF components to plot
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf')

head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', 'betas', f'sub-*_{eeg_reg_type}_iRRR_{hp_flag}_betas.pkl')))

#%% plot each subject
for betas_file in betas_files:
    subject = re.search(r'(sub-\d+)', os.path.basename(betas_file)).group(1)
    if subject in excluded_subj:
        continue
    print(f"Plotting {subject}")

    with open(betas_file, 'rb') as f:
        betas_dict = pickle.load(f)
    with open(get_stats_path(get_prefix_from_betas_path(betas_file)), 'rb') as f:
        stats_dict = pickle.load(f)
    # stats_dict stores the Y_all/dm_all paths used for fitting; files may have moved since, so
    # rebuild them from the AR-IRLS prefix (dm_all file name minus suffix, in this subject's dir)
    ar_irls_prefix = os.path.join(os.path.dirname(os.path.dirname(betas_file)),
                                  os.path.basename(stats_dict['dm_all_path'])[:-len('_dm_all.pkl.gz')])
    with gzip.open(get_Y_all_path(ar_irls_prefix), 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(get_dm_all_path(ar_irls_prefix), 'rb') as f:
        dm_all = pickle.load(f)

    betas_all = betas_dict['betas']
    betas_eeg = betas_dict['betas_eeg']
    mu = stats_dict['intercept']  # (parcel, 1)
    n_regressor = len(betas_eeg.regressor)
    delay_t = np.arange(n_regressor) * (len_delay / n_regressor)

    subj_plot_dir = os.path.join(plot_dir, subject, f'{eeg_reg_type}_iRRR')
    os.makedirs(subj_plot_dir, exist_ok=True)

    #%% Y_partial vs Y_hat_eeg
    # Y_all was built after the drift and GSR OLS stages, so it is already
    # Y_partial = Y_raw - Y_hat_drift - Y_hat_gsr
    Y_partial = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel').values
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    C = betas_all.sel(chromo=select_chromo, regressor=X_da.regressor.values).transpose('regressor', 'parcel').values
    Y_hat_eeg = X_da.values @ C + mu.T  # (time x parcel)
    ss_res = np.nansum((Y_partial - Y_hat_eeg)**2, axis=0)
    ss_tot = np.nansum((Y_partial - np.nanmean(Y_partial, axis=0))**2, axis=0)
    r2_eeg = 1 - ss_res / ss_tot
    parcel_names = Y_all.parcel.values
    p_i = np.where(parcel_names == select_parcel)[0][0]
    print(f"R2 (EEG): {select_parcel} = {r2_eeg[p_i]:.4f}, median over parcels = {np.nanmedian(r2_eeg):.4f}")

    t_all = Y_all.time.values
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(t_all, Y_partial[:, p_i], 'k', lw=0.8, label='Y_partial')
    ax.plot(t_all, Y_hat_eeg[:, p_i], 'r', lw=0.8, label='Y_hat_eeg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{select_chromo} (M)')
    ax.set_title(f'{subject} {select_parcel}: R2 = {r2_eeg[p_i]:.4f}')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(subj_plot_dir, f'{subject}_{select_parcel}_Y_partial_vs_Y_hat_eeg.png'))
    plt.close(fig)

    #%% top n_comp shared HRF components: SVD of the (delay x parcel) HRF matrix
    hrf_mat = betas_eeg.sel(chromo=select_chromo).transpose('regressor', 'parcel').values
    U, S, Vt = np.linalg.svd(hrf_mat, full_matrices=False)
    hrf_parcel_names = betas_eeg.parcel.values

    # render each component's per-parcel weight (row of Vt) on the brain surface
    surf_paths = []
    for c_i in range(n_comp):
        weight_by_parcel = dict(zip(hrf_parcel_names, Vt[c_i]))
        vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
        clim_max = np.nanmax(np.abs(vertex_vals))
        X_surf = xr.DataArray(
            np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )
        surf_path = os.path.join(subj_plot_dir, f'component{c_i+1}_weight')
        image_recon_multi_view(
            X_ts=X_surf, head=head, cmap='seismic', clim=(-clim_max, clim_max),
            view_type='hbo_brain', title_str=f'Component {c_i+1} weight',
            SAVE=True, filename=surf_path, wdw_size=(1600, 800),
        )
        surf_paths.append(surf_path + '.png')

    fig, axes = plt.subplots(n_comp, 2, figsize=(14, 12),
                             gridspec_kw={'width_ratios': [1, 1.6]})
    for c_i in range(n_comp):
        ax = axes[c_i, 0]
        ax.plot(delay_t, U[:, c_i] * S[c_i], 'b')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_ylabel(f'comp {c_i+1}')
        ax.set_title(f'SV = {S[c_i]:.3g} ({S[c_i]**2 / np.sum(S**2) * 100:.1f}% var)')
        axes[c_i, 1].imshow(plt.imread(surf_paths[c_i]))
        axes[c_i, 1].axis('off')
        axes[c_i, 1].set_title(f'Component {c_i+1} per-parcel weight')
    axes[-1, 0].set_xlabel('Delay (s)')
    fig.suptitle(f'{subject} iRRR shared HRFs (rank = {stats_dict["rank"]})')
    fig.tight_layout()
    fig.savefig(os.path.join(subj_plot_dir, f'{subject}_iRRR_shared_HRF_components.png'))
    # plt.close(fig)
    plt.show()

# %%
#%% plot HRF of a selected parcel (or all networks) from iRRR results
def plot_iRRR_hrf(betas_dict, select_parcel=None, is_network=False, chromo='HbO',
                  len_delay=15, ax=None, label=None, **plot_kwargs):
    """Plot HRFs from an iRRR betas_dict (the *_iRRR_*_betas.pkl content).

    is_network=False: plot the HRF of select_parcel on a single axes.
    is_network=True: ignore select_parcel and plot every network in its own subplot, as the
    mean HRF (+/- SEM across parcels) of all parcels sharing a name prefix (e.g. 'DefaultA').
    Pass the returned ax back in to overlay more results (e.g. other subjects).
    """
    betas_eeg = betas_dict['betas_eeg'].sel(chromo=chromo)
    n_regressor = len(betas_eeg.regressor)
    delay_t = np.arange(n_regressor) * (len_delay / n_regressor)

    if not is_network:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(delay_t, betas_eeg.sel(parcel=select_parcel).values, label=label or select_parcel, **plot_kwargs)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'iRRR HRF: {select_parcel}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Delay (s)')
        ax.set_ylabel(f'{chromo} (M)')
        ax.legend(loc='upper right')
        return ax

    parcel_names = [p for p in betas_eeg.parcel.values if not p.startswith('Background+FreeSurfer')]
    networks = sorted(set(p.split('_')[0] for p in parcel_names))
    if ax is None:
        n_row, n_col = 6, 3
        fig, ax = plt.subplots(n_row, n_col, figsize=(4 * n_col, 2.2 * n_row), sharex=True, sharey=True)
        for a in ax.flat[len(networks):]:
            a.axis('off')
    axes = np.atleast_1d(ax).flat
    for net, a in zip(networks, axes):
        net_parcels = [p for p in parcel_names if p.split('_')[0] == net]
        hrf_net = betas_eeg.sel(parcel=net_parcels).transpose('parcel', 'regressor').values
        hrf = hrf_net.mean(0)
        hrf_sem = hrf_net.std(0, ddof=1) / np.sqrt(len(net_parcels)) if len(net_parcels) > 1 else np.zeros_like(hrf)
        line, = a.plot(delay_t, hrf, label=label, **plot_kwargs)
        a.fill_between(delay_t, hrf - hrf_sem, hrf + hrf_sem, color=line.get_color(), alpha=0.2, lw=0)
        a.axhline(0, color='gray', lw=0.5)
        a.set_title(f'{net} (n={len(net_parcels)})')
        a.grid(True, alpha=0.3)
    ax_arr = np.atleast_2d(ax)
    # x label on the lowest visible subplot of each column (hidden slots sit at the bottom)
    for c_i in range(ax_arr.shape[1]):
        a = [a for a in ax_arr[:, c_i] if a.axison][-1]
        a.xaxis.set_tick_params(labelbottom=True)
        a.set_xlabel('Delay (s)')
    for a in ax_arr[:, 0]:
        a.set_ylabel(f'{chromo} (M)')
    if label is not None:
        ax_arr.flat[0].legend(loc='upper right', fontsize='small')
    return ax

#%% example: parcel HRF (is_network=False) or all network HRFs (is_network=True), one figure per subject
is_network = True
for betas_file in betas_files:
    subject = re.search(r'(sub-\d+)', os.path.basename(betas_file)).group(1)
    if subject in excluded_subj:
        continue
    with open(betas_file, 'rb') as f:
        betas_dict = pickle.load(f)
    plot_iRRR_hrf(betas_dict, select_parcel, is_network=is_network, chromo=select_chromo,
                  len_delay=len_delay)
    fig = plt.gcf()
    if is_network:
        fig.suptitle(f'{subject} iRRR HRF per network')
        fig_name = f'{subject}_iRRR_network_HRF.png'
    else:
        fig.suptitle(subject)
        fig_name = f'{subject}_{select_parcel}_iRRR_HRF.png'
    fig.tight_layout()
    subj_plot_dir = os.path.join(plot_dir, subject, f'{eeg_reg_type}_iRRR')
    os.makedirs(subj_plot_dir, exist_ok=True)
    fig.savefig(os.path.join(subj_plot_dir, fig_name))
    plt.show()

# %%
#%% cross-subject network HRF: per-subject network means, then mean +/- SEM across subjects
net_hrf_list, xsubj_subjects = [], []
for betas_file in betas_files:
    subject = re.search(r'(sub-\d+)', os.path.basename(betas_file)).group(1)
    if subject in excluded_subj:
        continue
    with open(betas_file, 'rb') as f:
        betas_eeg = pickle.load(f)['betas_eeg'].sel(chromo=select_chromo)
    parcel_names = [p for p in betas_eeg.parcel.values if not p.startswith('Background+FreeSurfer')]
    network_of = xr.DataArray([p.split('_')[0] for p in parcel_names], dims='parcel',
                              coords={'parcel': parcel_names}, name='network')
    net_hrf_list.append(betas_eeg.sel(parcel=parcel_names).groupby(network_of).mean('parcel'))
    xsubj_subjects.append(subject)
net_hrf_all = xr.concat(net_hrf_list, dim=xr.DataArray(xsubj_subjects, dims='subject'))  # (subject, network, regressor)
networks = net_hrf_all.network.values
n_subj = len(xsubj_subjects)
n_regressor = len(net_hrf_all.regressor)
delay_t = np.arange(n_regressor) * (len_delay / n_regressor)

n_row, n_col = 6, 3
fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 2.2 * n_row), sharex=True)
for a in axes.flat[len(networks):]:
    a.axis('off')
for net, a in zip(networks, axes.flat):
    hrf_subj = net_hrf_all.sel(network=net).transpose('subject', 'regressor').values
    hrf_mean = np.nanmean(hrf_subj, 0)
    hrf_sem = np.nanstd(hrf_subj, 0, ddof=1) / np.sqrt(n_subj)
    for s_i in range(n_subj):
        a.plot(delay_t, hrf_subj[s_i], color='gray', lw=0.6, alpha=0.6,
               label='subjects' if s_i == 0 else None)
    a.plot(delay_t, hrf_mean, color='C0', lw=2, label=f'mean ± SEM (n={n_subj})')
    a.fill_between(delay_t, hrf_mean - hrf_sem, hrf_mean + hrf_sem, color='C0', alpha=0.25, lw=0)
    a.axhline(0, color='gray', lw=0.5)
    a.set_title(net)
    a.grid(True, alpha=0.3)
for c_i in range(n_col):
    a = [a for a in axes[:, c_i] if a.axison][-1]
    a.xaxis.set_tick_params(labelbottom=True)
    a.set_xlabel('Delay (s)')
for a in axes[:, 0]:
    a.set_ylabel(f'{select_chromo} (M)')
axes.flat[0].legend(loc='upper right', fontsize='small')
fig.suptitle(f'Cross-subject iRRR HRF per network ({eeg_reg_type})')
fig.tight_layout()
xsubj_plot_dir = os.path.join(plot_dir, 'group', f'{eeg_reg_type}_iRRR')
os.makedirs(xsubj_plot_dir, exist_ok=True)
fig.savefig(os.path.join(xsubj_plot_dir, 'xSubj_iRRR_network_HRF.png'))
plt.show()

#%% network HRF of the group iRRR fit (run_model_cont_EEG_fNIRS_iRRR_group.py)
group_betas_file = os.path.join(eeg_der_dir, 'group', f'group_{eeg_reg_type}_iRRR_{hp_flag}_betas.pkl')
with open(group_betas_file, 'rb') as f:
    group_betas_dict = pickle.load(f)
with open(group_betas_file.replace('_betas.pkl', '_stats.pkl'), 'rb') as f:
    group_stats_dict = pickle.load(f)
plot_iRRR_hrf(group_betas_dict, is_network=True, chromo=select_chromo, len_delay=len_delay)
fig = plt.gcf()
fig.suptitle(f'Group iRRR HRF per network ({len(group_stats_dict["subjects"])} subjects, '
             f'rank = {group_stats_dict["rank"]})')
fig.tight_layout()
xsubj_plot_dir = os.path.join(plot_dir, 'group', f'{eeg_reg_type}_iRRR')
os.makedirs(xsubj_plot_dir, exist_ok=True)
fig.savefig(os.path.join(xsubj_plot_dir, 'group_iRRR_network_HRF.png'))
plt.show()

#%% group iRRR betas: every parcel HRF (parcel x delay heatmap, grouped by network) and the
# shared HRF components (SVD of the delay x parcel HRF matrix) with per-parcel weights on the surface
group_betas_eeg = group_betas_dict['betas_eeg'].sel(chromo=select_chromo)
group_rank = int(group_stats_dict['rank'])
n_regressor = len(group_betas_eeg.regressor)
delay_t = np.arange(n_regressor) * (len_delay / n_regressor)

# heatmap: parcels ordered by network (sorted names keep each network's parcels contiguous)
parcel_names = sorted(p for p in group_betas_eeg.parcel.values if not p.startswith('Background+FreeSurfer'))
networks = np.array([p.split('_')[0] for p in parcel_names])
hrf_mat = group_betas_eeg.sel(parcel=parcel_names).transpose('parcel', 'regressor').values
clim_max = np.nanmax(np.abs(hrf_mat))
fig, ax = plt.subplots(1, 1, figsize=(8, 12))
im = ax.imshow(hrf_mat, aspect='auto', cmap='seismic', vmin=-clim_max, vmax=clim_max,
               extent=[delay_t[0], delay_t[-1], len(parcel_names) - 0.5, -0.5], interpolation='nearest')
net_start = np.r_[0, np.where(networks[1:] != networks[:-1])[0] + 1]
net_end = np.r_[net_start[1:], len(networks)]
for b in net_start[1:]:
    ax.axhline(b - 0.5, color='k', lw=0.5)
ax.set_yticks((net_start + net_end - 1) / 2, networks[net_start])
ax.set_xlabel('Delay (s)')
fig.colorbar(im, ax=ax, label=f'{select_chromo} (M)', shrink=0.5)
ax.set_title(f'Group iRRR HRF per parcel (rank = {group_rank})')
fig.tight_layout()
fig.savefig(os.path.join(xsubj_plot_dir, 'group_iRRR_parcel_HRF_heatmap.png'))
plt.show()

# shared components: only the first `rank` singular vectors are non-zero
hrf_mat = group_betas_eeg.transpose('regressor', 'parcel').values
U, S, Vt = np.linalg.svd(hrf_mat, full_matrices=False)
n_plot = max(1, min(n_comp, group_rank))
surf_paths = []
for c_i in range(n_plot):
    weight_by_parcel = dict(zip(group_betas_eeg.parcel.values, Vt[c_i]))
    vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
    clim_max = np.nanmax(np.abs(vertex_vals))
    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    surf_path = os.path.join(xsubj_plot_dir, f'group_component{c_i+1}_weight')
    image_recon_multi_view(
        X_ts=X_surf, head=head, cmap='seismic', clim=(-clim_max, clim_max),
        view_type='hbo_brain', title_str=f'Component {c_i+1} weight',
        SAVE=True, filename=surf_path, wdw_size=(1600, 800),
    )
    surf_paths.append(surf_path + '.png')

fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot), squeeze=False,
                         gridspec_kw={'width_ratios': [1, 1.6]})
for c_i in range(n_plot):
    ax = axes[c_i, 0]
    ax.plot(delay_t, U[:, c_i] * S[c_i], 'b')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel(f'comp {c_i+1}')
    ax.set_title(f'SV = {S[c_i]:.3g} ({S[c_i]**2 / np.sum(S**2) * 100:.1f}% var)')
    axes[c_i, 1].imshow(plt.imread(surf_paths[c_i]))
    axes[c_i, 1].axis('off')
    axes[c_i, 1].set_title(f'Component {c_i+1} per-parcel weight')
axes[-1, 0].set_xlabel('Delay (s)')
fig.suptitle(f'Group iRRR shared HRFs ({len(group_stats_dict["subjects"])} subjects, rank = {group_rank})')
fig.tight_layout()
fig.savefig(os.path.join(xsubj_plot_dir, 'group_iRRR_shared_HRF_components.png'))
plt.show()
