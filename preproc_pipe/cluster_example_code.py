#%% 
"""
do the HRF clustering as per David's code
"""
import os
import sys
import gzip
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import xarray as xr
from matplotlib.colors import ListedColormap

from scipy.stats import t

from cedalion.plots import image_recon_multi_view
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/cedalion-dab-funcs/modules/')
import module_image_recon as irf
import module_spatial_basis_funs_ced as sbf

pv.set_jupyter_backend('static')
plt.rcParams['font.size'] = 10

sys.path.append('/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/cedalion_pipeline/')
import gradCPT_funcs as gcpt

import importlib
#%% LOAD IN EVOKED RESPONSES  
ROOT = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
SAVEDIR = os.path.join(ROOT, 'derivatives', 'plots', 'evoked_response')
DATADIR = os.path.join(ROOT, 'derivatives', 'processed_data', 'image_space')

FLAG_DO_SPATIAL_SMOOTHING = True
FLAG_USE_ONLY_SENSITIVE = True
TRIAL_TYPE_TO_CLUSTER = 'mnt-correct'


alpha_spatial = 1e-3
SB = False
alpha_meas = 1e4
direct_name = 'indirect'
Cmeas_name = 'Cmeas'

sigma_brain = 1
sigma_scalp = 5
glm_method = 'ar_irls'

if SB:
    filepath = os.path.join(DATADIR, f'image_hrf_ts_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')
else:
    filepath = os.path.join(DATADIR, f'image_hrf_ts_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{glm_method}.pkl.gz')

with gzip.open(filepath, 'rb') as f:
    image_results = pickle.load(f)

#%% load in the probe 
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/ICBM152/'
Adot = load_Adot(Adot_path  + 'Adot.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
sensitivity_mask = sbf.get_sensitivity_mask(Adot_brain)
sensitive_vertices = np.where(sensitivity_mask)[0]
Adot_sensitive = Adot_brain.sel(vertex=sensitivity_mask)
#%%
name_mapping =  {'VisCent': 'VisC',
                'VisPeri': 'VisP',
                'SomMotA': 'Motor1',
                'SomMotB': 'Motor2',
                'DorsAttnA': 'DAN1',
                'DorsAttnB': 'DAN2',
                'SalVentAttnA': 'VAN1',
                'SalVentAttnB': 'VAN2',
                'LimbicA': 'LimbOFC',
                'LimbicB': 'LimbTempPole',
                'ContA': 'Exec1',
                'ContB': 'Exec2',
                'ContC': 'Exec3',
                'DefaultA': 'DMN1',
                'DefaultB': 'DMN2',
                'DefaultC': 'DMN3',
                'TempPar': 'TempPar',
}

# extract base parcel names (before underscore)
base_parcels = [str(p.values).split('_')[0] for p in Adot_brain.parcel]

# apply name mapping (default to original name if not in mapping)
mapped_parcels = [name_mapping.get(p, p) for p in base_parcels]

# assign these mapped names as 'parcel' coordinate
sensitivity_mask = sensitivity_mask.assign_coords(parcel=('vertex', mapped_parcels))

if FLAG_USE_ONLY_SENSITIVE:
    parcel_labels = sensitivity_mask.sel(vertex=sensitive_vertices).parcel
else:
    parcel_labels = sensitivity_mask.parcel

#%% load the head 
head, PARCEL_DIR = irf.load_head_model('ICBM152', with_parcels=True)
n_brain_vertices = head.brain.mesh.vertices.shape[0]

# get MNI coordinates
V_ijk = head.brain.mesh.vertices  # shape (N,3)
M = head.t_ijk2ras.values  # shape (4,4)
V_h = np.c_[V_ijk, np.ones((V_ijk.shape[0], 1))]      # (N,4)
V_ras_brain = (V_h @ M.T)[:, :3]                      # (N,3), mm in scanner RAS

if FLAG_USE_ONLY_SENSITIVE:
    V_ras = V_ras_brain[sensitivity_mask, :]
else:
    V_ras = V_ras_brain

#%% get the group average either with or without spatial smoothing 
trials = ['mnt-correct', 'mnt-incorrect']

if FLAG_USE_ONLY_SENSITIVE:
    X_hrf_per_subj = image_results['X_hrf_ts'].sel(trial_type=trials, vertex=sensitive_vertices)
    X_mse_per_subj = image_results['X_mse'].sel(trial_type=trials, vertex=sensitive_vertices)
else:
    X_hrf_per_subj = image_results['X_hrf_ts'].sel(trial_type=trials, vertex=Adot.is_brain.values)
    X_mse_per_subj = image_results['X_mse'].sel(trial_type=trials, vertex=Adot.is_brain.values)

X_template = image_results['X_hrf_ts_weighted'].sel(trial_type=trials, vertex=Adot.is_brain.values)

X_hrf_per_subj = X_hrf_per_subj.transpose('trial_type','subj','chromo','vertex','time')
X_mse_per_subj = X_mse_per_subj.transpose('trial_type','subj','chromo','vertex')

if FLAG_DO_SPATIAL_SMOOTHING:

    sigma_mm = 50.0  # FWHM
    W = gcpt.get_spatial_smoothing_kernel(V_ras, sigma_mm)

    H_global = gcpt.compute_Hglobal_from_PCA(X_hrf_per_subj, X_mse_per_subj, W)
    X_hrf_per_subj_new = X_hrf_per_subj - H_global

    Xgrp, Xgrp_tstat, Xgrp_mse = gcpt.get_weighted_group_average(X_hrf_per_subj_new, X_mse_per_subj)
    Hgrp, _, _ = gcpt.get_weighted_group_average(H_global, X_mse_per_subj)

else:
    # get the group average 
    if FLAG_USE_ONLY_SENSITIVE:
        Xgrp = image_results['X_hrf_ts_weighted'].sel(vertex=sensitive_vertices, trial_type=trials)
        Xgrp_tstat = Xgrp / image_results['X_std_err'].sel(vertex=sensitive_vertices, trial_type=trials)
    else:
        Xgrp = image_results['X_hrf_ts_weighted'].sel(vertex=Adot.is_brain.values, trial_type=trials)
        Xgrp_tstat = Xgrp / image_results['X_std_err'].sel(vertex=Adot.is_brain.values, trial_type=trials)

#%% get the correlation 
# --- Correlation+Proximity Clustering of HRFs (vertices or parcels) ---
# ========= CONFIG =========
CHROM = 'HbO'           # chromophore
DIST_MM = 5.0          # max Euclidean distance (mm) to consider neighbors
TAU = 0.99               # correlation threshold (0..1)
MIN_SIZE = 20           # drop clusters smaller than this
# =========================

Xts = Xgrp.sel(trial_type=TRIAL_TYPE_TO_CLUSTER, chromo=CHROM).values  # (vertex, time)

ROI_VERTICES_CORR = gcpt.do_HRF_clustering(Xts, V_ras, DIST_MM=DIST_MM, 
                                            TAU=TAU, MIN_SIZE=MIN_SIZE)


alpha = 0.05 
n = len(image_results['X_hrf_ts'].subj)
df = n - 1
alpha_bonf = alpha / (len(ROI_VERTICES_CORR) * 30)

# Two-tailed critical t-value
t_crit = t.ppf(1 - alpha / 2, df)
t_crit_bonf = t.ppf(1 - alpha_bonf / 2, df)
print(f"t_crit = {t_crit:.4f}")
print(f"t_crit_bonf = {t_crit_bonf:.4f}")

#%% plot the ROIs
X_to_plot = X_template.sel(trial_type=TRIAL_TYPE_TO_CLUSTER).isel(time=0).pint.dequantify().copy()
X_to_plot[:] = np.nan

for k,(name,idx) in enumerate(ROI_VERTICES_CORR.items(),1):
    if FLAG_USE_ONLY_SENSITIVE:
        idx = sensitive_vertices[idx]
    X_to_plot.loc[:, idx] = k

title =  f'Corr+Prox clusters ({TRIAL_TYPE_TO_CLUSTER}, {CHROM})'
fname = SAVEDIR + f'/ROIs_corrprox_{TRIAL_TYPE_TO_CLUSTER}_{CHROM}_verts_{"SB" if SB else "noSB"}_{glm_method}'

cmap = gcpt.make_tab10_shaded(len(ROI_VERTICES_CORR), n_shades=4)
image_recon_multi_view(X_to_plot, head, cmap=cmap, clim=(1, len(ROI_VERTICES_CORR)),
                        view_type='hbo_brain', title_str=title,
                        filename=fname, SAVE=False, geo3d_plot=None,
                        wdw_size=(2000,1000))

#%% plot HRF for each ROI 
flag_plot_tstat = True
flag_use_weighted = True

n_clusters = len(ROI_VERTICES_CORR)
ncols = int(np.ceil(np.sqrt(n_clusters)))
nrows = int(np.ceil(n_clusters / ncols))

if flag_use_weighted:
    # want to get weighted average across the ROI and then the subject average 
    roi_mean_per_subj = xr.concat(
                    [
                        # select vertices for this ROI
                        (
                            X_hrf_per_subj_new.isel(vertex=idx) *
                            (1 / X_mse_per_subj.isel(vertex=idx))      # weight = 1/variance
                        ).sum('vertex') / (1 / X_mse_per_subj.isel(vertex=idx)).sum('vertex')
                        for idx in ROI_VERTICES_CORR.values()
                    ],
                    dim='roi'
                ).assign_coords(roi=list(ROI_VERTICES_CORR.keys()))
    
    roi_mse_per_subj =  xr.concat(
                    [
                        # select vertices for this ROI
                          1 / (1 / X_mse_per_subj.isel(vertex=idx)).sum('vertex')
                        for idx in ROI_VERTICES_CORR.values()
                    ],
                    dim='roi'
                ).assign_coords(roi=list(ROI_VERTICES_CORR.keys()))
    
    
    # now get the subject average 
    roi_means, roi_tstat_means, roi_mse_total = gcpt.get_weighted_group_average(roi_mean_per_subj, roi_mse_per_subj)
else:
    roi_means = xr.concat(
                        [Xgrp.isel(vertex=idx).mean('vertex') for idx in ROI_VERTICES_CORR.values()],
                        dim='roi'
                    )
    roi_means = roi_means.assign_coords({'roi':  list(ROI_VERTICES_CORR.keys())})

    roi_tstat_means = xr.concat(
                        [Xgrp_tstat.isel(vertex=idx).mean('vertex') for idx in ROI_VERTICES_CORR.values()],
                        dim='roi'
                    )
    roi_tstat_means = roi_tstat_means.assign_coords({'roi':  list(ROI_VERTICES_CORR.keys())})


fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True) #, sharey=True)
axes = axes.flatten()
trial_types = ['mnt-correct', 'mnt-incorrect']

for trial_type in trial_types:
    for (roi_name, roi_indices), ax in zip(ROI_VERTICES_CORR.items(), axes):

        if flag_plot_tstat:
            foo = roi_tstat_means.sel(roi=roi_name)
        else:
            foo = roi_means.sel(roi=roi_name)

        ax.plot(foo.time, foo.sel(trial_type=trial_type, chromo='HbO'), label=trial_type)
        ax.set_title(gcpt.roi_title(roi_name, roi_indices, V_ras))
        if flag_plot_tstat:
            ax.set_ylabel('HbO Change (t-stat)')
            ax.axhline(t_crit_bonf, color='r', linestyle='dashed', label=f't_crit_bonf={t_crit_bonf:.2f}')
            ax.axhline(-t_crit_bonf, color='r', linestyle='dashed')
        else:
            ax.set_ylabel('HbO Change (mean)')
        ax.set_xlabel('Time (s)')
        ax.legend()

plt.tight_layout()
# plt.savefig(SAVEDIR + f'/ROIs_hrf_ts_{TRIAL_TYPE_TO_CLUSTER}_{CHROM}_verts_{"SB" if SB else "noSB"}_{glm_method}_{"tstat" if flag_plot_tstat else "mag"}.png', dpi=500)#%%  

mnt_correct_results = {'clusters': ROI_VERTICES_CORR,
                        'roi_mean': roi_means,
                        'roi_tstat': roi_tstat_means,
                        'roi_mse': roi_mse_total}

with open(DATADIR + f'/clusters_trial-{TRIAL_TYPE_TO_CLUSTER}_chromo-{CHROM}.pkl', 'wb') as f:
    pickle.dump(mnt_correct_results, f)

# %%
# get the correlation matrix of the clusters
# choose the condition you want
# TRIAL = 'mnt-correct'
CHROM = 'HbO'
corr_thresh = 0.9

# select (roi, time); drop units if present
da = roi_means.sel(trial_type=TRIAL_TYPE_TO_CLUSTER, chromo=CHROM).pint.dequantify()

groups_idx, corr = gcpt.get_correlation_matrix(da, corr_thresh=corr_thresh)
gcpt.plot_ordered_corr_matrix(groups_idx, corr, corr_thresh=corr_thresh, save_path = SAVEDIR + f'/ROIs_corrmatrix_{TRIAL_TYPE_TO_CLUSTER}_{CHROM}_verts_{"SB" if SB else "noSB"}_{glm_method}.png')


# %%
# Plot Xgrp_roi time courses for a selected connected group
# Options:
PLOT_TRIAL = 'mnt-correct'      
PLOT_CHROM = 'HbO'      # 'HbO' or 'HbR' etc.
GROUP_IDX = 1    # 1-based index into groups_idx
USE_TSTAT = False       # if True and Xgrp_tstat_roi exists, plot t-stats instead of mean
n_groups = len(groups_idx)

assert 1 <= GROUP_IDX <= len(groups_idx), "GROUP_IDX out of range"

# Prepare figure
ncols = int(np.ceil(np.sqrt(n_groups)))
nrows = int(np.ceil(n_groups / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols, 8*nrows), sharex=True) #, sharey=True)
axes = axes.flatten()

for GROUP_IDX in range(n_groups):

    ax = axes[GROUP_IDX]
    roi_group = groups_idx[GROUP_IDX]

    if USE_TSTAT and 'Xgrp_tstat_roi' in globals() and roi_tstat_means is not None:
        da_group = roi_tstat_means.sel(trial_type=PLOT_TRIAL, chromo=PLOT_CHROM).pint.dequantify()
        y_label = f'{PLOT_CHROM} (t-stat)'
    else:
        da_group = roi_means.sel(trial_type=PLOT_TRIAL, chromo=PLOT_CHROM).pint.dequantify()
        y_label = f'{PLOT_CHROM} (group mean)'


    # Plot each ROI in the group and capture line colors
    colors_in_order = []
    for idx in roi_group:
        y = da_group.isel(roi=idx)
        time = da_group.time.values.astype(float)
        line, = ax.plot(time, y, label=f'ROI {idx}')
        colors_in_order.append(line.get_color())

    # Optional: overlay group average
    # group_mean = da_group.isel(roi=roi_group).mean('roi')
    # plt.plot(time, group_mean, linewidth=3, label='Group mean')
    ax.set_title(f'Xgrp_roi time courses â€” Group {GROUP_IDX+1} (n={len(roi_group)}) | {PLOT_TRIAL}, {PLOT_CHROM}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    # Visualize only the ROIs in the selected group, colored to match the time-course plot

    # Build a per-vertex label image with group-local indices 1..K (K = len(roi_group))
    # Map ROI indices -> ROI names to look up vertices
    roi_names_all = list(roi_means.roi.values.astype(str))
    selected_roi_names = [roi_names_all[i] for i in roi_group]

    da_img = Xgrp_tstat.sel(trial_type=PLOT_TRIAL).isel(time=0).copy().pint.dequantify()
    da_img[:] = np.nan

    # Assign 1..K to each selected ROI's vertices in the same order as the time-course plot
    # Colormap that matches the line colors (order preserved)
    cmap_group = ListedColormap(colors_in_order, name='group_colors')

    title_str = f'ROI group {GROUP_IDX+1} locations â€” {PLOT_TRIAL}, {PLOT_CHROM}'
    fname_str = SAVEDIR + f'/ROIs_group{GROUP_IDX+1}_{PLOT_TRIAL}_{PLOT_CHROM}_{"SB" if SB else "noSB"}_{glm_method}'

    da_img = X_template.sel(trial_type=TRIAL_TYPE_TO_CLUSTER).isel(time=0).pint.dequantify().copy()
    da_img[:] = np.nan

    for k, (roi_idx, roi_name) in enumerate(zip(roi_group, selected_roi_names), start=1):
        verts = ROI_VERTICES_CORR[roi_name]
        if FLAG_USE_ONLY_SENSITIVE:
            verts = sensitive_vertices[verts]
        da_img[:, verts] = k


    image_recon_multi_view(
        da_img, head, cmap=cmap_group, clim=(1, len(roi_group)),
        view_type='hbo_brain', title_str=title_str,
        filename=fname_str, SAVE=False, geo3d_plot=None,
        wdw_size=(1500,768)
    )

plt.tight_layout()
# plt.savefig(SAVEDIR + f'/ROIs_subgroup_ts_{TRIAL_TYPE_TO_CLUSTER}_{CHROM}_verts_{"SB" if SB else "noSB"}_{glm_method}.png', dpi=500)

#%% look at the location of the clusters 
# what is the overlap they have with the different anatomical network locations
# parcel_labels = Adot_sensitive.parcel
results = []

for cname, verts in ROI_VERTICES_CORR.items():
    # get parcel labels for those vertices
    cluster_parcels = parcel_labels.sel(vertex=verts).values

    # count how many vertices fall in each parcel
    counts = pd.Series(cluster_parcels).value_counts()

    # turn into a DataFrame
    df = pd.DataFrame({
        "cluster": cname,
        "parcel": counts.index,
        "n_overlap": counts.values,
        "frac_cluster_in_parcel": counts.values / len(verts)
    })

    results.append(df)

cluster_parcel_df = pd.concat(results, ignore_index=True)



heatmap_data = cluster_parcel_df.pivot_table(
    index="cluster", columns="parcel", values="frac_cluster_in_parcel", fill_value=0
)

heatmap_data = heatmap_data.replace(0, np.nan)
# dominant_parcel = heatmap_data.idxmax(axis=1)
# heatmap_data = heatmap_data.loc[dominant_parcel.sort_values().index]


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 12))
cmap = plt.cm.viridis.with_extremes(bad='gray')
sns.heatmap(
    heatmap_data,
    cmap=cmap,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Fraction of cluster in parcel'}
)

plt.title("Clusterâ€“Parcel Overlap")
plt.xlabel("Parcel")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()


# %%
min_frac = 0.6
cluster_name = ["CorrClus_" + str(i) for i in groups_idx[0]]
cluster_rows = cluster_parcel_df[cluster_parcel_df["cluster"].isin(cluster_name)]

subset = cluster_rows[cluster_rows["frac_cluster_in_parcel"] > min_frac]

parcels = subset['parcel'].unique()
# Create subplots (one row per parcel)
fig, axes = plt.subplots(len(parcels)//2, 2, figsize=(14, 8), sharex=True)
axes = axes.flatten()

if len(parcels) == 1:
    axes = [axes]  # Ensure axes is iterable even if only 1 subplot

for ax, parcel in zip(axes, parcels):

    clusters_in_parcel = subset[subset['parcel'] == parcel]['cluster']

    for clus in clusters_in_parcel:
        if flag_plot_tstat:
            foo = roi_tstat_means.sel(roi=clus)
        else:
            foo = roi_means.sel(roi=clus)

        ax.plot(foo.time, foo.sel(trial_type=trial_type, chromo='HbO'), label=clus)

    ax.set_title(parcel)
    ax.legend()

plt.xlabel('Time points')
plt.tight_layout()


# %%
min_frac = 0.49
plt.rcParams['font.size'] = 60
flag_plot_tstat=False
lw=8
all_overlapping = cluster_parcel_df[cluster_parcel_df["frac_cluster_in_parcel"] > min_frac]
parcels = all_overlapping['parcel'].unique()
parcels = ['VisC', 'Motor1', 'Motor2', 'DAN1', 'DAN2',
        'VAN1', 'VAN2', 'Exec1', 'Exec2', 'DMN1', 
        'DMN2', 'DMN3', 'TempPar']

# Create subplots (one row per parcel)
ncols = 4
nrows = int(np.ceil(len(parcels) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(70, 50), sharex=True)
axes = axes.flatten()

if len(parcels) == 1:
    axes = [axes]  # ensure iterable

for ax, parcel in zip(axes, parcels):
    # Subset clusters for this parcel
    parcel_df = all_overlapping[all_overlapping['parcel'] == parcel]
    clusters_in_parcel = parcel_df['cluster']
    weights = parcel_df['n_overlap'].values

    cluster_series = []
    for clus in clusters_in_parcel:
        if flag_plot_tstat:
            foo = roi_tstat_means.sel(roi=clus)
        else:
            foo = roi_means.sel(roi=clus)

        ts = foo.sel(trial_type=trial_type, chromo='HbO').values
        cluster_series.append(ts)
        # ax.plot(foo.time, ts, color='orange',lw =lw,alpha=0.5) #, label=f'{clus}')

    # Stack timeseries and normalize weights
    if len(cluster_series) == 0:
        continue
    cluster_series = np.vstack(cluster_series)
    weights = weights / np.sum(weights)

    # Weighted mean
    weighted_mean = np.average(cluster_series, axis=0, weights=weights)

    # Weighted variance (then SE)
    weighted_var = np.average((cluster_series - weighted_mean)**2, axis=0, weights=weights)
    weighted_se = np.sqrt(weighted_var) / np.sqrt(len(clusters_in_parcel))

    # Plot weighted mean and SE shading
    ax.plot(foo.time, weighted_mean, color='red', linewidth=lw, label='Weighted mean')
    ax.fill_between(
        foo.time,
        weighted_mean - weighted_se,
        weighted_mean + weighted_se,
        color='red',
        alpha=0.2,
        label='Weighted Â±SE'
    )

    ax.set_title(parcel)
    ax.legend(fontsize=20)

plt.tight_layout()
plt.show()# %%

# %%