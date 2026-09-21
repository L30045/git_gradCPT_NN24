#%% Cross-subject clustering of shared HRF shapes (Vt rows) from single_subj_HRF_analysis.py.
# For each subject, load the saved per-subject decomposition (SVD or ICA, set by
# decomp_method below) and take its first N HRF shapes (Vt rows), where N is the rank
# of that subject's Vt. Stack all subjects' HRFs into one (n_subj*N, delay) matrix,
# correlate every pair, convert to a distance matrix. Do the same for each HRF's
# per-parcel surface weight (matching column of U), giving a second distance matrix
# in "where on the brain" space. Combine the two (weighted average) into one distance
# matrix, visualize it, and cluster HRFs by cutting at a correlation threshold of 0.8.
import os
import glob
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

from params_setting import *

#%% key: which decomposition method to load ('svd' or 'ica'; must match a decomp_out_path
# already saved by single_subj_HRF_analysis.py) and the correlation threshold for clustering
decomp_method = 'svd'
assert decomp_method in ('svd', 'ica')
corr_cluster_threshold = 0.7

# how much the combined distance weighs surface location vs. HRF time-course shape
weight_loc = 0.5
weight_t = 1 - weight_loc

eeg_reg_type = 'cont_EEG_cz_3-stage_bspline-test'
is_hp_fNIRS = True
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'

#%% subject list: every subject with a saved HRF decomposition for decomp_method,
# excluding subjects already flagged for low fNIRS quality
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
decomp_files = sorted(glob.glob(os.path.join(
    eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_{decomp_method}_HRF_decomp.pkl')))

subjects = []
for f in decomp_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if subject in excluded_subj:
        continue
    subjects.append(subject)
print(f'Found {len(subjects)} subjects with saved {decomp_method.upper()} HRF decompositions: {subjects}')

#%% for each subject, load Vt and keep its first N rows, where N = rank(Vt); collect
# into one (n_subj*N, delay) matrix, tracking which subject/component each row is from.
# Also collect the matching column of U (that component's per-parcel surface weight)
# into a (n_subj*N, parcel) matrix -- valid only because every subject shares the same
# sensitivity-masked parcel set, in the same order (asserted below).
all_hrfs = []
all_surf_weights = []
row_labels = []  # (subject, component_idx) per row of all_hrfs, same order
t_delay = None
parcel_names_ref = None

for subject in subjects:
    decomp_path = os.path.join(
        eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_{decomp_method}_HRF_decomp.pkl')
    with open(decomp_path, 'rb') as f:
        decomp_out = pickle.load(f)

    Vt = decomp_out['Vt']  # (n_components, delay)
    U = decomp_out['U']    # (parcel, n_components)
    if t_delay is None:
        t_delay = decomp_out['t_delay']
    if parcel_names_ref is None:
        parcel_names_ref = decomp_out['parcel_names']
    else:
        assert np.array_equal(decomp_out['parcel_names'], parcel_names_ref), \
            f'{subject} has a different parcel set/order than the rest -- cannot compare U columns directly'

    N = np.linalg.matrix_rank(Vt)
    all_hrfs.append(Vt[:N])
    all_surf_weights.append(U[:, :N].T)  # (N, parcel)
    row_labels.extend([(subject, i) for i in range(N)])

all_hrfs = np.concatenate(all_hrfs, axis=0)  # (n_subj*N, delay)
all_surf_weights = np.concatenate(all_surf_weights, axis=0)  # (n_subj*N, parcel)
n_rows = all_hrfs.shape[0]
print(f'Collected {n_rows} HRFs total across {len(subjects)} subjects.')

#%% correlate every pair of HRFs' time courses and every pair of HRFs' surface weights,
# each converted to a distance matrix as 1 - correlation (0 = identical, 2 = anti-correlated)
time_dist_mat = 1 - np.corrcoef(all_hrfs)
np.fill_diagonal(time_dist_mat, 0)  # avoid tiny negative values from floating-point error

surf_dist_mat = 1 - np.corrcoef(all_surf_weights)
np.fill_diagonal(surf_dist_mat, 0)

#%% combine the two distance matrices into one, weighting HRF time-course shape vs.
# surface location by weight_t / weight_loc (weight_t = 1 - weight_loc)
dist_mat = weight_t * time_dist_mat + weight_loc * surf_dist_mat

#%% visualize the three n_subj*N x n_subj*N distance matrices (time course, surface
# location, and their weighted combination) side by side, with subject boundaries marked
rows_per_subj = [sum(1 for lbl in row_labels if lbl[0] == s) for s in subjects]
boundary_pos = np.cumsum(rows_per_subj)[:-1] - 0.5

fig, axs = plt.subplots(1, 3, figsize=(24, 8))
mats_titles = [
    (time_dist_mat, f'Time-course distance (weight={weight_t:.2f})'),
    (surf_dist_mat, f'Surface-weight distance (weight={weight_loc:.2f})'),
    (dist_mat, 'Combined distance'),
]
for ax, (mat, title) in zip(axs, mats_titles):
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=2)
    for pos in boundary_pos:
        ax.axhline(pos, color='w', linewidth=0.5, alpha=0.5)
        ax.axvline(pos, color='w', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('HRF index (subject x component)')
    ax.set_ylabel('HRF index (subject x component)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Distance (1 - correlation)', fraction=0.046)

fig.suptitle(f'Cross-subject {decomp_method.upper()} HRF distance matrices (n={n_rows} HRFs, {len(subjects)} subjects)')
plt.tight_layout()
plt.show()

#%% cluster HRFs from the distance matrix: average-linkage hierarchical clustering,
# cut so that clusters merge while their average pairwise correlation stays >= threshold
# (i.e. cut the dendrogram at distance = 1 - corr_cluster_threshold)
condensed_dist = squareform(dist_mat, checks=False)
Z = linkage(condensed_dist, method='average')
cluster_labels = fcluster(Z, t=1 - corr_cluster_threshold, criterion='distance')

n_clusters = len(np.unique(cluster_labels))
print(f'Found {n_clusters} clusters at correlation threshold {corr_cluster_threshold} '
      f'(distance cutoff {1 - corr_cluster_threshold:.2f}).')
for c in np.unique(cluster_labels):
    members = [row_labels[i] for i in np.where(cluster_labels == c)[0]]
    subj_counts = {}
    for s, _ in members:
        subj_counts[s] = subj_counts.get(s, 0) + 1
    print(f'  Cluster {c}: {len(members)} HRFs from {len(subj_counts)} subjects')

#%% visualize the dendrogram with the cluster-cut threshold marked
fig2, ax2 = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax2, color_threshold=1 - corr_cluster_threshold,
           labels=[f'{s}_{i}' for s, i in row_labels], leaf_rotation=90, leaf_font_size=6)
ax2.axhline(1 - corr_cluster_threshold, color='r', linestyle='--',
            label=f'cut at r={corr_cluster_threshold} (dist={1 - corr_cluster_threshold:.2f})')
ax2.set_xlabel('HRF (subject_component)')
ax2.set_ylabel('Combined distance')
ax2.set_title(f'Hierarchical clustering of cross-subject {decomp_method.upper()} HRFs '
              f'(time weight={weight_t:.2f}, surface weight={weight_loc:.2f})')
ax2.legend()
plt.tight_layout()
plt.show()

#%% plot the mean HRF shape (+/- SD) for each cluster
fig3, axs3 = plt.subplots(n_clusters, 1, figsize=(7, 3.5 * n_clusters), sharex=True)
axs3 = np.atleast_1d(axs3)
for ax, c in zip(axs3, np.unique(cluster_labels)):
    members_idx = np.where(cluster_labels == c)[0]
    cluster_hrfs = all_hrfs[members_idx]
    mean_hrf = cluster_hrfs.mean(axis=0)
    sd_hrf = cluster_hrfs.std(axis=0)
    for curve in cluster_hrfs:
        ax.plot(t_delay, curve, color='gray', alpha=0.3, linewidth=0.8)
    ax.plot(t_delay, mean_hrf, color='r', linewidth=2, label=f'mean (n={len(members_idx)})')
    ax.fill_between(t_delay, mean_hrf - sd_hrf, mean_hrf + sd_hrf, color='r', alpha=0.2)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title(f'Cluster {c} (n={len(members_idx)} HRFs)')
    ax.legend()
    ax.grid()
fig3.supxlabel('Delay (s)')
fig3.supylabel(f'{decomp_method.upper()} component (a.u.)')
fig3.suptitle(f'Cluster-mean HRF shapes (threshold r={corr_cluster_threshold})')
plt.tight_layout()
plt.show()

# %%
