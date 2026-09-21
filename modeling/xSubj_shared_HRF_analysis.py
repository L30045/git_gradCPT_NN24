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
import xarray as xr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment

import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

from params_setting import *

#%% key: which decomposition method to load ('svd' or 'ica'; must match a decomp_out_path
# already saved by single_subj_HRF_analysis.py) and the correlation threshold for clustering
decomp_method = 'svd'
assert decomp_method in ('svd', 'ica')

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
all_var_explained = []
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
    var_explained = decomp_out['var_explained']  # (n_components,)
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
    all_var_explained.append(var_explained[:N])
    row_labels.extend([(subject, i) for i in range(N)])

all_hrfs = np.concatenate(all_hrfs, axis=0)  # (n_subj*N, delay)
all_surf_weights = np.concatenate(all_surf_weights, axis=0)  # (n_subj*N, parcel)
all_var_explained = np.concatenate(all_var_explained, axis=0)  # (n_subj*N,)
n_rows = all_hrfs.shape[0]
print(f'Collected {n_rows} HRFs total across {len(subjects)} subjects.')

#%% correlate every pair of HRFs' time courses and every pair of HRFs' surface weights,
# each converted to a distance matrix as 1 - correlation (0 = identical, 2 = anti-correlated)
time_dist_mat = 1 - np.abs(np.corrcoef(all_hrfs))
np.fill_diagonal(time_dist_mat, 0)  # avoid tiny negative values from floating-point error

surf_dist_mat = 1 - np.abs(np.corrcoef(all_surf_weights))
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

#%% cluster HRFs from the distance matrix: average-linkage hierarchical clustering.
# Rather than fixing the cut distance, sweep it and pick the value that jointly
# minimizes the number of clusters (favors merging) and the mean within-cluster
# distance (favors splitting) -- two goals that trade off against each other as the
# cut distance grows. For each candidate cut distance, min-max normalize both
# quantities to [0, 1] across the sweep and take the cut that minimizes their sum
# (a simple elbow-style tradeoff, with no free weighting parameter).
condensed_dist = squareform(dist_mat, checks=False)
Z = linkage(condensed_dist, method='average')


def mean_within_cluster_dist(labels):
    """Average, over clusters with >1 member, of that cluster's mean pairwise distance
    (singleton clusters contribute 0 and are excluded, since they have no within-cluster
    pairs to average)."""
    within_dists = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            within_dists.append(dist_mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)].mean())
    return np.mean(within_dists) if within_dists else 0.0


cut_distances = np.linspace(dist_mat[dist_mat > 0].min(), dist_mat.max(), 200)
n_clusters_sweep = []
within_dist_sweep = []
for cut_dist in cut_distances:
    labels_sweep = fcluster(Z, t=cut_dist, criterion='distance')
    n_clusters_sweep.append(len(np.unique(labels_sweep)))
    within_dist_sweep.append(mean_within_cluster_dist(labels_sweep))
n_clusters_sweep = np.array(n_clusters_sweep)
within_dist_sweep = np.array(within_dist_sweep)


def min_max_norm(x):
    x_range = x.max() - x.min()
    return (x - x.min()) / x_range if x_range > 0 else np.zeros_like(x)


score_sweep = min_max_norm(n_clusters_sweep) + min_max_norm(within_dist_sweep)
best_i = np.argmin(score_sweep)
cut_dist_opt = cut_distances[best_i]
corr_cluster_threshold = 1 - cut_dist_opt
print(f'Optimized cut distance = {cut_dist_opt:.3f} (correlation threshold = {corr_cluster_threshold:.3f}): '
      f'{n_clusters_sweep[best_i]} clusters, mean within-cluster distance = {within_dist_sweep[best_i]:.3f}')

#%% visualize the threshold sweep: number of clusters and mean within-cluster distance
# (both normalized) vs. cut distance, with the chosen optimum marked
fig_opt, ax_opt = plt.subplots(figsize=(9, 5))
ax_opt.plot(cut_distances, min_max_norm(n_clusters_sweep), label='n clusters (normalized)')
ax_opt.plot(cut_distances, min_max_norm(within_dist_sweep), label='mean within-cluster distance (normalized)')
ax_opt.plot(cut_distances, score_sweep, label='sum (score to minimize)', color='k', linewidth=2)
ax_opt.axvline(cut_dist_opt, color='r', linestyle='--', label=f'chosen cut = {cut_dist_opt:.3f}')
ax_opt.set_xlabel('Cut distance')
ax_opt.set_ylabel('Normalized value')
ax_opt.set_title('Cluster-cut threshold optimization')
ax_opt.legend()
ax_opt.grid()
plt.tight_layout()
plt.show()

#%% cut the dendrogram at the optimized distance
cluster_labels = fcluster(Z, t=cut_dist_opt, criterion='distance')

n_clusters = len(np.unique(cluster_labels))
print(f'Found {n_clusters} clusters at optimized correlation threshold {corr_cluster_threshold:.3f} '
      f'(distance cutoff {cut_dist_opt:.3f}).')
for c in np.unique(cluster_labels):
    members = [row_labels[i] for i in np.where(cluster_labels == c)[0]]
    subj_counts = {}
    for s, _ in members:
        subj_counts[s] = subj_counts.get(s, 0) + 1
    print(f'  Cluster {c}: {len(members)} HRFs from {len(subj_counts)} subjects')

#%% visualize the dendrogram with the cluster-cut threshold marked
fig2, ax2 = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax2, color_threshold=cut_dist_opt,
           labels=[f'{s}_{i}' for s, i in row_labels], leaf_rotation=90, leaf_font_size=6)
ax2.axhline(cut_dist_opt, color='r', linestyle='--',
            label=f'optimized cut (r={corr_cluster_threshold:.2f}, dist={cut_dist_opt:.2f})')
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
fig3.suptitle(f'Cluster-mean HRF shapes (optimized threshold r={corr_cluster_threshold:.2f})')
plt.tight_layout()
plt.show()

#%% Hungarian-method matching: for a reference subject, find each other subject's best-
# matching HRF, the same way EEGLAB's matcorr.m does -- build the full correlation
# matrix between the two subjects' rows, then solve the linear-sum-assignment problem
# to find the pairing that maximizes total abs(correlation) (Hungarian method requires
# minimizing a cost, so the cost matrix is -abs(corr); scipy's linear_sum_assignment
# handles the rectangular case natively when the two subjects have different N).
# Matching is done three ways: on HRF time-course shape only, on surface weight only,
# and on both jointly (the same weight_t/weight_loc-weighted correlation used for
# dist_mat above). Matched pairs are sorted by abs(corr) descending, as matcorr does.
ref_subject = 'sub-723'
assert ref_subject in subjects, f'{ref_subject} not found among subjects with saved decompositions: {subjects}'

ref_idx = np.array([i for i, (s, _) in enumerate(row_labels) if s == ref_subject])
other_subjects = [s for s in subjects if s != ref_subject]


def hungarian_match(corr_sub, ref_local_idx, other_local_idx):
    """Given the full corr matrix (n_rows, n_rows) and the row-indices belonging to
    the reference subject and one other subject, solve the assignment problem that
    maximizes total abs(correlation) between matched (ref, other) row pairs. Returns
    (ref_i, other_i, corr) arrays, sorted by abs(corr) descending, one entry per
    matched pair (length = min(N_ref, N_other), as in matcorr.m)."""
    sub_corr = corr_sub[np.ix_(ref_local_idx, other_local_idx)]  # (N_ref, N_other)
    row_ind, col_ind = linear_sum_assignment(-np.abs(sub_corr))
    matched_corr = sub_corr[row_ind, col_ind]
    order = np.argsort(-np.abs(matched_corr))
    return ref_local_idx[row_ind[order]], other_local_idx[col_ind[order]], matched_corr[order]


corr_time_full = np.corrcoef(all_hrfs)
corr_surf_full = np.corrcoef(all_surf_weights)
corr_both_full = weight_t * corr_time_full + weight_loc * corr_surf_full

match_methods = {
    'temporal': corr_time_full,
    'spatial': corr_surf_full,
    'both': corr_both_full,
}

# matches[method][other_subject] = (ref_i, other_i, corr), all arrays length
# min(N_ref, N_other), sorted by abs(corr) descending
matches = {method: dict() for method in match_methods}
for method, corr_full in match_methods.items():
    for other_subject in other_subjects:
        other_idx = np.array([i for i, (s, _) in enumerate(row_labels) if s == other_subject])
        matches[method][other_subject] = hungarian_match(corr_full, ref_idx, other_idx)
    n_pairs_report = ', '.join(
        f'{s}: {len(matches[method][s][2])} pairs' for s in other_subjects)
    print(f'Hungarian match ({method}) from {ref_subject}: {n_pairs_report}')

#%% plot the matched HRFs: one figure per matching method (temporal / spatial / both),
# one row per other subject, showing ref_subject's HRF time-course next to its
# best-matching HRF time-course from that subject, labeled with the matched correlation
for method in match_methods:
    n_other = len(other_subjects)
    max_pairs = max(len(matches[method][s][2]) for s in other_subjects)
    fig4, axs4 = plt.subplots(n_other, max_pairs, figsize=(3.5 * max_pairs, 3 * n_other),
                               sharex=True, squeeze=False)
    for row, other_subject in enumerate(other_subjects):
        ref_i, other_i, corr = matches[method][other_subject]
        for col in range(max_pairs):
            ax = axs4[row, col]
            if col >= len(corr):
                ax.set_visible(False)
                continue
            ax.plot(t_delay, all_hrfs[ref_i[col]], color='k', linewidth=1.5,
                    label=f'{ref_subject} #{row_labels[ref_i[col]][1]}')
            ax.plot(t_delay, all_hrfs[other_i[col]], color='r', linewidth=1.5, alpha=0.8,
                    label=f'{other_subject} #{row_labels[other_i[col]][1]}')
            ax.axhline(0, color='gray', linewidth=0.6)
            ax.set_title(f'r={corr[col]:.2f}', fontsize=9)
            ax.legend(fontsize=6)
            ax.grid()
    fig4.supxlabel('Delay (s)')
    fig4.supylabel(f'{decomp_method.upper()} component (a.u.)')
    fig4.suptitle(f'Hungarian-matched HRFs vs. {ref_subject} ({method} matching)')
    plt.tight_layout()
    plt.show()

#%% using sub-723 (ref_subject) as the template, regroup the temporal-matching results
# by ref_subject's component index: for each of its N components, find what each other
# subject's Hungarian-matched HRF (temporal matching) was. Some other subjects may have
# fewer components than ref_subject (min(N_ref, N_other) pairs per hungarian_match call),
# so a given ref component is not guaranteed a match from every other subject.
n_ref_components = len(ref_idx)
# ref_component_matches[i] = {other_subject: (other_component_idx, corr)}
ref_component_matches = {i: dict() for i in range(n_ref_components)}
for other_subject in other_subjects:
    ref_i, other_i, corr = matches['temporal'][other_subject]
    for r_i, o_i, c in zip(ref_i, other_i, corr):
        ref_comp = row_labels[r_i][1]
        other_comp = row_labels[o_i][1]
        ref_component_matches[ref_comp][other_subject] = (other_comp, c, o_i)

#%% for each of ref_subject's components, plot (1) the mean HRF +/- SD across
# ref_subject's own HRF and its temporal-matched HRF from every other subject, (2) a bar
# plot of each other subject's matched component index, and (3) a bar plot of each other
# subject's matched correlation -- one column per component, combined into one figure.
# A matched HRF with negative correlation to ref_subject's HRF is sign-flipped before
# averaging/plotting (it is the same shape inverted), and the correlation bar shows
# abs(corr) to match.
fig5, axs5 = plt.subplots(3, n_ref_components, figsize=(3 * n_ref_components, 9), sharex='row')
for ax in axs5[1, 1:]:
    ax.sharey(axs5[1, 0])
for ax in axs5[2, 1:]:
    ax.sharey(axs5[2, 0])

for ref_comp in range(n_ref_components):
    ax_hrf, ax_comp, ax_corr = axs5[:, ref_comp]
    comp_matches = ref_component_matches[ref_comp]  # {other_subject: (other_comp, corr, other_i)}
    matched_subjects = list(comp_matches.keys())

    group_hrfs = [all_hrfs[ref_idx[ref_comp]]] + [
        np.sign(comp_matches[s][1]) * all_hrfs[comp_matches[s][2]] for s in matched_subjects]
    group_hrfs = np.stack(group_hrfs, axis=0)
    mean_hrf = group_hrfs.mean(axis=0)
    sd_hrf = group_hrfs.std(axis=0)

    for curve in group_hrfs:
        ax_hrf.plot(t_delay, curve, color='gray', alpha=0.3, linewidth=0.8)
    ax_hrf.plot(t_delay, mean_hrf, color='r', linewidth=2, label=f'mean (n={len(group_hrfs)})')
    ax_hrf.fill_between(t_delay, mean_hrf - sd_hrf, mean_hrf + sd_hrf, color='r', alpha=0.2)
    ax_hrf.axhline(0, color='gray', linewidth=0.8)
    ref_var_explained = all_var_explained[ref_idx[ref_comp]]
    ax_hrf.set_title(f'{ref_subject} #{ref_comp} ({ref_var_explained*100:.1f}%)', fontsize=9)
    ax_hrf.legend(fontsize=6)
    ax_hrf.grid()

    matched_comp_idx = [comp_matches[s][0] for s in matched_subjects]
    matched_var_explained = [all_var_explained[comp_matches[s][2]] for s in matched_subjects]
    bars_comp = ax_comp.bar(matched_subjects, matched_comp_idx, color='tab:blue')
    for bar, ve in zip(bars_comp, matched_var_explained):
        ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{ve*100:.1f}%',
                     ha='center', va='bottom', fontsize=5)
    ax_comp.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax_comp.margins(y=0.15)
    ax_comp.grid(axis='y')

    matched_corr = [abs(comp_matches[s][1]) for s in matched_subjects]
    ax_corr.bar(matched_subjects, matched_corr, color='tab:orange')
    ax_corr.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax_corr.grid(axis='y')

    if ref_comp == 0:
        ax_hrf.set_ylabel(f'{decomp_method.upper()} component (a.u.)')
        ax_comp.set_ylabel('Matched component #')
        ax_corr.set_ylabel('|Matched correlation|')

fig5.supxlabel('Delay (s) (top row) / Subject (bottom two rows)')
fig5.suptitle(f'{ref_subject} components: mean HRF and temporal-matched component/correlation per subject')
plt.tight_layout()
plt.show()

#%% plot the median per-parcel U weight (surface loading), across ref_subject's own
# component and its temporal-matched components from every other subject, on the brain
# surface for each template (ref_subject) component. A matched component's U column is
# sign-flipped the same way its HRF was (same sign ambiguity in the underlying
# decomposition), so it is combined consistently with the mean HRF above.

head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

surf_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'components_analysis')
os.makedirs(surf_plot_dir, exist_ok=True)

for ref_comp in range(n_ref_components):
    comp_matches = ref_component_matches[ref_comp]
    matched_subjects = list(comp_matches.keys())

    group_weights = [all_surf_weights[ref_idx[ref_comp]]] + [
        np.sign(comp_matches[s][1]) * all_surf_weights[comp_matches[s][2]] for s in matched_subjects]
    median_weight = np.median(np.stack(group_weights, axis=0), axis=0)

    weight_by_parcel = dict(zip(parcel_names_ref, median_weight))
    vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
    clim_max = np.nanmax(np.abs(vertex_vals))

    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    surf_path = os.path.join(surf_plot_dir, f'{ref_subject}_component{ref_comp}_median_U_weight')
    image_recon_multi_view(
        X_ts=X_surf, head=head, cmap='seismic', clim=(-clim_max, clim_max),
        view_type='hbo_brain',
        title_str=f'{ref_subject} #{ref_comp}: median U weight (n={len(group_weights)})',
        SAVE=True, filename=surf_path,
        wdw_size=(1600, 800),
    )

    # compose the saved surface image with the cross-subject HRF result for this
    # component (ref_subject's own HRF plus its temporal-matched HRF from every other
    # subject, sign-flipped and averaged the same way as the earlier mean-HRF figure)
    # into one figure, since image_recon_multi_view saves its own plot rather than
    # drawing into a given axis
    group_hrfs = [all_hrfs[ref_idx[ref_comp]]] + [
        np.sign(comp_matches[s][1]) * all_hrfs[comp_matches[s][2]] for s in matched_subjects]
    group_hrfs = np.stack(group_hrfs, axis=0)
    mean_hrf = group_hrfs.mean(axis=0)
    sd_hrf = group_hrfs.std(axis=0)

    fig6, (ax_surf, ax_hrf) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    surf_img = plt.imread(surf_path + '.png')
    ax_surf.imshow(surf_img)
    ax_surf.axis('off')

    for curve in group_hrfs:
        ax_hrf.plot(t_delay, curve, color='gray', alpha=0.3, linewidth=0.8)
    ax_hrf.plot(t_delay, mean_hrf, color='r', linewidth=2, label=f'mean (n={len(group_hrfs)})')
    ax_hrf.fill_between(t_delay, mean_hrf - sd_hrf, mean_hrf + sd_hrf, color='r', alpha=0.2)
    ax_hrf.axhline(0, color='gray', linewidth=0.8)
    ax_hrf.set_xlabel('Delay (s)')
    ax_hrf.set_ylabel(f'{decomp_method.upper()} component (a.u.)')
    ref_var_explained = all_var_explained[ref_idx[ref_comp]]
    ax_hrf.set_title(f'{ref_subject} #{ref_comp} cross-subject HRF ({ref_var_explained*100:.1f}%)')
    ax_hrf.legend(fontsize=8)
    ax_hrf.grid()

    fig6.suptitle(f'{ref_subject} #{ref_comp}: median U weight (n={len(group_weights)}) + cross-subject HRF')
    plt.tight_layout()
    plt.savefig(surf_path + '.png', dpi=150)
    plt.show()
    plt.close(fig6)
    print(f'Saved {surf_path}.png')

#%% for each template (ref_subject) component, compare its median per-parcel weight
# against a DorsAttn-vs-Default network template over all parcels: template = -1 for
# DorsAttn parcels, +1 for Default parcels, 0 for every other-network parcel (no
# masking -- all parcels participate in the correlation).
is_dorsattn = np.array([p.startswith('DorsAttn') for p in parcel_names_ref])
is_default = np.array([p.startswith('Default') for p in parcel_names_ref])
network_template = np.where(is_dorsattn, -1, np.where(is_default, 1, 0))
print(f'Network template: {is_dorsattn.sum()} DorsAttn parcels (-1), '
      f'{is_default.sum()} Default parcels (+1), '
      f'{(~(is_dorsattn | is_default)).sum()} other-network parcels (0).')

print(f'\n{ref_subject} template component vs. DorsAttn(-1)/Default(+1) network template:')
for ref_comp in range(n_ref_components):
    comp_matches = ref_component_matches[ref_comp]
    matched_subjects = list(comp_matches.keys())

    group_weights = [all_surf_weights[ref_idx[ref_comp]]] + [
        np.sign(comp_matches[s][1]) * all_surf_weights[comp_matches[s][2]] for s in matched_subjects]
    median_weight = np.median(np.stack(group_weights, axis=0), axis=0)

    network_corr = np.corrcoef(network_template, median_weight)[0, 1]
    print(f'  #{ref_comp}: r = {network_corr:.3f}')

# %%
