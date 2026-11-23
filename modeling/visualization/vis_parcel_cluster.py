#%% Visualize how brain parcels cluster together by spatial distance.
# Get each brain vertex's coordinates, average them within each parcel to get a
# per-parcel centroid, build the parcel x parcel distance matrix from those centroids,
# then hierarchically cluster the parcels and pick a distance threshold that balances
# the number of clusters against how tight (spatially compact) each cluster is.
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import pyvista as pv
import cedalion.dot
from cedalion.vis.anatomy import image_recon_multi_view

pv.set_jupyter_backend('static')

#%% brain surface setup: vertex coordinates (mm) and each vertex's parcel label
head = cedalion.dot.get_standard_headmodel('icbm152')
coords = head.brain.vertices.pint.dequantify().values  # (n_vertex, 3), mm
vertex_parcel = head.brain.vertices.parcel.values  # (n_vertex,)

# drop the non-brain medial-wall "parcels" -- they aren't real anatomical parcels
mask_medial = np.isin(vertex_parcel, ['Background+FreeSurfer_Defined_Medial_Wall_LH',
                                       'Background+FreeSurfer_Defined_Medial_Wall_RH'])
coords = coords[~mask_medial]
vertex_parcel = vertex_parcel[~mask_medial]

#%% categorize vertices into parcels and take the centroid of each parcel's vertex
# coordinates as that parcel's coordinate
parcel_names = np.unique(vertex_parcel)
parcel_coords = np.array([coords[vertex_parcel == p].mean(axis=0) for p in parcel_names])
n_parcels = len(parcel_names)
print(f'{n_parcels} parcels, centroid from {len(coords)} brain vertices.')

#%% parcel distance matrix: pairwise Euclidean distance between parcel centroids (mm)
dist_mat = squareform(pdist(parcel_coords))  # (n_parcels, n_parcels)

#%% hierarchical clustering (average-linkage) on the parcel distance matrix
condensed_dist = squareform(dist_mat, checks=False)
Z = linkage(condensed_dist, method='average')

#%% sweep the cut distance and, for each threshold, record the number of clusters and
# the mean within-cluster pairwise distance (averaged over clusters with >1 member --
# singleton clusters have no within-cluster distance to speak of). This is the
# number-of-clusters/compactness tradeoff: low threshold -> many, tight clusters;
# high threshold -> few, loose clusters.
def mean_within_cluster_dist(cluster_labels):
    within = []
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < 2:
            continue
        within.append(dist_mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)].mean())
    return np.mean(within) if within else 0.0

thresholds = np.linspace(Z[:, 2].min(), Z[:, 2].max(), 100)
n_clusters_arr = np.zeros(len(thresholds), dtype=int)
within_dist_arr = np.zeros(len(thresholds))
for i, t in enumerate(thresholds):
    cluster_labels = fcluster(Z, t=t, criterion='distance')
    n_clusters_arr[i] = len(np.unique(cluster_labels))
    within_dist_arr[i] = mean_within_cluster_dist(cluster_labels)

#%% pick the optimal threshold as the "elbow" of the tradeoff curve: normalize both the
# (decreasing) cluster count and the (increasing) within-cluster distance to [0, 1] and
# take the threshold that minimizes their sum -- i.e. as few clusters as possible while
# keeping clusters as tight as possible.
n_clusters_norm = (n_clusters_arr - n_clusters_arr.min()) / (n_clusters_arr.max() - n_clusters_arr.min())
within_dist_norm = (within_dist_arr - within_dist_arr.min()) / (within_dist_arr.max() - within_dist_arr.min())
elbow_idx = np.argmin(n_clusters_norm + within_dist_norm)
opt_threshold = thresholds[elbow_idx]

cluster_labels = fcluster(Z, t=opt_threshold, criterion='distance')
n_clusters = len(np.unique(cluster_labels))
print(f'Optimal cut distance = {opt_threshold:.1f} mm -> {n_clusters} clusters '
      f'(mean within-cluster distance = {within_dist_arr[elbow_idx]:.1f} mm).')

#%% visualize the number-of-clusters vs. within-cluster-distance tradeoff, with the
# chosen threshold marked
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(thresholds, n_clusters_arr, color='tab:blue', label='# clusters')
ax1.set_xlabel('Cut distance (mm)')
ax1.set_ylabel('# clusters', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(thresholds, within_dist_arr, color='tab:orange', label='mean within-cluster distance')
ax2.set_ylabel('Mean within-cluster distance (mm)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

ax1.axvline(opt_threshold, color='k', linestyle='--',
            label=f'chosen threshold = {opt_threshold:.1f} mm')
ax1.legend(loc='upper right')
ax1.set_title('Parcel clustering: # clusters vs. within-cluster distance tradeoff')
fig.tight_layout()
plt.show()

#%% visualize the dendrogram with the chosen cut threshold marked
fig2, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, color_threshold=opt_threshold, labels=parcel_names,
           leaf_rotation=90, leaf_font_size=4)
ax.axhline(opt_threshold, color='r', linestyle='--',
           label=f'cut at {opt_threshold:.1f} mm ({n_clusters} clusters)')
ax.set_xlabel('Parcel')
ax.set_ylabel('Distance (mm)')
ax.set_title('Hierarchical clustering of parcels by centroid distance')
ax.legend()
fig2.tight_layout()
plt.show()

#%% render a brain surface colored by parcel, for both the distance-based clusters and
# the canonical Schaefer 17-network assignment (parcel name's leading token, e.g.
# "VisCent_Striate_1_LH" -> "VisCent"), side by side for comparison, using
# image_recon_multi_view like the rest of the repo's brain-surface plots.
parcel_to_cluster = dict(zip(parcel_names, cluster_labels))
parcel_to_network = {p: p.split('_')[0] for p in parcel_names}
networks = sorted(set(parcel_to_network.values()))
network_to_id = {n: i for i, n in enumerate(networks)}

# image_recon_multi_view/image_recon expect a chromo dim (HbO/HbR) and an is_brain
# coord over vertex; encode the discrete cluster/network id as the (duplicated) scalar
# and use a ListedColormap so each id gets a distinct, evenly-spaced color.
full_vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

cluster_id_by_vertex = np.array([
    parcel_to_cluster.get(p, np.nan) for p in full_vertex_parcel
], dtype=float)
network_id_by_vertex = np.array([
    network_to_id.get(parcel_to_network.get(p), np.nan) for p in full_vertex_parcel
], dtype=float)


def make_vertex_img(id_by_vertex):
    return xr.DataArray(
        np.tile(id_by_vertex, (2, 1)),
        dims=['chromo', 'vertex'],
        coords={
            'chromo': ['HbO', 'HbR'],
            # head.brain.vertices are all brain vertices (no scalp mixed in), so mark
            # every one of them as brain -- image_recon needs this mask to select data.
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool)),
        },
    )


cluster_img = make_vertex_img(cluster_id_by_vertex)
network_img = make_vertex_img(network_id_by_vertex)

cluster_cmap = ListedColormap(plt.cm.get_cmap('tab20', n_clusters)(np.arange(n_clusters)))
network_cmap = ListedColormap(plt.cm.get_cmap('tab20', len(networks))(np.arange(len(networks))))

image_recon_multi_view(
    X_ts=cluster_img,
    head=head,
    cmap=cluster_cmap,
    clim=[0, n_clusters - 1],
    view_type='hbo_brain',
    title_str=f'Distance-based clusters (n={n_clusters})',
    filename=None,
    SAVE=False,
    wdw_size=(1300, 768),
)

image_recon_multi_view(
    X_ts=network_img,
    head=head,
    cmap=network_cmap,
    clim=[0, len(networks) - 1],
    view_type='hbo_brain',
    title_str=f'Canonical Schaefer networks (n={len(networks)})',
    filename=None,
    SAVE=False,
    wdw_size=(1300, 768),
)

# %%
