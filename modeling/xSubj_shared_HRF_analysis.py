#%% Cross-subject shared HRF shapes via one joint SVD across all subjects.
# Load every subject's saved per-parcel EEG HRF (hrf_centered, from
# single_subj_HRF_analysis.py), stack all subjects' parcel rows into one
# (sum_subj(n_parcel), delay) matrix, and decompose it with a single SVD. This gives
# one shared set of HRF shapes (Vt) directly comparable across subjects (no per-subject
# decomposition + cross-subject matching needed). Each shared component's per-parcel
# surface weight (U column) is then averaged across subjects (parcels are aligned/shared
# across subjects, so averaging is well-defined) and rendered on the brain surface,
# alongside a comparison against a DorsAttn-vs-Default network template.
import os
import glob
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

from params_setting import *

#%% key: which per-subject decomposition's saved hrf_centered to load (must match a
# decomp_out_path already saved by single_subj_HRF_analysis.py) and how many joint SVD
# components to keep
decomp_method = 'svd'
assert decomp_method == 'svd', 'joint decomposition below is SVD-specific'
n_components = 15

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

#%% for each subject, load its per-parcel HRF (hrf_centered: parcel x delay, already
# parcel-mean-subtracted by single_subj_HRF_analysis.py) and stack rows across subjects
# into one (n_subj*n_parcel, delay) matrix, tracking which subject/parcel each row is
# from. Requires every subject to share the same parcel set/order and delay axis
# (asserted below), since rows are later grouped back into per-parcel weights.
all_hrf_centered = []
row_labels = []  # (subject, parcel_name) per row of all_hrf_centered, same order
t_delay = None
parcel_names_ref = None

for subject in subjects:
    decomp_path = os.path.join(
        eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_{decomp_method}_HRF_decomp.pkl')
    with open(decomp_path, 'rb') as f:
        decomp_out = pickle.load(f)

    hrf_centered = decomp_out['hrf_centered']  # (parcel, delay)
    parcel_names = decomp_out['parcel_names']
    if t_delay is None:
        t_delay = decomp_out['t_delay']
    if parcel_names_ref is None:
        parcel_names_ref = parcel_names
    else:
        assert np.array_equal(parcel_names, parcel_names_ref), \
            f'{subject} has a different parcel set/order than the rest -- cannot stack rows directly'

    all_hrf_centered.append(hrf_centered)
    row_labels.extend([(subject, p) for p in parcel_names])

all_hrf_centered = np.concatenate(all_hrf_centered, axis=0)  # (n_subj*n_parcel, delay)
n_rows = all_hrf_centered.shape[0]
print(f'Stacked {n_rows} parcel-rows ({len(subjects)} subjects x {len(parcel_names_ref)} parcels) '
      f'into one matrix for joint SVD.')

#%% joint SVD across all subjects: all_hrf_centered (n_subj*n_parcel, delay) = U @ Vt.
# Vt's rows are the shared HRF shapes (one set for all subjects); U's columns are the
# per-(subject,parcel)-row loading onto each shape. Components ordered by singular
# value; variance explained is sigma_i^2 / sum(sigma^2), exact by construction.
U_full, S_full, Vt_full = np.linalg.svd(all_hrf_centered, full_matrices=False)
U = U_full[:, :n_components]        # (n_subj*n_parcel, n_components)
S = S_full[:n_components]
Vt = Vt_full[:n_components]         # (n_components, delay)
var_explained = (S_full**2 / np.sum(S_full**2))[:n_components]
print(f'Joint SVD: kept {n_components} components, '
      f'{var_explained.sum()*100:.1f}% of total variance explained.')

#%% plot the shared HRF shapes (Vt rows), each labeled with its % variance explained --
# same layout as single_subj_HRF_analysis.py's per-subject component plot
n_plot_rows = int(np.ceil(n_components / 3))
fig, axs = plt.subplots(n_plot_rows, 3, figsize=(12, 2.8 * n_plot_rows), sharex=True)
axs = axs.flatten()
for i in range(n_components):
    ax = axs[i]
    ax.plot(t_delay, Vt[i], color='r')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title(f'Component {i+1} ({var_explained[i]*100:.1f}%)')
    ax.grid()
for ax in axs[n_components:]:
    ax.set_visible(False)
fig.supxlabel('Delay (s)')
fig.supylabel(f'Joint {decomp_method.upper()} component (a.u.)')
fig.suptitle(f'Cross-subject joint {decomp_method.upper()}: first {n_components} shared HRF components '
             f'(n={len(subjects)} subjects, {n_rows} parcel-rows)')
plt.tight_layout()
plt.show()

#%% for each shared component, average its per-parcel U weight across subjects (parcels
# are aligned/shared across subjects, so grouping rows by parcel name and averaging over
# the subject axis is well-defined) to get one per-parcel weight map per component
parcel_row_idx = {p: [] for p in parcel_names_ref}
for i, (subject, parcel) in enumerate(row_labels):
    parcel_row_idx[parcel].append(i)

# avg_U_weight[comp] = {parcel_name: mean U weight across subjects}
avg_U_weight = []
for comp in range(n_components):
    weight_by_parcel = {p: U[idx, comp].mean() for p, idx in parcel_row_idx.items()}
    avg_U_weight.append(weight_by_parcel)

#%% render each component's subject-averaged per-parcel weight on the brain surface,
# composed next to the shared HRF shape for that component
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

surf_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'components_analysis')
os.makedirs(surf_plot_dir, exist_ok=True)

for comp in range(n_components):
    weight_by_parcel = avg_U_weight[comp]
    vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
    clim_max = np.nanmax(np.abs(vertex_vals))

    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    surf_path = os.path.join(surf_plot_dir, f'joint_{decomp_method}_component{comp}_avg_U_weight')
    image_recon_multi_view(
        X_ts=X_surf, head=head, cmap='seismic', clim=(-clim_max, clim_max),
        view_type='hbo_brain',
        title_str=f'Joint component {comp}: subject-averaged U weight (n={len(subjects)} subjects)',
        SAVE=True, filename=surf_path,
        wdw_size=(1600, 800),
    )

    # compose the saved surface image with this component's shared HRF shape into one
    # figure, since image_recon_multi_view saves its own plot rather than drawing into
    # a given axis
    fig6, (ax_surf, ax_hrf) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    surf_img = plt.imread(surf_path + '.png')
    ax_surf.imshow(surf_img)
    ax_surf.axis('off')

    ax_hrf.plot(t_delay, Vt[comp], color='r', linewidth=2)
    ax_hrf.axhline(0, color='gray', linewidth=0.8)
    ax_hrf.set_xlabel('Delay (s)')
    ax_hrf.set_ylabel(f'Joint {decomp_method.upper()} component (a.u.)')
    ax_hrf.set_title(f'Component {comp} shared HRF shape ({var_explained[comp]*100:.1f}%)')
    ax_hrf.grid()

    fig6.suptitle(f'Joint component {comp}: subject-averaged U weight (n={len(subjects)} subjects) + shared HRF shape')
    plt.tight_layout()
    plt.savefig(surf_path + '.png', dpi=150)
    plt.show()
    plt.close(fig6)
    print(f'Saved {surf_path}.png')

#%% for each shared component, compare its subject-averaged per-parcel weight against a
# DorsAttn-vs-Default network template over all parcels: template = -1 for DorsAttn
# parcels, +1 for Default parcels, 0 for every other-network parcel (no masking -- all
# parcels participate in the correlation).
is_dorsattn = np.array([p.startswith('DorsAttn') for p in parcel_names_ref])
is_default = np.array([p.startswith('Default') for p in parcel_names_ref])
network_template = np.where(is_dorsattn, -1, np.where(is_default, 1, 0))
print(f'Network template: {is_dorsattn.sum()} DorsAttn parcels (-1), '
      f'{is_default.sum()} Default parcels (+1), '
      f'{(~(is_dorsattn | is_default)).sum()} other-network parcels (0).')

print(f'\nJoint {decomp_method.upper()} components vs. DorsAttn(-1)/Default(+1) network template:')
for comp in range(n_components):
    weight_vec = np.array([avg_U_weight[comp][p] for p in parcel_names_ref])
    network_corr = np.corrcoef(network_template, weight_vec)[0, 1]
    print(f'  #{comp}: r = {network_corr:.3f}')

# %%
