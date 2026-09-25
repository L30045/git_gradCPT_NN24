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
import gzip
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

#%% use the joint Vt as a fixed template and check how much variance each component
# explains within each subject individually. Since Vt's rows are orthonormal (SVD),
# each subject's own hrf_centered (parcel, delay) can be projected onto each Vt row by
# a simple dot product along the delay axis to get that subject's per-parcel loading on
# the template component; the component's variance explained for that subject is the
# variance of its rank-1 reconstruction (outer product of the subject-specific loading
# and the template Vt row) divided by that subject's total hrf_centered variance.
var_explained_subj = np.zeros((len(subjects), n_components))  # (n_subj, n_components)

for si, subject in enumerate(subjects):
    subj_rows = [i for i, (s, _) in enumerate(row_labels) if s == subject]
    X = all_hrf_centered[subj_rows]  # (parcel, delay), this subject's own HRF data
    U_subj = X @ Vt.T  # (parcel, n_components): per-parcel loading onto each template component
    total_var = (X**2).sum()
    for comp in range(n_components):
        recon = np.outer(U_subj[:, comp], Vt[comp])
        var_explained_subj[si, comp] = (recon**2).sum() / total_var

print(f'\nVariance explained by each template component, within each subject:')
for si, subject in enumerate(subjects):
    per_comp = ', '.join(f'#{c}={var_explained_subj[si, c]*100:.1f}%' for c in range(n_components))
    print(f'  {subject}: {per_comp}')

#%% grouped bar chart: template components on x-axis, one bar per subject, showing that
# subject's variance explained by that component -- reveals which components generalize
# well across subjects (similarly tall bars) vs. are subject-specific (high variance
# across bars within a component)
fig7, ax7 = plt.subplots(figsize=(2.2 * n_components, 6))
bar_width = 0.8 / len(subjects)
x = np.arange(n_components)
for si, subject in enumerate(subjects):
    ax7.bar(x + si * bar_width, var_explained_subj[si] * 100, width=bar_width, label=subject)
ax7.set_xticks(x + bar_width * (len(subjects) - 1) / 2)
ax7.set_xticklabels([f'#{c}' for c in range(n_components)])
ax7.set_xlabel('Template component')
ax7.set_ylabel('Variance explained within subject (%)')
ax7.set_title(f'Per-subject variance explained by each joint {decomp_method.upper()} template component')
ax7.legend(fontsize=8, ncol=min(len(subjects), 6))
ax7.grid(axis='y')
plt.tight_layout()
plt.show()

#%% use the joint SVD template to reconstruct Y_all for one subject, and compare how
# much variance in Y_all is captured by the template-based reconstruction vs. that
# subject's own trained (real) betas.
#
# Per-parcel template HRF: U_subj @ Vt, where U_subj is that subject's own loading onto
# each template component (X @ Vt.T, X = subject's hrf_centered, valid since Vt's rows
# are orthonormal) -- this projects that subject's own real-unit HRF data onto the
# template subspace, giving a (parcel, delay) HRF already at the subject's own scale
# (see note below on why S must not be reapplied here).
#
# Y_all/dm_all/betas were fit in bspline-coefficient space (dm_all.common has 15
# bspline regressors, not the full 7500 delay taps -- the full-resolution delay design
# matrix was projected down via basis_da before being saved, and isn't itself saved).
# So the per-delay template HRF is projected back into that same bspline-coefficient
# space by least-squares against basis_da (delay x bspline), giving template betas
# directly comparable to the subject's real (trained) betas, both usable with the same
# saved dm_all.common to compute Y_hat.
target_subject = 'sub-723'
assert target_subject in subjects, f'{target_subject} not found among subjects with saved decompositions: {subjects}'

target_base = os.path.join(
    eeg_der_dir, target_subject, f'{target_subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')

with gzip.open(target_base + '_Y_all.pkl.gz', 'rb') as f:
    Y_all = pickle.load(f)  # dims: chromo, parcel, time

with gzip.open(target_base + '_dm_all.pkl.gz', 'rb') as f:
    dm_all = pickle.load(f)  # .common dims: time, regressor (bspline), chromo

with open(target_base + '_betas.pkl', 'rb') as f:
    betas_dict = pickle.load(f)
betas_subj = betas_dict['betas']    # (parcel, chromo, regressor=bspline): subject's real trained betas
basis_da = betas_dict['basis_da']   # (regressor=delay, component=bspline)

# this subject's per-parcel loading onto each template component (X_target @ Vt.T,
# valid since Vt's rows are orthonormal), and its template HRF reconstruction at that
# subject's own scale (U_target @ Vt). Note: U_target here is NOT a slice of the joint
# SVD's U_full -- it is a fresh projection of this subject's own (real-unit) HRF data
# onto the template shapes, so it already carries the correct magnitude. S must NOT be
# reapplied on top (S only rescales the joint SVD's own unit-norm U_full columns back
# to real units; multiplying U_target by S a second time double-scales it and shrinks
# the reconstruction by ~1/S, which is why an earlier version of this section produced
# a near-zero template Y_hat).
target_rows = [i for i, (s, _) in enumerate(row_labels) if s == target_subject]
X_target = all_hrf_centered[target_rows]  # (parcel, delay), target subject's own HRF data
target_parcel_names = [row_labels[i][1] for i in target_rows]
U_target = X_target @ Vt.T                # (parcel, n_components)
template_hrf_parcel = U_target @ Vt       # (parcel, delay)

# project the per-delay template HRF into bspline-coefficient space by least-squares:
# template_hrf_parcel[parcel, delay] ~= template_betas_bspline[parcel, bspline] @ basis_da[delay, bspline].T
template_betas_bspline, *_ = np.linalg.lstsq(basis_da.values, template_hrf_parcel.T, rcond=None)
template_betas_bspline = template_betas_bspline.T  # (parcel, n_bspline)

template_betas_da = xr.DataArray(
    template_betas_bspline[:, None, :],  # (parcel, chromo, bspline)
    dims=('parcel', 'chromo', 'regressor'),
    coords={'parcel': target_parcel_names, 'chromo': betas_subj.chromo.values,
            'regressor': basis_da.component.values},
)

#%% reconstruct Y_hat using the template betas and using the subject's real trained
# betas, both against the same (already-saved) bspline-space design matrix, then
# compare each reconstruction's variance to Y_all's total variance per parcel
Y_hat_template = xr.dot(dm_all.common, template_betas_da, dims='regressor')  # dims: time, chromo, parcel
Y_hat_subj = xr.dot(dm_all.common, betas_subj, dims='regressor')             # dims: time, chromo, parcel

target_chromo = Y_all.chromo.values[0]
y_true = Y_all.sel(chromo=target_chromo).transpose('time', 'parcel').values
y_hat_template = Y_hat_template.sel(chromo=target_chromo).transpose('time', 'parcel').sel(
    parcel=Y_all.parcel.values).values
y_hat_subj = Y_hat_subj.sel(chromo=target_chromo).transpose('time', 'parcel').sel(
    parcel=Y_all.parcel.values).values

var_y_all = y_true.var(axis=0)             # (parcel,)
var_y_hat_template = y_hat_template.var(axis=0)
var_y_hat_subj = y_hat_subj.var(axis=0)

frac_var_template = var_y_hat_template / var_y_all
frac_var_subj = var_y_hat_subj / var_y_all

# the per-parcel ratio is dominated by a handful of very-low-signal parcels (tiny
# var_y_all in the denominator blows up the ratio there), so summarize with the median
# (robust to those outliers) alongside the pooled ratio sum(Var(Y_hat))/sum(Var(Y_all))
# (equivalent to weighting each parcel by its own variance, so low-signal parcels barely
# move it) rather than a plain mean across parcels
print(f'\n{target_subject}: Var(Y_hat)/Var(Y_all), template vs. subject-specific betas '
      f'(over {len(Y_all.parcel.values)} parcels):')
print(f'  template : median={np.nanmedian(frac_var_template)*100:.1f}%, '
      f'pooled={var_y_hat_template.sum() / var_y_all.sum() * 100:.1f}%')
print(f'  subject  : median={np.nanmedian(frac_var_subj)*100:.1f}%, '
      f'pooled={var_y_hat_subj.sum() / var_y_all.sum() * 100:.1f}%')

#%% per-parcel comparison: scatter of Var(Y_hat_template)/Var(Y_all) vs.
# Var(Y_hat_subj)/Var(Y_all), one point per parcel, with the y=x line marked. Log-log
# scale, since a few very-low-signal parcels (tiny Var(Y_all)) blow up the template
# ratio while the subject-specific ratio correctly shrinks toward 0 there (the fitted
# betas were regressed against that parcel's own near-zero-signal data) -- on a linear
# scale these outliers would crush every other parcel toward the origin.
fig8, ax8 = plt.subplots(figsize=(6, 6))
valid = (frac_var_subj > 0) & (frac_var_template > 0) & np.isfinite(frac_var_subj) & np.isfinite(frac_var_template)
ax8.scatter(frac_var_subj[valid], frac_var_template[valid], s=10, alpha=0.5)
lims = [min(frac_var_subj[valid].min(), frac_var_template[valid].min()) * 0.5,
        max(frac_var_subj[valid].max(), frac_var_template[valid].max()) * 2]
ax8.plot(lims, lims, color='gray', linestyle='--', linewidth=1, label='y = x')
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_xlim(lims)
ax8.set_ylim(lims)
ax8.set_xlabel('Var(Y_hat_subj) / Var(Y_all)  (subject-specific betas)')
ax8.set_ylabel('Var(Y_hat_template) / Var(Y_all)  (joint SVD template betas)')
ax8.set_title(f'{target_subject}: per-parcel variance captured, template vs. subject-specific reconstruction')
ax8.legend()
ax8.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
