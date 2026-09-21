#%% SVD/ICA across parcels to look for cross-parcel structure in the EEG-related HRF.
# For every subject with saved continuous-EEG GLM outputs (Y_all, dm_all, betas;
# 3-stage), get each parcel's EEG delay-response HRF (betas_eeg, already expanded from
# bspline betas via dm_all's basis), stack into a parcel x delay matrix, and decompose
# it (SVD or ICA, set by decomp_method below) into a shared HRF shape per component
# (Vt's rows) and a per-parcel weight onto each shape (U's columns). Each subject is
# decomposed independently (component sign/order is not aligned across subjects).
import os
import gzip
import glob
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.decomposition import FastICA
from scipy.spatial.distance import pdist, squareform

import cedalion.io
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view
from params_setting import *

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
# (matches the mask applied in run_model_cont_EEG_fNIRS.py / other vis_hrf_*.py scripts)
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = cedalion.io.load_Adot(Adot_path + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
mask_medial = Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH',
    'Background+FreeSurfer_Defined_Medial_Wall_RH'
])
Adot_brain = Adot_brain.sel(vertex=~mask_medial)

intensity = np.log10(Adot_brain[:, :, 1].sum('channel'))
sensitivity_mask = (intensity > -2).drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values)
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex')
sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (601 -> 417)

#%% Parcel distance matrix
coords = head.brain.vertices
dist_mat = squareform(pdist(coords))  # NxN  - all pairwise distances
# Adj = (dist_mat <= radius) & (dist_mat > 0)  # Neightbors within the set radius

#%% key: which decomposition method to use for the plots below ('svd' or 'ica')
decomp_method = 'svd'
assert decomp_method in ('svd', 'ica')

#%% subject list: every subject with saved continuous-EEG GLM outputs (Y_all, dm_all,
# betas) for eeg_reg_type, excluding subjects already flagged for low fNIRS quality
eeg_reg_type = 'cont_EEG_cz_3-stage_bspline-test'
is_hp_fNIRS = True
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'
n_components = 15
n_surf_components = 6

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(
    eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')))

subjects = []
for f in betas_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if subject in excluded_subj:
        continue
    subjects.append(subject)
print(f'Found {len(subjects)} subjects with continuous-EEG GLM outputs: {subjects}')

#%% brain surface setup (shared across subjects)
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

#%% run the SVD/ICA cross-parcel HRF analysis for each subject in turn
for subject in subjects:
    #%% load this subject's Y_all, dm_all, and betas (continuous-EEG GLM, 3-stage)
    base = os.path.join(eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')

    with gzip.open(base + '_Y_all.pkl.gz', 'rb') as f:
        Y_all = pickle.load(f)  # dims: chromo, parcel, time

    with gzip.open(base + '_dm_all.pkl.gz', 'rb') as f:
        dm_all = pickle.load(f)  # .common dims: time, chromo, regressor

    with open(base + '_betas.pkl', 'rb') as f:
        betas_dict = pickle.load(f)
    betas_eeg = betas_dict['betas_eeg']  # dims: parcel, chromo, regressor (delay tap); regressor = bspline basis @ betas_bspline

    #%% calculate EEG-related HRF for each parcel: delay-tap time axis + parcel x delay matrix
    # betas_eeg's parcels are already sensitivity-masked (saved that way by
    # run_model_cont_EEG_fNIRS.py); re-select against sensitive_parcels to make the mask
    # explicit here too, in case betas_eeg ever carries unmasked parcels
    betas_eeg = betas_eeg.sel(parcel=betas_eeg.parcel.isin(sensitive_parcels))

    fnirs_sfreq = 1 / np.diff(Y_all.time.values).mean()
    n_delay_taps = len(betas_eeg.regressor)
    # hard coded visualization time scale. The frequency should be EEG sampling/resampling frequency instead of fNIRS sampling frequency.
    if n_delay_taps>900:
        t_delay = np.arange(n_delay_taps) / 500
    else:
        t_delay = np.arange(n_delay_taps) / fnirs_sfreq

    hrf_parcel = betas_eeg.sel(chromo=select_chromo).values  # (parcel, delay)
    parcel_names = betas_eeg.parcel.values

    #%% decompose hrf_centered (parcel x delay) = U @ Vt, rows=parcels, cols=delay, via
    # either SVD or ICA (set by decomp_method above)
    # - SVD: U's columns are cross-parcel weighting patterns, Vt's rows (orthogonal) are
    #   the shared HRF shapes; components are ordered by singular value, and variance
    #   explained is sigma_i^2 / sum(sigma^2), exact by construction.
    # - ICA: treat each parcel as one ICA sample (row) and delay as the feature axis, so
    #   ICA finds independent sources across the delay axis (Vt's rows, not necessarily
    #   orthogonal) and a per-parcel mixing weight for each source (U's columns). U
    #   (ica.fit_transform's output) lives in unit-variance whitened source space, so Vt
    #   must be ica.mixing_.T (the actual mixing matrix, in original data units) rather
    #   than ica.components_ (the unmixing matrix in whitened space) for U @ Vt to
    #   reconstruct hrf_centered. Components are ordered by their actual reconstruction
    #   variance (var of U[:,i] outer Vt[i]), largest first, since ICA components have no
    #   intrinsic order like SVD's singular values.
    hrf_centered = hrf_parcel - hrf_parcel.mean(axis=0, keepdims=True)

    if decomp_method == 'svd':
        U_full, S_full, Vt_full = np.linalg.svd(hrf_centered, full_matrices=False)
        U = U_full[:, :n_components]
        S = S_full[:n_components]
        Vt = Vt_full[:n_components]
        var_explained = (S_full**2 / np.sum(S_full**2))[:n_components]
    elif decomp_method == 'ica':
        ica = FastICA(n_components=n_components, whiten='unit-variance', random_state=0, max_iter=1000)
        U = ica.fit_transform(hrf_centered)  # (parcel, n_components): per-parcel mixing weights
        Vt = ica.mixing_.T  # (n_components, delay): independent HRF shapes, in original data units

        component_recon_var = np.array([
            (np.outer(U[:, i], Vt[i])).var() for i in range(n_components)
        ])
        order = np.argsort(component_recon_var)[::-1]
        U = U[:, order]
        Vt = Vt[order]
        component_recon_var = component_recon_var[order]
        var_explained = component_recon_var / component_recon_var.sum()
        S = None  # ICA has no singular-value equivalent

    #%% save the per-parcel EEG HRFs and the decomposition results for later analysis
    decomp_out = {
        'decomp_method': decomp_method,
        'parcel_names': parcel_names,      # (parcel,)
        't_delay': t_delay,                # (delay,)
        'hrf_parcel': hrf_parcel,          # (parcel, delay): per-parcel EEG HRF, pre-centering
        'hrf_centered': hrf_centered,      # (parcel, delay): parcel-mean-subtracted, what was decomposed
        'U': U,                            # (parcel, n_components): per-parcel component weights
        'S': S,                            # (n_components,) singular values, SVD only (None for ICA)
        'Vt': Vt,                          # (n_components, delay): shared HRF shapes
        'var_explained': var_explained,    # (n_components,)
    }
    decomp_out_path = os.path.join(
        eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_{decomp_method}_HRF_decomp.pkl')
    with open(decomp_out_path, 'wb') as f:
        pickle.dump(decomp_out, f)
    print(f'Saved {decomp_method.upper()} decomposition to {decomp_out_path}')

    #%% plot the first 15 shared HRF components, each labeled with its % variance explained
    fig, axs = plt.subplots(5, 3, figsize=(12, 14), sharex=True)
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
    fig.supylabel(f'{decomp_method.upper()} component (a.u.)')
    fig.suptitle(f'{subject}: first {n_components} cross-parcel EEG HRF {decomp_method.upper()} components (n={len(parcel_names)} parcels)')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    #%% for each of the first 6 components, plot the shared HRF shape (left) next to its
    # U-column weight (per-parcel loading onto that shape) rendered on the brain surface (right)
    surf_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, f'{decomp_method.upper()}_components')
    os.makedirs(surf_plot_dir, exist_ok=True)

    for i in range(n_surf_components):
        # broadcast this component's per-parcel U weight onto the brain surface vertices
        weight_by_parcel = dict(zip(parcel_names, U[:, i]))
        vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
        clim_max = np.nanmax(np.abs(vertex_vals))

        X_surf = xr.DataArray(
            np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )
        surf_path = os.path.join(surf_plot_dir, f'component{i+1}_U_weight')
        image_recon_multi_view(
            X_ts=X_surf, head=head, cmap='seismic', clim=(-clim_max, clim_max),
            view_type='hbo_brain',
            title_str=f'Component {i+1} U weight',
            SAVE=True, filename=surf_path,
            wdw_size=(1600, 800),
        )

        fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5))
        axs3[0].plot(t_delay, Vt[i], color='r')
        axs3[0].axhline(0, color='gray', linewidth=0.8)
        axs3[0].set_xlabel('Delay (s)')
        axs3[0].set_ylabel(f'{decomp_method.upper()} component (a.u.)')
        axs3[0].set_title(f'Component {i+1} HRF shape ({var_explained[i]*100:.1f}% variance explained)')
        axs3[0].grid()

        surf_img = plt.imread(surf_path + '.png')
        axs3[1].imshow(surf_img)
        axs3[1].axis('off')
        axs3[1].set_title(f'Component {i+1} U weight (per-parcel loading)')

        fig3.suptitle(f'{subject}: cross-parcel EEG HRF component {i+1}')
        plt.tight_layout()
        plt.show()
        plt.close(fig3)

# %%
