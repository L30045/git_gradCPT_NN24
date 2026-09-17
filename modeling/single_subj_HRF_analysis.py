#%% SVD across parcels to look for cross-parcel structure in the EEG-related HRF.
# Load sub-723's Y_all, dm_all, and betas (continuous-EEG GLM, 3-stage), get each
# parcel's EEG delay-response HRF (betas_eeg, already expanded from bspline betas via
# dm_all's basis), stack into a parcel x delay matrix, and run SVD on it. The first
# left/right singular vector pair is the dominant cross-parcel HRF component; its
# variance explained is sigma_1^2 / sum(sigma^2).
import os
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

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

#%% load sub-723's Y_all, dm_all, and betas (continuous-EEG GLM, 3-stage)
subject = 'sub-723'
eeg_reg_type = 'cont_EEG_cz_3-stage'
is_hp_fNIRS = False
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
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
t_delay = np.arange(n_delay_taps) / fnirs_sfreq

hrf_parcel = betas_eeg.sel(chromo=select_chromo).values  # (parcel, delay)
parcel_names = betas_eeg.parcel.values

#%% SVD across parcels: hrf_parcel = U @ diag(S) @ Vt, rows=parcels, cols=delay
# U's columns are cross-parcel weighting patterns, Vt's rows are the shared HRF shapes
hrf_centered = hrf_parcel - hrf_parcel.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(hrf_centered, full_matrices=False)
var_explained = S**2 / np.sum(S**2)

#%% plot the first 15 shared HRF components, each labeled with its % variance explained
n_components = 15
fig, axs = plt.subplots(5, 3, figsize=(12, 14), sharex=True, sharey=True)
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
fig.supylabel('SVD component (a.u.)')
fig.suptitle(f'{subject}: first {n_components} cross-parcel EEG HRF components (n={len(parcel_names)} parcels)')
plt.tight_layout()
plt.show()

#%% same 15 components, but scaled by their singular value (S[i] * Vt[i]) so the
# y-scale reflects each mode's actual contribution to hrf_centered, not just its shape
fig2, axs2 = plt.subplots(5, 3, figsize=(12, 14), sharex=True, sharey=True)
axs2 = axs2.flatten()
for i in range(n_components):
    ax = axs2[i]
    ax.plot(t_delay, S[i] * Vt[i], color='r')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title(f'Component {i+1} ({var_explained[i]*100:.1f}%, S={S[i]:.2g})')
    ax.grid()
for ax in axs2[n_components:]:
    ax.set_visible(False)
fig2.supxlabel('Delay (s)')
fig2.supylabel('S x SVD component (a.u.)')
fig2.suptitle(f'{subject}: first {n_components} cross-parcel EEG HRF components, scaled by S (n={len(parcel_names)} parcels)')
plt.tight_layout()
plt.show()

#%% for each of the first 6 components, plot the shared HRF shape (left) next to its
# U-column weight (per-parcel loading onto that shape) rendered on the brain surface (right)
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

surf_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, 'SVD_components')
os.makedirs(surf_plot_dir, exist_ok=True)

n_surf_components = 6
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
    axs3[0].set_ylabel('SVD component (a.u.)')
    axs3[0].set_title(f'Component {i+1} HRF shape ({var_explained[i]*100:.1f}% variance explained)')
    axs3[0].grid()

    surf_img = plt.imread(surf_path + '.png')
    axs3[1].imshow(surf_img)
    axs3[1].axis('off')
    axs3[1].set_title(f'Component {i+1} U weight (per-parcel loading)')

    fig3.suptitle(f'{subject}: cross-parcel EEG HRF component {i+1}')
    plt.tight_layout()
    plt.show()

# %%
