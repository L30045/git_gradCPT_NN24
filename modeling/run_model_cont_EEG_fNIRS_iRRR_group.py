#%% load library
import numpy as np
import pickle
import copy
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.io
import cedalion.models.glm as glm
from cedalion.sigproc import frequency
from statsmodels.gam.smooth_basis import BSplines
from scipy.signal import butter, sosfiltfilt, filtfilt, windows
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% select model type
eeg_reg_type = 'cont_EEG_cz_3-stage'
is_save = True # If True, save group betas and stats
is_plot = True # If True, generate visualization plots
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
select_chromo='HbO'
select_parcel='DefaultA_PFCd_1_LH'
len_delay = 15 # Delay time in HRF (sec); must match run_model_cont_EEG_fNIRS.py
n_comp = 4 # max number of shared HRF components to plot
irrr_lam1 = 1.0 # iRRR nuclear-norm penalty (Y is scaled to unit std before fitting, so this is scale-free)
# AR-IRLS run (run_model_cont_EEG_fNIRS.py) whose Y_all / dm_all / B-spline basis are reused here;
# iRRR shares the same preprocessed Y and design matrix, so only the fit differs
ar_irls_reg_type = 'cont_EEG_cz_3-stage_bspline-test'
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'

group_save_path = os.path.join(project_path, 'derivatives', 'eeg', 'group')
os.makedirs(group_save_path, exist_ok=True)
betas_save_path = os.path.join(group_save_path, f'group_{eeg_reg_type}_iRRR_{hp_flag}_betas.pkl')
stats_save_path = os.path.join(group_save_path, f'group_{eeg_reg_type}_iRRR_{hp_flag}_stats.pkl')

#%% find subjects with AR-IRLS Y_all / dm_all available
_Y_all_files = sorted(glob.glob(os.path.join(
    project_path, 'derivatives', 'eeg', 'sub-*',
    f'sub-*_parcel_Y_all_truncated_to_trials_{hp_flag}.pkl.gz')))
subj_list = []
for _f in _Y_all_files:
    subject = re.search(r'(sub-\d+)', os.path.basename(_f)).group(1)
    if subject in excluded_subj:
        continue
    _prefix = os.path.join(os.path.dirname(_f), f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    if all(os.path.exists(f'{_prefix}_{s}') for s in ['dm_all.pkl.gz', 'betas.pkl']):
        subj_list.append((subject, _prefix, _f))
print(f"Found {len(subj_list)} subjects: {[s for s, _, _ in subj_list]}")

#%% load and concatenate Y_all and dm_all across subjects (runs are already concatenated within subject)
Y_list, X_list, subj_labels, t_list = [], [], [], []
parcels, regressors, basis_da = None, None, None
for subject, ar_irls_prefix, Y_all_path in subj_list:
    with gzip.open(Y_all_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(f'{ar_irls_prefix}_dm_all.pkl.gz', 'rb') as f:
        dm_all = pickle.load(f)
    if basis_da is None:
        with open(f'{ar_irls_prefix}_betas.pkl', 'rb') as f:
            basis_da = pickle.load(f)['basis_da']
    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    if parcels is None:
        parcels, regressors = Y_da.parcel.values, X_da.regressor.values
    assert np.array_equal(Y_da.parcel.values, parcels), f"{subject}: parcel mismatch"
    assert np.array_equal(X_da.regressor.values, regressors), f"{subject}: regressor mismatch"
    Y_np, X_np = Y_da.values, X_da.values
    # demean per subject: equivalent to a subject-specific intercept in the per-subject fits
    Y_list.append(Y_np - np.nanmean(Y_np, 0, keepdims=True))
    X_list.append(X_np - X_np.mean(0, keepdims=True))
    subj_labels.append(np.full(len(Y_np), subject))
    t_list.append(Y_da.time.values)
    print(f"{subject}: {len(Y_np)} time points")
Y_np = np.concatenate(Y_list, axis=0)
X_np = np.concatenate(X_list, axis=0)
subj_labels = np.concatenate(subj_labels)
n_regressor = len(basis_da.regressor)

#%% fit one iRRR model on the concatenated data
# Y (time x parcel) ~ X (time x bspline), nuclear-norm penalty on the (bspline x parcel)
# coefficient matrix so the HRFs across parcels share a low-rank structure
print(f"Start group iRRR fitting: Y {Y_np.shape}, X {X_np.shape}")
Y_scale = np.nanstd(Y_np)
n_time, n_parcel = Y_np.shape
irrr_weight = [np.max(svdvals(X_np)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_np))) / n_time]
t0 = time.time()
C, mu, _, _, _, irrr_details = irrr_normal(Y_np / Y_scale, [X_np], irrr_lam1,
                                           {'varyrho': True, 'Tol': 0.01, 'fig': False,
                                            'weight': irrr_weight},
                                            
                                           return_details=True)
print(f"iRRR done in {time.time() - t0:.1f} s")
C = C * Y_scale  # back to original Y units
mu = mu * Y_scale
betas_all = xr.DataArray(
    C.T[:, None, :],
    dims=('parcel', 'chromo', 'regressor'),
    coords={'parcel': parcels, 'chromo': [select_chromo], 'regressor': regressors},
)
stats_dict = {'irrr_lam1': irrr_lam1, 'irrr_weight': irrr_weight, 'Y_scale': Y_scale,
              'intercept': mu, 'rank': matrix_rank(C), 'singular_values': svdvals(C),
              'details': irrr_details,
              'subjects': [s for s, _, _ in subj_list],
              'n_time_per_subj': {s: int((subj_labels == s).sum()) for s, _, _ in subj_list},
              'ar_irls_prefixes': [p for _, p, _ in subj_list]}
print(f"iRRR fit: rank(B) = {stats_dict['rank']} (of {min(C.shape)})")
# extract HRF (delay-regressor betas) per parcel, then expand the low-rank
# bspline coefficients back to full per-delay resolution via the same basis
eeg_reg = [p for p in betas_all.regressor.values if 'bspline' in p]
betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])

#%% save betas (Y_all / dm_all are not re-saved; stats_dict points to the AR-IRLS copies)
if is_save:
    betas_dict = dict()
    betas_dict['betas'] = betas_all
    betas_dict['betas_eeg'] = betas_eeg
    betas_dict['betas_bspline'] = betas_bspline
    betas_dict['basis_da'] = basis_da
    with open(betas_save_path, 'wb') as f:
        pickle.dump(betas_dict, f)

    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"Saved {betas_save_path}")

#%% visualization
if is_plot:
    import cedalion.dot
    from cedalion.vis.anatomy.image_recon import image_recon_multi_view

    plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', 'group', f'{eeg_reg_type}_iRRR')
    os.makedirs(plot_dir, exist_ok=True)
    subjects = stats_dict['subjects']
    delay_t = np.arange(n_regressor) * (len_delay / n_regressor)

    #%% Y_partial vs Y_hat_eeg per subject for one parcel, and R2 per subject
    # Y_np / X_np are per-subject demeaned, so Y_hat_eeg = X @ C + mu on the concatenated data
    Y_hat_eeg = X_np @ betas_all.sel(chromo=select_chromo, regressor=regressors).transpose('regressor', 'parcel').values + mu.T
    p_i = np.where(parcels == select_parcel)[0][0]
    r2_eeg = np.zeros((len(subjects), n_parcel))
    fig, axes = plt.subplots(len(subjects), 1, figsize=(14, 2.5 * len(subjects)), squeeze=False)
    for s_i, subject in enumerate(subjects):
        idx = subj_labels == subject
        Y_s, Y_hat_s = Y_np[idx], Y_hat_eeg[idx]
        ss_res = np.nansum((Y_s - Y_hat_s)**2, axis=0)
        ss_tot = np.nansum((Y_s - np.nanmean(Y_s, axis=0))**2, axis=0)
        r2_eeg[s_i] = 1 - ss_res / ss_tot
        ax = axes[s_i, 0]
        ax.plot(t_list[s_i], Y_s[:, p_i], 'k', lw=0.8, label='Y_partial')
        ax.plot(t_list[s_i], Y_hat_s[:, p_i], 'r', lw=0.8, label='Y_hat_eeg')
        ax.set_ylabel(f'{select_chromo} (M)')
        ax.set_title(f'{subject}: R2 = {r2_eeg[s_i, p_i]:.4f}')
        print(f"R2 (EEG) {subject}: {select_parcel} = {r2_eeg[s_i, p_i]:.4f}, "
              f"median over parcels = {np.nanmedian(r2_eeg[s_i]):.4f}")
    axes[0, 0].legend(loc='upper right')
    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(f'Group iRRR {select_parcel}')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'group_{select_parcel}_Y_partial_vs_Y_hat_eeg.png'))
    plt.close(fig)

    # distribution of per-parcel R2 for each subject
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.boxplot(r2_eeg.T, showfliers=False)
    ax.set_xticks(np.arange(1, len(subjects) + 1), subjects)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('R2 (EEG) across parcels')
    ax.set_title('Group iRRR fit per subject')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'group_R2_per_subject.png'))
    plt.show()
    plt.close(fig)

    #%% shared HRF components: SVD of the (delay x parcel) HRF matrix
    hrf_mat = betas_eeg.sel(chromo=select_chromo).transpose('regressor', 'parcel').values
    U, S, Vt = np.linalg.svd(hrf_mat, full_matrices=False)
    n_plot = max(1, min(n_comp, stats_dict['rank']))

    # render each component's per-parcel weight (row of Vt) on the brain surface
    head = cedalion.dot.get_standard_headmodel('icbm152')
    vertex_parcel = head.brain.vertices.parcel.values
    n_vertex = head.brain.nvertices
    surf_paths = []
    for c_i in range(n_plot):
        weight_by_parcel = dict(zip(betas_eeg.parcel.values, Vt[c_i]))
        vertex_vals = np.array([weight_by_parcel.get(p, np.nan) for p in vertex_parcel])
        clim_max = np.nanmax(np.abs(vertex_vals))
        X_surf = xr.DataArray(
            np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )
        surf_path = os.path.join(plot_dir, f'component{c_i+1}_weight')
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
    fig.suptitle(f'Group iRRR shared HRFs ({len(subjects)} subjects, rank = {stats_dict["rank"]})')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'group_iRRR_shared_HRF_components.png'))
    plt.show()
