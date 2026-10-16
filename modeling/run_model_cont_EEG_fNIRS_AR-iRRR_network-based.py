#%% Network-based AR-iRRR and fixAR-IRLS
# Same as run_model_cont_EEG_fNIRS_AR-iRRR.py / run_model_cont_EEG_fNIRS_fixAR-IRLS.py, except that the
# parcels are split by functional network (parcel name before the first '_', e.g. 'DefaultA') and
# each network is fit separately:
#   - one spectral-whitening (AR) kernel per network, estimated from the OLS residuals of that network's
#     parcels and applied to that network's Y and to X
#   - AR-iRRR: one iRRR model per network; irrr_lam1 is the smallest penalty whose coefficient matrix
#     has rank target_rank (same selection as cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py)
#   - fixAR-IRLS: IRLS per parcel on the network-whitened Y and X, for comparison
# Both models share the per-network whitened Y and Y_scale.
#%% load library
import numpy as np
import pickle
import gzip
import glob
import sys
import os
import re
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% select model type
eeg_reg_type = 'cont_EEG_cz_3-stage'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save betas and stats
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
is_plot = False # If True, show visualization plots
select_chromo = 'HbO'
is_scale_Y = True # If True, scale whitened Y (per network) to unit std before the fits (betas are scaled back)
max_jobs = -1 # parallel jobs across parcels (fixAR-IRLS)
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685) # same robust norm as model.my_ar_irls_GLM
target_rank = 2 # iRRR rank used to pick irrr_lam1 (capped at the network's parcel count)
# candidate iRRR nuclear-norm penalties (scale-free only when is_scale_Y; otherwise in units of Y)
if is_scale_Y:
    cand_irrr_lam1 = 10 ** np.arange(-3, 2.01, 0.125)
else:
    cand_irrr_lam1 = 10 ** np.arange(-11, -8, 0.1)
n_bisect = 20 # bisection steps on log10(lam1) if no candidate gives exactly target_rank
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf')
# AR-IRLS run (run_model_cont_EEG_fNIRS.py) whose dm_all / B-spline basis are reused here;
# Y_all is the shared parcel_Y_all_truncated_to_trials file, so only the fit differs
ar_irls_reg_type = 'cont_EEG_cz_3-stage_bspline-test'

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
irrr_model_name = f'{eeg_reg_type}_AR-iRRR-network{get_scale_tag(is_scale_Y)}_{hp_flag}'
irls_model_name = f'{eeg_reg_type}_fixAR-IRLS-network{get_scale_tag(is_scale_Y)}_{hp_flag}'

#%% find subjects with a shared Y_all (only saved for subjects with fNIRS and enough EEG epochs)
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')
subj_id_array = sorted(
    int(re.search(r'sub-(\d+)', f).group(1))
    for f in glob.glob(get_shared_Y_all_path(os.path.join(_eeg_deriv, 'sub-*'), 'sub-*', hp_flag))
)
subj_id_array = [x for x in subj_id_array if f'sub-{x}' not in excluded_subj]

def get_irrr_weight(X, n_parcel):
    # theoretical iRRR weight: max(svd(X_c)) * (sqrt(q) + sqrt(rank(X_c))) / n, X_c column-centered
    X_c = X - X.mean(0, keepdims=True)
    return [np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / X.shape[0]]

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    irrr_prefix = os.path.join(data_save_path, f'{subject}_{irrr_model_name}')
    irls_prefix = os.path.join(data_save_path, f'{subject}_{irls_model_name}')
    if not is_overwrite and all(os.path.exists(get_betas_path(p)) for p in [irrr_prefix, irls_prefix]):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% load Y_all, dm_all and the B-spline basis saved by the AR-IRLS run
    ar_irls_prefix = os.path.join(data_save_path, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    Y_all_load_path = get_shared_Y_all_path(data_save_path, subject, hp_flag)
    dm_all_load_path = get_dm_all_path(ar_irls_prefix)
    ar_irls_betas_path = get_betas_path(ar_irls_prefix)
    if not all(os.path.exists(f) for f in [Y_all_load_path, dm_all_load_path, ar_irls_betas_path]):
        print(f"{subject}: Y_all or AR-IRLS dm_all/betas not found ({ar_irls_prefix}_*), skipping.")
        continue

    with gzip.open(Y_all_load_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(dm_all_load_path, 'rb') as f:
        dm_all = pickle.load(f)
    with open(ar_irls_betas_path, 'rb') as f:
        ar_irls_betas_dict = pickle.load(f)
    basis_da = ar_irls_betas_dict['basis_da']
    n_regressor = len(basis_da.regressor)

    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    X_ols = np.column_stack([np.ones(len(X_da.time)), X_da.values])
    samples = Y_da.samples.values  # per-run sample index; resets mark run starts in Y_all
    run_bounds = np.where(np.diff(samples) <= 0)[0] + 1
    regressors = X_da.regressor.values
    eeg_reg = [p for p in regressors if 'bspline' in p]

    # parcel name before the first '_' is the network it belongs to
    parcels = Y_da.parcel.values
    network_labels = np.array([p.split('_')[0] for p in parcels])
    networks = list(dict.fromkeys(network_labels))  # unique, in parcel order
    print(f"{subject}: {len(networks)} networks, {len(parcels)} parcels, {len(run_bounds) + 1} runs")

    irrr_subj_plot_dir = os.path.join(plot_dir, subject, irrr_model_name)
    os.makedirs(irrr_subj_plot_dir, exist_ok=True)

    C_irrr = np.full((len(parcels), len(regressors)), np.nan)
    C_irls = np.full((len(parcels), len(regressors)), np.nan)
    irrr_stats = {'irrr_lam1': {}, 'irrr_weight': {}, 'Y_scale': {}, 'target_rank': {},
                  'scan_lam1': {}, 'scan_rank': {},
                  'intercept': {}, 'rank': {}, 'singular_values': {},
                  'whiten_kernel_fft': {}, 'acf_kernel': {}}
    irls_stats = {'Y_scale': {}, 'whiten_kernel_fft': {}, 'acf_kernel': {},
                  'f_test_full_noEEG': {}, 't_test_0_eeg': {}}
    f_hypotheses = [f'{name} = 0' for name in eeg_reg]  # test if EEG can explain more variance
    t_hypotheses = '+'.join(eeg_reg) + ' = 0'  # test if EEG betas sums to 0

    for net in networks:
        net_idx = np.where(network_labels == net)[0]
        Y_net = Y_da.values[:, net_idx]
        n_parcel = len(net_idx)
        print(f"{subject} [{net}]: {n_parcel} parcels")

        #%% spectral whitening of Y and X with this network's kernel
        # the residual spectrum is estimated from the OLS residuals of Y_net ~ 1 + X (Y_all is already
        # drift- and GSR-residualized), averaged across the network's parcels, and its 1/sqrt(power)
        # filter is applied to both Y_net and X. Runs are padded and filtered separately.
        B_ols = np.linalg.lstsq(X_ols, Y_net, rcond=None)[0]
        noise_model = model.NoiseModel(method='spectral')
        noise_model.fit(model.split_runs(Y_net - X_ols @ B_ols, samples))
        Y_white = np.vstack(noise_model.whiten(model.split_runs(Y_net, samples)))
        X_white = np.vstack(noise_model.whiten(model.split_runs(X_da.values, samples)))
        Y_scale = np.nanstd(Y_white) if is_scale_Y else 1.0
        Y_s = Y_white / Y_scale  # shared by AR-iRRR and fixAR-IRLS

        #%% AR-iRRR: irrr_lam1 is the smallest lam1 (least shrinkage) whose fit has rank net_rank
        net_rank = min(target_rank, n_parcel)
        irrr_weight = get_irrr_weight(X_white, n_parcel)

        def fit_irrr(lam1):
            C, mu, _, _, _ = irrr_normal(Y_s, [X_white], lam1,
                                         {'varyrho': True, 'Tol': 0.01, 'fig': False, 'weight': irrr_weight})
            return {'lam1': lam1, 'C': C, 'mu': np.ravel(mu), 'rank': matrix_rank(C)}

        irrr_scan = [fit_irrr(lam1) for lam1 in tqdm(cand_irrr_lam1, desc=f'{net} iRRR lam1 scan')]
        rank_hits = [f for f in irrr_scan if f['rank'] == net_rank]
        if not rank_hits:
            # bisect on log10(lam1) between the largest lam1 above net_rank and the next lam1 below it
            lo = max(f['lam1'] for f in irrr_scan if f['rank'] > net_rank)
            hi = min(f['lam1'] for f in irrr_scan if f['rank'] < net_rank and f['lam1'] > lo)
            for _ in range(n_bisect):
                fit = fit_irrr(np.sqrt(lo * hi))
                irrr_scan.append(fit)
                if fit['rank'] == net_rank:
                    rank_hits = [fit]
                    break
                lo, hi = (fit['lam1'], hi) if fit['rank'] > net_rank else (lo, fit['lam1'])
            else:
                raise RuntimeError(f"{subject} [{net}]: no irrr_lam1 in [{lo:.4g}, {hi:.4g}] gives rank {net_rank}")
        irrr_scan = sorted(irrr_scan, key=lambda f: f['lam1'])
        irrr_fit = min(rank_hits, key=lambda f: f['lam1'])
        irrr_lam1 = irrr_fit['lam1']
        print(f"{subject} [{net}]: irrr_lam1 = {irrr_lam1:.4g} (rank {irrr_fit['rank']})")

        scan_lam1 = np.array([f['lam1'] for f in irrr_scan])
        scan_rank = np.array([f['rank'] for f in irrr_scan])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(scan_lam1, scan_rank, marker='o', ms=4)
        ax.axvline(irrr_lam1, color='r', ls='--', label=f'lam1 = {irrr_lam1:.3g} (rank {net_rank})')
        ax.set_xscale('log')
        ax.set_xlabel('irrr_lam1')
        ax.set_ylabel('rank(B)')
        ax.set_title(f'{subject} {net} ({n_parcel} parcels) iRRR lam1 selection')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(irrr_subj_plot_dir, f'{subject}_{net}_iRRR_lam1_vs_rank.png'), dpi=150)
        if is_plot:
            plt.show()
        plt.close(fig)

        C = irrr_fit['C'] * Y_scale  # back to original Y units
        C_irrr[net_idx] = C.T
        for key, val in [('irrr_lam1', irrr_lam1), ('irrr_weight', irrr_weight), ('Y_scale', Y_scale),
                         ('target_rank', net_rank), ('scan_lam1', scan_lam1), ('scan_rank', scan_rank),
                         ('intercept', irrr_fit['mu'] * Y_scale), ('rank', irrr_fit['rank']),
                         ('singular_values', svdvals(C)), ('whiten_kernel_fft', noise_model.W_fft),
                         ('acf_kernel', noise_model.acf_kernel)]:
            irrr_stats[key][net] = val

        #%% fixAR-IRLS: IRLS per parcel on the same whitened, scaled data (no further AR whitening)
        # the whitened series are zero-mean (DC bin of the kernel is 0), so no intercept is added
        x_df = pd.DataFrame(X_white, columns=regressors)
        rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
            delayed(model.irls_fit)(pd.Series(Y_s[:, p_i]), x_df, M=irls_norm)
            for p_i in range(n_parcel)
        )
        C_irls[net_idx] = np.stack([res.params.values for res in rlm_results]) * Y_scale
        # statistics are in units of Y / Y_scale
        for p, res in zip(parcels[net_idx], rlm_results):
            irls_stats['f_test_full_noEEG'][p] = res.f_test(f_hypotheses)
            irls_stats['t_test_0_eeg'][p] = res.t_test(t_hypotheses)
        for key, val in [('Y_scale', Y_scale), ('whiten_kernel_fft', noise_model.W_fft),
                         ('acf_kernel', noise_model.acf_kernel)]:
            irls_stats[key][net] = val

    #%% assemble betas across networks and save
    common_stats = {'networks': networks, 'network_labels': network_labels, 'run_bounds': run_bounds,
                    'Y_all_path': Y_all_load_path, 'dm_all_path': dm_all_load_path}
    irrr_stats.update(common_stats, cand_irrr_lam1=cand_irrr_lam1)
    irls_stats.update(common_stats)

    for C_all, prefix, stats_dict in [(C_irrr, irrr_prefix, irrr_stats), (C_irls, irls_prefix, irls_stats)]:
        betas_all = xr.DataArray(
            C_all[:, None, :],
            dims=('parcel', 'chromo', 'regressor'),
            coords={'parcel': parcels, 'chromo': [select_chromo], 'regressor': regressors},
        ).assign_coords(network=('parcel', network_labels))
        # extract HRF (delay-regressor betas) per parcel, then expand the low-rank
        # bspline coefficients back to full per-delay resolution via the same basis
        betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
        betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
        betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])

        # Y_all / dm_all are not re-saved; stats_dict points to the loaded copies
        if is_save:
            betas_dict = {'betas': betas_all, 'betas_eeg': betas_eeg,
                          'betas_bspline': betas_bspline, 'basis_da': basis_da}
            if prefix == irrr_prefix:
                betas_dict['irrr_lam1'] = irrr_stats['irrr_lam1']  # per-network rank-target_rank lam1
            betas_save_path = get_betas_path(prefix)
            stats_save_path = get_stats_path(prefix)
            os.makedirs(os.path.dirname(betas_save_path), exist_ok=True)
            with open(betas_save_path, 'wb') as f:
                pickle.dump(betas_dict, f)
            os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
            with open(stats_save_path, 'wb') as f:
                pickle.dump(stats_dict, f)
