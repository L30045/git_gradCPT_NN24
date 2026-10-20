#%% Network-based AR-iRRR and fixAR-IRLS on the mnt-correct, Cz-EEG-weighted HRF
# Same models and shared Y_all as run_model_gaussian-kernel_mnt-correct_network-based.py
# (parcel_Y_all_truncated_to_trials_<Hp|noHp>: per-run highpass, trial window from the first event onset
# to the last onset + 15 s, drift/GSR OLS-regressed out, runs concatenated as gradcpt1, 2, 3), but the
# design matrix is the delayed continuous Cz EEG of run_model_cont_EEG_city_mnt.py instead of the
# Gaussian kernels: the Cz EEG is zeroed outside the mnt-correct trials (onset -> onset + duration), its
# delays [-t_pre, t_post) are projected onto a cubic B-spline basis (at the EEG rate), downsampled to the
# fNIRS rate, and sampled at the Y_all samples. Every fNIRS sample is modeled by the within-trial EEG from
# t_post seconds before to t_pre seconds after it, so the betas are an event-locked HRF over
# [-t_pre, t_post) weighted by the within-trial EEG (M / V).
# The regressors are built on each full EEG run (so trials near the window edges keep their whole delay
# history); the EEG is zero outside the trials, so the zero padding beyond the recording is exact as long
# as the trials are inside it.
# The parcels are split by functional network (parcel name before the first '_') and each network is fit
# separately:
#   - one spectral-whitening (AR) kernel per network, estimated from the OLS residuals of that network's
#     parcels and applied to that network's Y and to X
#   - AR-iRRR: one iRRR model per network; irrr_lam1 is the smallest penalty whose coefficient matrix
#     has rank target_rank
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
from statsmodels.gam.smooth_basis import BSplines
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from cedalion.models.glm.design_matrix import DesignMatrix
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% select model type
t_pre = 2 # EEG lead (sec): each fNIRS sample also sees the EEG up to t_pre seconds after it
t_post = 18 # EEG lag (sec): each fNIRS sample sees the EEG up to t_post seconds before it
len_delay = t_pre + t_post # length of the delay window [-t_pre, t_post) (sec)
reg_type = f'cont_EEG_cz_mnt-correct_pre{t_pre}post{t_post}_3-stage'
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3'] # order of the runs in Y_all
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save betas, stats and dm_all
is_hp_fNIRS = True # If True, use the Y_all highpassed by 0.02 Hz
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
max_extend_decades = 6 # decades below cand_irrr_lam1[0] to scan if no candidate reaches target_rank
# iRRR ADMM stopping tolerance: with Tol = 0.01 the solver stops after ~10 iterations, far from the
# solution, and the rank of C is not monotone in lam1; 1e-8 is within ~0.5% of the 1e-10 solution
irrr_params = {'varyrho': True, 'Tol': 1e-8, 'Niter': 20000, 'fig': False}
# continuous-EEG design: Cz EEG of the mnt-correct trials, delays projected onto cubic B-splines
# (same basis as run_model_cont_EEG_city_mnt.py)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis
hrf_dt = 0.01 # delay resolution (sec) of the saved HRFs; basis_da keeps the full EEG-rate basis
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf')

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
irrr_model_name = f'{reg_type}_AR-iRRR-network{get_scale_tag(is_scale_Y)}_{hp_flag}'
irls_model_name = f'{reg_type}_fixAR-IRLS-network{get_scale_tag(is_scale_Y)}_{hp_flag}'

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

def get_mnt_correct_eeg_regressors(subject, run_key, EEG, bspline_basis, run_samples, dt):
    """Delayed Cz-EEG B-spline regressors (time x bspline) of the mnt-correct trials of one run, sampled
    at run_samples (fNIRS time = sample index * dt, starting at 0 like the fNIRS run).
    The Cz EEG is zeroed outside the mnt-correct trials (onsets / durations in the run's nirs events.tsv,
    fNIRS time) and the EEG <-> fNIRS clock offset comes from the first onset of the EEG and nirs
    events.tsv (nirs_time = eeg_time + t_offset), as in run_model_cont_EEG_city_mnt.py."""
    run_tag = f"run-{run_key[-1]:0>2}"
    nirs_ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                          f"{subject}_task-gradCPT_{run_tag}_events.tsv"), sep='\t')
    eeg_ev_df = pd.read_csv(os.path.join(project_path, 'derivatives', 'eeg', subject, 'preprocessed_EEG_and_events',
                                         f"{subject}_task-gradCPT_{run_tag}_events.tsv"), sep='\t')
    t_offset = nirs_ev_df['onset'].values[0] - eeg_ev_df['onset'].values[0]
    trials_df = nirs_ev_df[(nirs_ev_df['trial_type'] == 'mnt') & (nirs_ev_df['response_code'] == 0)]

    # trial mask over the EEG samples, in fNIRS time
    sample_nirs_t = EEG.times + t_offset
    eeg_mask = np.zeros(len(sample_nirs_t), dtype=bool)
    for onset, duration in zip(trials_df['onset'].values, trials_df['duration'].values):
        eeg_mask |= (sample_nirs_t >= onset) & (sample_nirs_t < onset + duration)
    n_outside = int(np.sum((trials_df['onset'].values < sample_nirs_t[0]) |
                           (trials_df['onset'].values + trials_df['duration'].values > sample_nirs_t[-1])))

    # the masked EEG is zero outside the trials, so the zero padding beyond the recording is exact; the
    # regressors are lowpassed at the fNIRS Nyquist at the EEG rate and read at the Y_all sample times
    X_run = model.sample_cont_EEG_bspline_regressor(EEG.get_data(picks='cz').flatten() * eeg_mask, sample_nirs_t,
                                                    bspline_basis, t_pre, run_samples * dt, 0.5 / dt)
    return X_run, len(trials_df), n_outside

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

    #%% load the shared Y_all
    Y_all_load_path = get_shared_Y_all_path(data_save_path, subject, hp_flag)
    with gzip.open(Y_all_load_path, 'rb') as f:
        Y_all = pickle.load(f)
    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    samples = Y_da.samples.values  # per-run fNIRS sample index; resets mark run starts in Y_all
    run_bounds = np.where(np.diff(samples) <= 0)[0] + 1
    seg_bounds = np.concatenate([[0], run_bounds, [len(samples)]])
    if len(seg_bounds) - 1 != len(run_keys):
        print(f"{subject}: Y_all has {len(seg_bounds) - 1} runs, expected {len(run_keys)}, skipping.")
        continue
    dt = np.median(np.diff(Y_da.time.values))

    # Y_all segments that hold the same fNIRS data (a run matched twice when Y_all was built) would
    # pair one fNIRS run with two runs' events. Drift/GSR were fit per (slightly different) trial window,
    # so duplicated segments are nearly, not exactly, equal.
    seg_slices = [slice(seg_bounds[k], seg_bounds[k + 1]) for k in range(len(run_keys))]
    is_dup = False
    for a in range(len(run_keys)):
        for b in range(a + 1, len(run_keys)):
            _, ia, ib = np.intersect1d(samples[seg_slices[a]], samples[seg_slices[b]], return_indices=True)
            if len(ia) > 100 and np.corrcoef(Y_da.values[seg_slices[a]][ia].ravel(),
                                             Y_da.values[seg_slices[b]][ib].ravel())[0, 1] > 0.99:
                print(f"{subject}: Y_all runs {run_keys[a]} and {run_keys[b]} hold the same fNIRS data, skipping.")
                is_dup = True
    if is_dup:
        continue

    #%% get continuous EEG (Cz)
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    if any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]] for run_key in run_keys):
        print(f"{subject}: Cz was removed in at least one run, skipping subject.")
        continue

    #%% B-spline basis over the delay axis (EEG rate); delay index k is delay k / eeg_sfreq - t_pre
    eeg_sfreq = single_subj_EEG_dict[run_keys[0]].info['sfreq']
    n_delay = np.round(len_delay * eeg_sfreq).astype(int)
    delay_names = [f"delay{d_i}" for d_i in range(n_delay)]
    delay_times = np.arange(n_delay) / eeg_sfreq - t_pre  # sec, -t_pre .. t_post
    bspline_basis = BSplines(np.arange(n_delay), df=[n_bspline_basis], degree=[bspline_degree],
                             include_intercept=True).basis  # (n_delay, n_bspline_basis)
    basis_da = xr.DataArray(
        bspline_basis,
        dims=("regressor", "component"),
        coords={"regressor": delay_names,
                "delay_time": ("regressor", delay_times),
                "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
    )
    # basis subsampled to hrf_dt for the saved HRFs (HRF time x component)
    hrf_step = int(round(hrf_dt * eeg_sfreq))
    hrf_basis = basis_da.isel(regressor=slice(None, None, hrf_step)).swap_dims(regressor='delay_time') \
                        .rename(delay_time='time').drop_vars('regressor').transpose('time', 'component')
    hrf_t = hrf_basis.time.values

    #%% design matrix: mnt-correct Cz-EEG B-spline regressors on each run's Y_all samples
    regressors = [f'mnt_correct_bspline{i}' for i in range(n_bspline_basis)]
    X_runs, n_trials = [], {}
    for run_key, seg in zip(run_keys, seg_slices):
        X_run, n_trials[run_key], n_outside = get_mnt_correct_eeg_regressors(
            subject, run_key, single_subj_EEG_dict[run_key], bspline_basis, samples[seg], dt)
        X_runs.append(X_run)
        print(f"{run_key}: {seg.stop - seg.start} samples, {n_trials[run_key]} mnt-correct trials"
              + (f" (WARNING: {n_outside} not fully inside the EEG recording)" if n_outside else ''))
    X_all = xr.DataArray(np.vstack(X_runs)[:, :, None], dims=('time', 'regressor', 'chromo'),
                         coords={'time': Y_da.time.values, 'regressor': regressors, 'chromo': [select_chromo]})
    dm_all = DesignMatrix(common=X_all)
    X_da = X_all.sel(chromo=select_chromo).transpose('time', 'regressor')
    X_ols = np.column_stack([np.ones(len(X_da.time)), X_da.values])

    # parcel name before the first '_' is the network it belongs to
    parcels = Y_da.parcel.values
    network_labels = np.array([p.split('_')[0] for p in parcels])
    networks = list(dict.fromkeys(network_labels))  # unique, in parcel order
    print(f"{subject}: {len(networks)} networks, {len(parcels)} parcels, {len(run_keys)} runs")

    irrr_subj_plot_dir = os.path.join(plot_dir, subject, irrr_model_name)
    os.makedirs(irrr_subj_plot_dir, exist_ok=True)

    C_irrr = np.full((len(parcels), len(regressors)), np.nan)
    C_irls = np.full((len(parcels), len(regressors)), np.nan)
    cov_irls = np.full((len(parcels), len(regressors), len(regressors)), np.nan)
    irrr_stats = {'irrr_lam1': {}, 'irrr_weight': {}, 'Y_scale': {}, 'target_rank': {},
                  'scan_lam1': {}, 'scan_rank': {},
                  'intercept': {}, 'rank': {}, 'singular_values': {},
                  'whiten_kernel_fft': {}, 'acf_kernel': {}}
    irls_stats = {'Y_scale': {}, 'whiten_kernel_fft': {}, 'acf_kernel': {}, 'f_test_full_noEEG': {}}
    f_restriction = np.eye(len(regressors))  # test if the mnt-correct EEG regressors explain any variance

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
            C, mu, _, _, _ = irrr_normal(Y_s, [X_white], lam1, {**irrr_params, 'weight': irrr_weight})
            return {'lam1': lam1, 'C': C, 'mu': np.ravel(mu), 'rank': matrix_rank(C)}

        # scan from the largest lam1 down; the rank grows as lam1 shrinks, so stop once it passes
        # net_rank (the small-lam1 fits are the slowest to converge)
        irrr_scan = []
        for lam1 in tqdm(cand_irrr_lam1[::-1], desc=f'{net} iRRR lam1 scan'):
            irrr_scan.append(fit_irrr(lam1))
            if irrr_scan[-1]['rank'] > net_rank:
                break
        # small networks may stay below net_rank even at the smallest candidate; extend the scan
        # downward a decade at a time until the rank reaches net_rank
        n_extend = 0
        while max(f['rank'] for f in irrr_scan) < net_rank:
            if n_extend == max_extend_decades:
                raise RuntimeError(f"{subject} [{net}]: rank < {net_rank} down to irrr_lam1 = "
                                   f"{min(f['lam1'] for f in irrr_scan):.4g}")
            irrr_scan.append(fit_irrr(min(f['lam1'] for f in irrr_scan) / 10))
            n_extend += 1
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
        cov_irls[net_idx] = np.stack([res.cov_params().values for res in rlm_results]) * Y_scale ** 2
        # statistics are in units of Y / Y_scale
        for p, res in zip(parcels[net_idx], rlm_results):
            irls_stats['f_test_full_noEEG'][p] = res.f_test(f_restriction)
        for key, val in [('Y_scale', Y_scale), ('whiten_kernel_fft', noise_model.W_fft),
                         ('acf_kernel', noise_model.acf_kernel)]:
            irls_stats[key][net] = val

    #%% assemble betas across networks, expand to HRFs and save
    common_stats = {'networks': networks, 'network_labels': network_labels, 'run_bounds': run_bounds,
                    'run_keys': run_keys, 'n_trials': n_trials, 't_pre': t_pre, 't_post': t_post,
                    'Y_all_path': Y_all_load_path, 'dm_all_path': get_dm_all_path(irls_prefix)}
    irrr_stats.update(common_stats, cand_irrr_lam1=cand_irrr_lam1)
    irls_stats.update(common_stats)

    hrf_coords = {'parcel': parcels, 'chromo': [select_chromo], 'time': hrf_t}
    for C_all, prefix, stats_dict in [(C_irrr, irrr_prefix, irrr_stats), (C_irls, irls_prefix, irls_stats)]:
        betas_all = xr.DataArray(
            C_all[:, None, :],
            dims=('parcel', 'chromo', 'regressor'),
            coords={'parcel': parcels, 'chromo': [select_chromo], 'regressor': regressors},
        ).assign_coords(network=('parcel', network_labels))
        # HRF (parcel x chromo x delay, at hrf_dt) = B-spline basis @ B-spline betas, in M / V
        betas_hrf = xr.DataArray((C_all @ hrf_basis.values.T)[:, None, :], dims=('parcel', 'chromo', 'time'),
                                 coords=hrf_coords).assign_coords(network=('parcel', network_labels))

        # Y_all is not re-saved; stats_dict points to the loaded copy
        if is_save:
            betas_dict = {'betas': betas_all, 'betas_hrf': betas_hrf, 'basis_da': basis_da,
                          't_pre': t_pre, 't_post': t_post}
            if prefix == irrr_prefix:
                betas_dict['irrr_lam1'] = irrr_stats['irrr_lam1']  # per-network rank-target_rank lam1
            else:
                # HRF standard error sqrt(diag(B cov B')) from the IRLS covariance of the B-spline betas
                se = np.sqrt(np.einsum('tk,pkl,tl->pt', hrf_basis.values, cov_irls, hrf_basis.values,
                                       optimize=True))
                betas_dict['betas_hrf_se'] = xr.DataArray(se[:, None, :], dims=('parcel', 'chromo', 'time'),
                                                          coords=hrf_coords)
                betas_dict['cov_betas'] = cov_irls
            betas_save_path = get_betas_path(prefix)
            stats_save_path = get_stats_path(prefix)
            os.makedirs(os.path.dirname(betas_save_path), exist_ok=True)
            with open(betas_save_path, 'wb') as f:
                pickle.dump(betas_dict, f)
            os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
            with open(stats_save_path, 'wb') as f:
                pickle.dump(stats_dict, f)

    # the design matrix is shared by both models; saved once next to the fixAR-IRLS outputs
    if is_save:
        dm_all_save_path = get_dm_all_path(irls_prefix)
        os.makedirs(os.path.dirname(dm_all_save_path), exist_ok=True)
        with gzip.open(dm_all_save_path, 'wb') as f:
            pickle.dump(dm_all, f)
