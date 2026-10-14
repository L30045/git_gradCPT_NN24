#%% load library
# Compare fixAR-IRLS and AR-iRRR on one subject, trained on ALL runs (gradcpt1/2/3), so every metric
# is in-sample. Same as cmp_fixAR-IRLS_AR-iRRR_single-subj_single-run.py, except that the models are
# fit on all runs and every run is truncated to the trial window BEFORE it is spectrally whitened.
# Per run: highpass -> crop to the span with EEG -> OLS drift -> OLS GSR -> truncate -> whiten.
# One whitening kernel is estimated from the truncated residuals of all runs (no lag crosses a run
# boundary) and applied to each run separately; Y_scale is the std of all whitened runs.
# The comparison is run twice on the same Y: (1) delayed Cz EEG regressors (B-spline basis), and
# (2) event-based consecutive Gaussian kernels locked to the mnt-correct onsets.
import numpy as np
import pickle
import gzip
import glob
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import mne
import xarray as xr
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
import cedalion.io
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon
from cedalion.vis.blocks import plot_labeled_points
import cedalion.models.glm as glm
from statsmodels.gam.smooth_basis import BSplines
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% parameters
subj_id = 670
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3'] # all runs are used for training
select_chromo = 'HbO'
l_cutoff = 0.02 # highpass cutoff for fNIRS (Hz)
len_delay = 15 # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis
is_scale_Y = True # If True, scale whitened Y by its std over all runs (all metrics are in these units)
target_rank = 2 # iRRR rank used to pick irrr_lam1
# candidate iRRR nuclear-norm penalties (scale-free since Y is scaled); also used for the rank curves
cand_irrr_lam1 = 10 ** np.arange(-3, 2.01, 0.125)
n_bisect = 20 # bisection steps on log10(lam1) if no candidate gives exactly target_rank
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685) # same robust norm as model.my_ar_irls_GLM
max_jobs = -1 # parallel jobs across parcels
is_save = True
is_plot = False # If True, show plots (they are always saved)
plot_parcels = ['DefaultA_PFCd_1_LH'] # parcels whose whitened Y and fits are plotted (one figure each);
                                      # their HRFs are plotted with the n_cz_parcels parcels nearest to Cz
n_cz_parcels = 3
plot_t_win = None # (start, stop) in s from the start of each truncated run to zoom in; None = whole run
cfg_GLM['do_GSR'] = True
# event-based design: consecutive Gaussian kernels for mnt-correct trials (same as run_model_EEG_inform_parcel_based.py)
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 10*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}

subject = f"sub-{subj_id}"
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)
plot_root = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject)
os.makedirs(plot_root, exist_ok=True)

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
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
sensitive_parcels = np.unique(Adot_brain_sens.parcel.values)

#%% load fNIRS parcel time series
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    all_stims = pickle.load(f)['stims']
with open(os.path.join(der_dir, subject, f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'), 'rb') as f:
    all_runs = pickle.load(f)['parcel_ts']

all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels) & (run.parcel != 'scalp'),
                    chromo=[select_chromo]) for run in all_runs]
for run in all_runs:
    run.time.attrs['units'] = units.s
fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

#%% load continuous EEG
single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
if any('cz' in [ch.lower() for ch in single_subj_rm_ch_dict[run_key]] for run_key in run_keys):
    raise ValueError(f"{subject}: Cz was removed in at least one run.")
eeg_sfreq = single_subj_EEG_dict[run_keys[0]].info['sfreq']

#%% match each EEG run (gradcpt1/2/3) to its fNIRS run via the first event onset
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
run_info = dict() # run_key -> (run_idx, t_offset, event onsets in fNIRS time); nirs_time = eeg_time + t_offset
for run_key in run_keys:
    run_num = f"{run_key[-1]:0>2}"
    eeg_ev_df = pd.read_csv(os.path.join(eeg_der_dir, subject, 'preprocessed_EEG_and_events',
                                         f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
    nirs_ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                          f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
    nirs_onsets = nirs_ev_df['onset'].values
    run_idx = next(r_i for r_i, stim in enumerate(all_stims)
                   if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onsets[0], atol=0.01))
    run_info[run_key] = (run_idx, nirs_onsets[0] - eeg_ev_df['onset'].values[0], nirs_onsets)

# keep parcels that are finite in every run used here
parcels = all_runs[0].parcel.values
for run_key in run_keys:
    run = all_runs[run_info[run_key][0]].sel(parcel=parcels).pint.dequantify()
    parcels = parcels[np.isfinite(run.values).all(axis=tuple(i for i, d in enumerate(run.dims) if d != 'parcel'))]
print(f"{subject}: {len(parcels)} parcels")

#%% B-spline basis on the EEG-rate delay taps (same as run_model_cont_EEG_fNIRS.py)
n_delay = np.round(len_delay * eeg_sfreq).astype(int)
bspline_basis = BSplines(np.arange(n_delay), df=[n_bspline_basis], degree=[bspline_degree],
                         include_intercept=True).basis # (n_delay, n_bspline_basis)
basis_da = xr.DataArray(
    bspline_basis,
    dims=("regressor", "component"),
    coords={"regressor": [f"delay{d_i}" for d_i in range(n_delay)],
            "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
)

#%% consecutive Gaussian kernels for the event-based design (time from onset x component, at the fNIRS rate)
gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])
gauss_basis = gauss_kernels(all_runs[0])

#%% per-run preprocessing up to (not including) whitening
def preprocess_run(run_key):
    """Highpass the full fNIRS run, crop it to the span covered by EEG (with len_delay of EEG
    lead-in for the delay taps), regress out drift and GSR, and build the EEG B-spline regressors
    on the same span, plus the event-based Gaussian regressors for mnt-correct trials.
    Returns Y (time x parcel), X (time x bspline), X_stim (time x gaussian) and the trial-window mask."""
    run_idx, t_offset, nirs_onsets = run_info[run_key]
    fnirs_run = all_runs[run_idx].sel(parcel=parcels)
    EEG = single_subj_EEG_dict[run_key]

    # highpass the full run
    fnirs_units = fnirs_run.pint.units
    sos = butter(4, l_cutoff, btype='highpass', fs=fnirs_sfreq, output='sos')
    fnirs_run = xr.apply_ufunc(
        sosfiltfilt, sos, fnirs_run.pint.dequantify(),
        input_core_dims=[[], ['time']],
        output_core_dims=[['time']],
        exclude_dims={'time'},
    ).transpose(*fnirs_run.dims).pint.quantify(fnirs_units)
    fnirs_run = fnirs_run.assign_coords({'time': all_runs[run_idx].time})

    # crop to the span where EEG exists, including len_delay of EEG before the first fNIRS sample
    t_f = fnirs_run.time.values
    span_start = max(t_f[0], t_offset + len_delay)
    span_stop = min(t_f[-1], EEG.times[-1] + t_offset)
    fnirs_run = fnirs_run.sel(time=slice(span_start, span_stop))
    fnirs_run.time.attrs['units'] = units.s
    t_span = fnirs_run.time.values

    # event-based regressors: consecutive Gaussian kernels at the mnt-correct onsets (fNIRS time)
    stim = all_stims[run_idx]
    stim = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
    stim['trial_type'] = 'mnt-correct'
    stim_dm = glm.design_matrix.hrf_regressors(fnirs_run, stim, gauss_kernels)
    X_stim = stim_dm.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    stim_names = X_stim.regressor.values
    X_stim = X_stim.values

    fnirs_run = fnirs_run.assign_coords(time=t_span - t_span[0])
    fnirs_run.time.attrs['units'] = units.s

    # OLS-regress out the Legendre drift, then GSR (computed from the drift-residualized signal)
    drift_dm = model.get_drift_legendre_regressors([fnirs_run], cfg_GLM)[0]
    drift_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, drift_dm, noise_model='ols').sm.params, drift_dm)
    fnirs_run = (fnirs_run - drift_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)
    gsr_dm = model.get_global_mean_regressor([fnirs_run])[0]
    gsr_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, gsr_dm, noise_model='ols').sm.params, gsr_dm)
    fnirs_run = (fnirs_run - gsr_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)
    Y = fnirs_run.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel').values

    # EEG delay regressors projected on the B-spline basis; the leading n_delay samples (the
    # lead-in) are dropped, so the first row lines up with the first fNIRS sample of the span
    eeg = EEG.copy().crop(tmin=t_span[0] - t_offset - len_delay, tmax=t_span[-1] - t_offset)
    X = model.get_cont_EEG_bspline_regressor(eeg.get_data(picks='cz').flatten(), bspline_basis)
    X = mne.filter.resample(X.T, up=fnirs_sfreq, down=eeg_sfreq, npad='auto', axis=-1).T
    n = min(len(X), len(Y))
    Y, X, X_stim, t_span = Y[:n], X[:n], X_stim[:n], t_span[:n]

    # trial window in fNIRS time: first event onset to last event onset + len_delay
    keep = (t_span >= nirs_onsets[0]) & (t_span <= nirs_onsets[-1] + len_delay)
    print(f"{run_key}: span {n} samples ({n / fnirs_sfreq:.0f} s), trial window {keep.sum()} samples, "
          f"{len(stim)} mnt-correct trials")
    return dict(Y=Y, X=X, X_stim=X_stim, stim_names=stim_names, keep=keep, time=t_span)

run_data = {run_key: preprocess_run(run_key) for run_key in run_keys}


#%% parcels for the HRF plots: plot_parcels and the n_cz_parcels parcels nearest to Cz
# parcel distance to Cz = distance from the Cz landmark of the standard head to the parcel's nearest
# brain vertex (same measure as cmp_model_hrf_r2.py), among the parcels fitted here
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
cz_pos = head.landmarks.sel(label='Cz').pint.dequantify().values
vertex_pos = head.brain.vertices.pint.dequantify().values
is_fitted = np.isin(vertex_parcel, parcels)
cz_dist = pd.Series(np.linalg.norm(vertex_pos[is_fitted] - cz_pos, axis=1)).groupby(vertex_parcel[is_fitted]).min()
cz_parcel_dist = cz_dist.drop(plot_parcels, errors='ignore').nsmallest(n_cz_parcels)
print(f"{n_cz_parcels} parcels nearest to Cz:\n{cz_parcel_dist.round(1).to_string()}")
hrf_parcels = [(p, 'plot parcel') for p in plot_parcels] + [(p, f'{d:.1f} mm from Cz') for p, d in cz_parcel_dist.items()]
hrf_parcels = [(p, desc) for p, desc in hrf_parcels if p in parcels]

# superior view of each parcel (red) on the brain, with the Cz landmark marked
# (same rendering as cmp_model_hrf_r2.py)
n_vertex = head.brain.nvertices
parcel_pngs = dict()
for parcel, _ in hrf_parcels:
    vertex_vals = np.where(vertex_parcel == parcel, 1.0, np.nan) # NaN -> gray
    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    p0, _, _ = image_recon(X_surf, head, cmap='Reds', clim=(0, 1), view_type='hbo_brain',
                           view_position='superior', off_screen=True, wdw_size=(800, 800))
    plot_labeled_points(p0, head.landmarks.sel(label=['Cz']))
    parcel_pngs[parcel] = os.path.join(plot_root, f'surf_parcel_{parcel}.png')
    p0.screenshot(parcel_pngs[parcel])
    p0.close()

#%% comparison of fixAR-IRLS and AR-iRRR for one design matrix
def run_comparison(design):
    """Truncate, whiten (kernel from all runs) and scale Y and design['X_key'], fit fixAR-IRLS and
    AR-iRRR on all runs concatenated, pick the rank-target_rank irrr_lam1, score every run and all
    runs together (in-sample), plot and save. design keys: name, model_name, X_key, reg_names,
    hrf_basis (hrf time x component), hrf_t, hrf_xlabel, beta_unit. Returns a dict with the data,
    fits and metrics."""
    model_name = design['model_name']
    plot_dir = os.path.join(plot_root, model_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"===== {subject}: {design['name']} =====")

    # truncate every run to its trial window first
    Y_trunc = [run_data[run_key]['Y'][run_data[run_key]['keep']] for run_key in run_keys]
    X_trunc = [run_data[run_key][design['X_key']][run_data[run_key]['keep']] for run_key in run_keys]
    run_len = [len(y) for y in Y_trunc]

    # spectral whitening, kernel estimated on all truncated runs: the residual spectrum comes from
    # the OLS residuals of Y ~ 1 + X (one intercept per run, shared slopes), averaged across parcels,
    # as in run_model_cont_EEG_fNIRS_iRRR_AR.py. NoiseModel whitens each run separately.
    run_icpt = np.repeat(np.eye(len(run_keys)), run_len, axis=0)
    X_ols = np.column_stack([run_icpt, np.concatenate(X_trunc)])
    B_ols = np.linalg.lstsq(X_ols, np.concatenate(Y_trunc), rcond=None)[0]
    resid = np.split(np.concatenate(Y_trunc) - X_ols @ B_ols, np.cumsum(run_len)[:-1])
    noise_model = model.NoiseModel(method='spectral').fit(resid)
    tukey_m = int(np.round(np.sqrt(sum(run_len))))
    acf_tukey = noise_model.acf_kernel[:tukey_m]

    white_data = dict()
    for run_key, y_white, x_white in zip(run_keys, noise_model.whiten(Y_trunc), noise_model.whiten(X_trunc)):
        t = run_data[run_key]['time'][run_data[run_key]['keep']]
        white_data[run_key] = {'Y_white': y_white, 'X_white': x_white, 'time_white': t - t[0]}
    Y_scale = np.nanstd(np.concatenate([d['Y_white'] for d in white_data.values()])) if is_scale_Y else 1.0
    for d in white_data.values():
        d['Y_white'] = d['Y_white'] / Y_scale
    print(f"Y_scale = {Y_scale:.4g}")

    # training data: all runs concatenated (each run was whitened on its own)
    Y_tr = np.concatenate([white_data[run_key]['Y_white'] for run_key in run_keys])
    X_tr = np.concatenate([white_data[run_key]['X_white'] for run_key in run_keys])
    n_time, n_parcel = Y_tr.shape
    score_data = {**white_data, 'all': {'Y_white': Y_tr, 'X_white': X_tr}} # scored per run and pooled

    def get_metrics(Y, Y_hat):
        """PMSE / RMSE over all time points and parcels; R2 pooled over parcels (each parcel centered
        on its own mean) and per parcel."""
        res = Y - Y_hat
        ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum(0)
        pmse = np.mean(res ** 2)
        return {'PMSE': pmse, 'RMSE': np.sqrt(pmse), 'R2': 1 - (res ** 2).sum() / ss_tot.sum(),
                'R2_parcel': 1 - (res ** 2).sum(0) / ss_tot}

    def evaluate(C, mu):
        return {run_key: get_metrics(d['Y_white'], d['X_white'] @ C + mu[None, :])
                for run_key, d in score_data.items()}

    # fixAR-IRLS: IRLS per parcel on the whitened data (no intercept, as in the fixAR-IRLS script)
    print(f"Start IRLS fitting ({subject})")
    x_df = pd.DataFrame(X_tr, columns=design['reg_names'])
    rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
        delayed(model.irls_fit)(pd.Series(Y_tr[:, p_i]), x_df, M=irls_norm)
        for p_i in tqdm(range(n_parcel))
    )
    C_irls = np.stack([res.params.values for res in rlm_results], axis=1) # (regressor x parcel)
    irls_metrics = evaluate(C_irls, np.zeros(n_parcel))
    irls_rank = matrix_rank(C_irls)

    # AR-iRRR: scan irrr_lam1 on all runs
    X_c = X_tr - X_tr.mean(0, keepdims=True)
    irrr_weight = [np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / n_time]

    def fit_irrr(lam1):
        C, mu, _, _, _ = irrr_normal(Y_tr, [X_tr], lam1,
                                     {'varyrho': True, 'Tol': 0.01, 'fig': False, 'weight': irrr_weight})
        mu = np.ravel(mu)
        return {'lam1': lam1, 'C': C, 'mu': mu, 'rank': matrix_rank(C), 'metrics': evaluate(C, mu)}

    irrr_scan = [fit_irrr(lam1) for lam1 in tqdm(cand_irrr_lam1, desc='iRRR lam1 scan')]

    # select irrr_lam1: the smallest lam1 (least shrinkage) whose fit has rank target_rank
    rank_hits = [f for f in irrr_scan if f['rank'] == target_rank]
    if not rank_hits:
        # bisect on log10(lam1) between the largest lam1 above target_rank and the next lam1 below it
        lo = max(f['lam1'] for f in irrr_scan if f['rank'] > target_rank)
        hi = min(f['lam1'] for f in irrr_scan if f['rank'] < target_rank and f['lam1'] > lo)
        for _ in range(n_bisect):
            fit = fit_irrr(np.sqrt(lo * hi))
            irrr_scan.append(fit)
            if fit['rank'] == target_rank:
                rank_hits = [fit]
                break
            lo, hi = (fit['lam1'], hi) if fit['rank'] > target_rank else (lo, fit['lam1'])
        else:
            raise RuntimeError(f"no irrr_lam1 in [{lo:.4g}, {hi:.4g}] gives rank {target_rank}")
    irrr_scan = sorted(irrr_scan, key=lambda f: f['lam1'])
    irrr_fit = min(rank_hits, key=lambda f: f['lam1'])
    irrr_lam1 = irrr_fit['lam1']
    print(f"irrr_lam1 = {irrr_lam1:.4g} (rank {irrr_fit['rank']})")

    # in-sample R2 and PMSE per run and over all runs
    summary = pd.DataFrame([
        {'design': design['name'], 'model': name, 'run': run_key,
         'rank': rank, 'PMSE': m['PMSE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
         'median R2 (parcel)': np.median(m['R2_parcel'])}
        for name, rank, metrics in [('fixAR-IRLS', irls_rank, irls_metrics),
                                    (f'AR-iRRR (lam1={irrr_lam1:.3g})', irrr_fit['rank'], irrr_fit['metrics'])]
        for run_key, m in metrics.items()
    ])
    print(f"{subject}: trained on all runs; in-sample metrics on whitened Y / Y_scale")
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    # iRRR: rank vs RMSE and rank vs R2 per run and over all runs (all in-sample)
    # each rank is represented by its smallest-lam1 fit (the same rule used to pick irrr_lam1);
    # fixAR-IRLS is marked at its own (full) rank for reference
    best_per_rank = {}
    for f in irrr_scan:
        if f['rank'] not in best_per_rank or f['lam1'] < best_per_rank[f['rank']]['lam1']:
            best_per_rank[f['rank']] = f
    ranks = sorted(best_per_rank)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(['RMSE', 'R2'], axs):
        for run_key, color in zip(run_keys + ['all'], ['C0', 'C1', 'C2', 'k']):
            label = 'all runs' if run_key == 'all' else run_key
            ax.plot(ranks, [best_per_rank[r]['metrics'][run_key][metric] for r in ranks],
                    marker='o', color=color, label=f'iRRR {label}')
            ax.plot(irls_rank, irls_metrics[run_key][metric], marker='x', ms=10, mew=2, ls='none',
                    color=color, label=f'IRLS {label}')
        ax.axvline(target_rank, color='r', ls='--', lw=1)
        ax.set_xlabel('iRRR rank')
        ax.set_ylabel(f'{metric} (whitened Y / Y_scale)' if metric == 'RMSE' else 'R$^2$ (pooled over parcels)')
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f'{subject}, {design["name"]}: trained on all runs (in-sample)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{subject}_iRRR_rank_vs_RMSE_R2.png'), dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)

    # whitened Y and the fits of fixAR-IRLS and iRRR (rank target_rank) per run, for selected parcels
    # top row: unscaled (whitened Y in M, i.e. multiplied back by Y_scale); bottom row: Y / Y_scale.
    # The two rows differ only by the factor Y_scale. R2 is per parcel and the same in both rows.
    for plot_parcel in plot_parcels:
        if plot_parcel not in parcels:
            print(f"{plot_parcel} is not among the fitted parcels, skipping its plot.")
            continue
        p_i = np.where(parcels == plot_parcel)[0][0]
        fig, axs = plt.subplots(2, len(run_keys), figsize=(6 * len(run_keys), 7), sharex='col', sharey='row')
        for c_i, run_key in enumerate(run_keys):
            d = white_data[run_key]
            t = d['time_white']
            win = np.ones(len(t), dtype=bool) if plot_t_win is None else (t >= plot_t_win[0]) & (t <= plot_t_win[1])
            y = d['Y_white'][:, p_i]
            y_irls = d['X_white'] @ C_irls[:, p_i]
            y_irrr = d['X_white'] @ irrr_fit['C'][:, p_i] + irrr_fit['mu'][p_i]
            r2_irls = irls_metrics[run_key]['R2_parcel'][p_i]
            r2_irrr = irrr_fit['metrics'][run_key]['R2_parcel'][p_i]
            for r_i, scale in enumerate([Y_scale, 1.0]):
                ax = axs[r_i, c_i]
                ax.plot(t[win], y[win] * scale, color='k', lw=0.8, alpha=0.6, label='whitened Y')
                ax.plot(t[win], y_irls[win] * scale, color='C0', lw=1, label=f'fixAR-IRLS (R$^2$={r2_irls:.3f})')
                ax.plot(t[win], y_irrr[win] * scale, color='C3', lw=1,
                        label=f'iRRR rank {irrr_fit["rank"]} (R$^2$={r2_irrr:.3f})')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, loc='upper right')
            axs[0, c_i].set_title(run_key)
            axs[1, c_i].set_xlabel('Time from first event (s)')
        axs[0, 0].set_ylabel(f'whitened {select_chromo} (M)')
        axs[1, 0].set_ylabel(f'whitened {select_chromo} / Y_scale')
        fig.suptitle(f'{subject}, {design["name"]}, {plot_parcel}: unscaled (top) and scaled by '
                     f'Y_scale = {Y_scale:.3g} M (bottom)')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{subject}_{plot_parcel}_Y_white_fit.png'), dpi=150)
        if is_plot:
            plt.show()
        plt.close(fig)

    # HRF: expand the basis betas to the HRF time course, in original units
    def to_hrf(C):
        # (component x parcel) -> HRF (parcel x hrf time)
        return xr.DataArray((C.T * Y_scale) @ design['hrf_basis'].T, dims=('parcel', 'time'),
                            coords={'parcel': parcels, 'time': design['hrf_t']})

    hrf_irls = to_hrf(C_irls)
    hrf_irrr = to_hrf(irrr_fit['C'])

    # HRF of both models at plot_parcels and the n_cz_parcels parcels nearest to Cz, with parcel locations
    fig, axs = plt.subplots(2, len(hrf_parcels), figsize=(4.5 * len(hrf_parcels), 8.4), squeeze=False,
                            gridspec_kw={'height_ratios': [1, 1.1]})
    for ax in axs[0, 1:]:
        ax.sharey(axs[0, 0])
    for ax, surf_ax, (parcel, desc) in zip(axs[0], axs[1], hrf_parcels):
        ax.plot(design['hrf_t'], hrf_irls.sel(parcel=parcel).values, color='C0', label='fixAR-IRLS')
        ax.plot(design['hrf_t'], hrf_irrr.sel(parcel=parcel).values, color='C3',
                label=f'iRRR rank {irrr_fit["rank"]}')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'{parcel}\n({desc})', fontsize='medium')
        ax.set_xlabel(design['hrf_xlabel'])
        ax.grid(alpha=0.3)
        surf_ax.imshow(plt.imread(parcel_pngs[parcel]))
        surf_ax.axis('off')
    axs[0, 0].set_ylabel(design['beta_unit'])
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].text(0, 0.5, 'Parcel location\n(superior view,\nred; Cz marked)', transform=axs[1, 0].transAxes,
                   ha='right', va='center', fontsize='small')
    fig.suptitle(f'{subject}, {design["name"]}: HRF trained on all runs')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{subject}_hrf_select_and_{n_cz_parcels}_nearest_Cz_parcels.png'), dpi=150)
    if is_plot:
        plt.show()
    plt.close(fig)

    # save
    results_dict = {
        'subject': subject, 'design': design['name'], 'train_runs': run_keys,
        'parcels': parcels, 'regressors': design['reg_names'], 'Y_scale': Y_scale,
        'hrf_basis': design['hrf_basis'], 'hrf_t': design['hrf_t'], 'whiten_kernel_acf': acf_tukey,
        'summary': summary,
        'irls': {'C': C_irls, 'rank': irls_rank, 'metrics': irls_metrics, 'betas_eeg': hrf_irls},
        'irrr': {'irrr_lam1': irrr_lam1, 'target_rank': target_rank, 'weight': irrr_weight,
                 'C': irrr_fit['C'], 'mu': irrr_fit['mu'], 'metrics': irrr_fit['metrics'],
                 'betas_eeg': hrf_irrr},
        'irrr_scan': [{k: f[k] for k in ['lam1', 'rank', 'metrics']} for f in irrr_scan],
    }
    if is_save:
        save_path = os.path.join(data_save_path, f'{subject}_{model_name}_cmp_fixAR-IRLS_AR-iRRR.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"saved {save_path}")
    results_dict['white_data'] = white_data
    return results_dict

#%% design 1: delayed Cz EEG regressors (B-spline basis over the delay axis)
res_eeg = run_comparison({
    'name': 'Cz EEG delay regressors', 'model_name': 'cont_EEG_cz_postWhiten_train-allRuns',
    'X_key': 'X', 'reg_names': basis_da.component.values,
    'hrf_basis': bspline_basis, 'hrf_t': np.arange(n_delay) / eeg_sfreq,
    'hrf_xlabel': 'Delay (s)', 'beta_unit': f'β ({select_chromo} M per V of Cz)',
})

#%% design 2: event-based consecutive Gaussian kernels for mnt-correct trials
stim_names = run_data[run_keys[0]]['stim_names']
if not all(np.array_equal(d['stim_names'], stim_names) for d in run_data.values()):
    raise ValueError("mnt-correct regressor names differ across runs.")
if gauss_basis.sizes['component'] != len(stim_names):
    raise ValueError(f"cfg_HRF_basis gives {gauss_basis.sizes['component']} kernels but the DM has {len(stim_names)}.")
res_stim = run_comparison({
    'name': 'mnt-correct Gaussian kernels', 'model_name': 'mnt-correct_gauss_postWhiten_train-allRuns',
    'X_key': 'X_stim', 'reg_names': stim_names,
    'hrf_basis': gauss_basis.transpose('time', 'component').values, 'hrf_t': gauss_basis.time.values,
    'hrf_xlabel': 'Time from mnt-correct onset (s)', 'beta_unit': f'{select_chromo} (M)',
})

#%% both designs side by side
print(pd.concat([res_eeg['summary'], res_stim['summary']]).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
