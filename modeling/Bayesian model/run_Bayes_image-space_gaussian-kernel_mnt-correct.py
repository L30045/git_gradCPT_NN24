#%% Joint Bayesian HRF + AR(p) noise for one image-space parcel (mnt-correct, Gaussian kernels)
# Same model as example_hrf_ar_pymc.py, fit on real fNIRS instead of simulated data:
#
#     y = a * (E * h) + N b + e,   h = B w,   e ~ AR(p) with Student-t innovations
#
# - y: HbO of one parcel (image space, sensitive parcels only), per-run highpassed (0.02 Hz) and truncated
#   to [first onset - t_pre, last onset + t_post], runs concatenated, standardized
# - B: consecutive Gaussian kernels (-2 to 18 s, 1 s spacing, 1 s std); E * B are the cedalion
#   hrf_regressors locked to the mnt-correct onsets
# - N: per-run Legendre drift (order 0..DRIFT_ORDER) and per-run GSR. Unlike the 3-stage strategy of
#   run_model_cont_EEG_fNIRS.py (OLS drift -> OLS GSR -> AR-IRLS), the nuisance and event regressors sit
#   in one design matrix and are fit jointly with the HRF and the AR noise. The per-run drift-0 columns
#   are the run intercepts, so the model has no separate b0.
# - the AR innovations are only evaluated where the p-lag history lies inside the same run.
#%% load library
import os
import sys
import gzip
import pickle
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import statsmodels.api as sm
from scipy.stats import gamma
from scipy.signal import butter, sosfiltfilt
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
git_path = os.path.join(modeling_path, os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *
import cedalion.io
import cedalion.models.glm as glm

#%% settings
subject = 'sub-723'
# select_parcel = 'DorsAttnB_PostC_2_RH'
select_parcel = 'SomMotA_25_LH'
select_chromo = 'HbO'
run_keys = ['gradcpt1'] # runs used for fitting, concatenated in this order (e.g. ['gradcpt1', 'gradcpt2', 'gradcpt3'])
is_hp_fNIRS = True   # If True, highpass each run at 0.02 Hz before truncation
l_cutoff = 0.02      # highpass cutoff (Hz)
is_save = True       # If True, save the posterior (netCDF) and figures
USE_GPU = False      # True: JAX/numpyro NUTS on the GPU (stops if JAX sees no GPU); False: PyMC NUTS on CPU
# AR order in samples: start at p_start and add one lag while the last lag of an AR(P) fit to the OLS
# residuals is significant (p < p_alpha), up to pmax
p_start = 10
pmax = 30
p_alpha = 0.05
plot_event_idx = 2   # third mnt-correct event (0-based, across concatenated runs)
plot_win = (-2.0, 18.0)  # seconds around the event onset
# event-based design: 20 consecutive Gaussian kernels (-2 to 18 s) for mnt-correct trials
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
cfg_GLM['drift_order'] = DRIFT_ORDER

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
model_name = f'Bayes-HRF-AR_mnt-correct_gauss_1-stage_{"-".join(run_keys)}_{hp_flag}'
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject, 'bayes')
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, model_name)

gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])

# fail before loading data if the GPU sampler cannot run on a GPU
if USE_GPU:
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError(f"USE_GPU = True but JAX runs on '{jax.default_backend()}' "
                           f"(devices: {jax.devices()}). Run on a GPU node with CUDA-enabled jax.")
    print(f"JAX devices: {jax.devices()}")


# ----------------------------------------------------------------------------
# Model pieces (from example_hrf_ar_pymc.py)
# ----------------------------------------------------------------------------
def canonical_prior_mean(B, t):
    """Double-gamma HRF at lags t (s) projected onto B, scaled to unit peak.

    Returns w0 and Q, an (L, L-1) orthonormal basis of the directions in w-space
    that leave the projection of h = B w onto the canonical shape h0 = B w0 unchanged.
    Writing w = w0 + Q v then fixes <h, h0> / <h0, h0> = 1 exactly, which identifies
    the scale and sign of the bilinear term a_k * h without a soft constraint.
    gamma.pdf is 0 for t < 0, so the pre-onset lags of the Gaussian kernels get a zero prior mean."""
    h_can = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    w0, *_ = np.linalg.lstsq(B, h_can, rcond=None)
    w0 = w0 / (B @ w0).max()
    g = B.T @ (B @ w0)                       # gradient of <B w, h0> w.r.t. w
    Q = np.linalg.svd(g[None, :])[2][1:].T   # null space of g: (L, L-1)
    return w0, Q


def pacf_to_ar(r, P):
    """Durbin-Levinson: partial autocorrelations in (-1, 1) -> stationary AR coefficients.
    phi_k = [phi_{k-1} - r_k * reverse(phi_{k-1}), r_k], one vector op per order (P is a Python int)."""
    phi = r[:1]
    for k in range(1, P):
        phi = pt.concatenate([phi - r[k] * phi[::-1], r[k:k + 1]])
    return phi


def lag_index(run_starts, T, P):
    """Times whose P-lag history lies inside one run, and their lags.
    Returns t_idx (n,) and lag_idx (n, P) with lag_idx[i, k] = t_idx[i] - (k + 1)."""
    bounds = list(run_starts) + [T]
    t_idx = np.concatenate([np.arange(s + P, stop) for s, stop in zip(bounds[:-1], bounds[1:])])
    return t_idx, t_idx[:, None] - np.arange(1, P + 1)[None, :]


def select_ar_order(e, run_starts, p_start, pmax, alpha):
    """Smallest P >= p_start whose last AR lag is not significant (conditional OLS t-test), capped at pmax.
    Each AR(P) is fit on e[t] ~ e[t-1..t-P] within runs; returns P and the last-lag p-value per P."""
    pvals = {}
    P = p_start
    while True:
        t_idx, lag_idx = lag_index(run_starts, len(e), P)
        pvals[P] = sm.OLS(e[t_idx], e[lag_idx]).fit().pvalues[-1]
        print(f"AR({P}): last-lag p = {pvals[P]:.3g}")
        if pvals[P] >= alpha or P == pmax:
            return P, pvals
        P += 1


def build_model(x, D, N, B, w0, Q, P, run_starts, pacf_scale=1.0):
    K, T, L = D.shape
    M = N.shape[1]
    lag_scale = pacf_scale / np.arange(1, P + 1)
    t_idx, lag_idx = lag_index(run_starts, T, P)

    with pm.Model() as bayes_model:
        # ---- HRF shape: w = w0 + Q v keeps the projection on the canonical HRF at 1 ----
        tau = pm.HalfNormal("tau", 0.5)
        v = pm.Normal("v", 0.0, 1.0, shape=L - 1)          # deviation from canonical (weak ridge)
        dev = pt.dot(Q, v)
        w = pm.Deterministic("w", w0 + dev)
        # smoothness: 2nd differences of the deviation, scale learned
        pm.Potential("smooth", pm.logp(pm.Normal.dist(0.0, tau), pt.diff(dev, n=2)).sum())
        pm.Deterministic("hrf", pt.dot(B, w))

        # ---- amplitudes and nuisance (per-run drift-0 columns are the intercepts) ----
        a = pm.Normal("a", 0.0, 1.0, shape=K)
        b = pm.Normal("b", 0.0, 5.0, shape=M)

        # ---- AR noise: unconstrained PACF -> stationary phi ----
        z = pm.Normal("z", 0.0, lag_scale, shape=P)
        phi = pm.Deterministic("phi", pacf_to_ar(pt.tanh(z), P))
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Deterministic("nu", 2.0 + pm.Gamma("nu_minus2", 2.0, 0.1))

        # ---- mean and residual ----
        Dw = pt.tensordot(D, w, axes=[[2], [0]])            # (K, T)
        mu = pt.dot(a, Dw) + pt.dot(N, b)
        e = x - mu

        # ---- whiten inside the model (vectorized): innov[t] = e[t] - sum_k phi_k e[t-k] ----
        # one gather of the (n, P) lag matrix and one matrix-vector product instead of P slices
        innov = e[t_idx] - pt.dot(e[lag_idx], phi)

        # The map e -> innov is unit lower-triangular (Jacobian = 1), so this Potential is the
        # exact conditional likelihood (conditioned on the first P samples of each run).
        pm.Potential("lik", pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), innov).sum())

    return bayes_model


def ar_one_step(e, phi, run_starts):
    """One-step AR prediction sum_k phi_k e[t-k], restarted at each run (0 for the first P samples)."""
    t_idx, lag_idx = lag_index(run_starts, len(e), len(phi))
    pred = np.zeros_like(e)
    pred[t_idx] = e[lag_idx] @ phi
    return pred


# ----------------------------------------------------------------------------
# Figure: posterior summary (same layout as plot_recovery in example_hrf_ar_pymc.py, no ground truth)
# ----------------------------------------------------------------------------
def plot_result(idata, hrf_t, h0, t, x_s, fit_s, event_s, ev_times, P, fs, title, path=None):
    """t: time (s) of each sample of x_s; ev_times: event onsets (s) on the same axis."""
    post = idata.posterior
    hrf = post["hrf"].stack(s=("chain", "draw")).values.T            # (draws, n_lag)
    phi = post["phi"].stack(s=("chain", "draw")).values.T            # (draws, P)
    a = post["a"].stack(s=("chain", "draw")).values.T                # (draws, K)

    est_c, prior_c = "#2a6fdb", "#9a9a9a"
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle(title)

    # (1) HRF
    lo, hi = np.percentile(hrf, [2.5, 97.5], 0)
    a0 = ax[0, 0]
    a0.fill_between(hrf_t, lo, hi, color=est_c, alpha=0.2, lw=0, label="95% credible band")
    a0.plot(hrf_t, hrf.mean(0), color=est_c, lw=2, label="Posterior mean")
    a0.plot(hrf_t, h0, color=prior_c, lw=1.2, ls=":", label="Canonical (prior mean)")
    a0.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    a0.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    a0.set(title="HRF shape", xlabel="Lag (s)", ylabel="h (projection on canonical = 1)")
    a0.legend(frameon=False, fontsize=9)

    # (2) AR coefficients
    a1 = ax[0, 1]
    lags = np.arange(1, P + 1)
    plo, phi_hi = np.percentile(phi, [2.5, 97.5], 0)
    a1.errorbar(lags, phi.mean(0), yerr=[phi.mean(0) - plo, phi_hi - phi.mean(0)],
                fmt="o", color=est_c, capsize=4, label="Posterior mean, 95% CI")
    a1.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    a1.set(title="AR coefficients φ", xlabel="Lag", ylabel="φ", xticks=lags)
    a1.legend(frameon=False, fontsize=9)

    # (3) Condition amplitudes
    a2 = ax[1, 0]
    for k in range(a.shape[1]):
        a2.hist(a[:, k], bins=40, color=f"C{k}", alpha=0.55, label="mnt-correct posterior")
    a2.axvline(0, color="#1f1f1f", lw=1, ls="--")
    a2.set(title="Amplitude a (standardized units)", xlabel="a", ylabel="Draws")
    a2.legend(frameon=False, fontsize=9)

    # (4) Data and fit, first 2 minutes
    a3 = ax[1, 1]
    n = int(120 * fs)
    ts = t[:n]
    a3.plot(ts, x_s[:n], color="#bbbbbb", lw=0.8, label="Data (standardized)")
    a3.plot(ts, fit_s[:n], color="#1f1f1f", lw=1.2, label="Posterior mean fit (event + nuisance)")
    a3.plot(ts, event_s[:n], color=est_c, lw=1.6, label="Posterior mean evoked")
    for i, t_o in enumerate(ev_times[(ev_times >= ts[0]) & (ev_times <= ts[-1])]):
        a3.axvline(t_o, color="#d1495b", lw=0.8, ls="--", label="mnt-correct onset" if i == 0 else None)
    a3.set(title="Data and fitted response (first 2 min)", xlabel="Time (s)", ylabel="y")
    a3.legend(frameon=False, fontsize=9, loc="upper right")

    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
    return fig


#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
# (same mask as run_model_cont_EEG_fNIRS.py; GSR is the mean over these parcels)
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
sensitive_parcels = np.unique(Adot_brain.sel(vertex=sensitivity_mask.values).parcel.values)
if select_parcel not in sensitive_parcels:
    raise ValueError(f"{select_parcel} is not among the {len(sensitive_parcels)} sensitive parcels.")

#%% load image-space parcel time series and stims
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    all_stims = pickle.load(f)['stims']
filepath = os.path.join(der_dir, subject,
                        f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl')
with open(filepath, 'rb') as f:
    image_results = pickle.load(f)
raw_runs = image_results['parcel_ts']
if len(raw_runs) != len(all_stims):
    raise ValueError(f"{len(raw_runs)} parcel runs but {len(all_stims)} stim tables.")

# match each run key to its image-space run via the first onset in the run's nirs events.tsv
run_key_to_run_idx = dict()
for run_key in run_keys:
    nirs_ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                          f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv"), sep='\t')
    for r_i, stim in enumerate(all_stims):
        if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_ev_df['onset'].values[0], atol=0.01):
            run_key_to_run_idx[run_key] = r_i
            break
    else:
        raise ValueError(f"{subject}: no image-space run matches {run_key}.")

#%% per run: highpass, truncate to the event window, mnt-correct Gaussian-kernel regressors
all_runs = []
event_dms = []
mnt_onsets = []   # per run, mnt-correct onsets in the truncated run's time (starts at 0)
for run_key in run_keys:
    r_i = run_key_to_run_idx[run_key]
    stim, run = all_stims[r_i], raw_runs[r_i]
    run = run.sel(parcel=run.parcel.isin(sensitive_parcels)).sel(chromo=[select_chromo])
    run = run.pint.to('micromolar')
    run.time.attrs['units'] = units.s
    fs = 1 / np.median(np.diff(run.time.values))

    if is_hp_fNIRS:
        run_units = run.pint.units
        sos = butter(4, l_cutoff, btype='highpass', fs=fs, output='sos')
        run = xr.apply_ufunc(
            sosfiltfilt, sos, run.pint.dequantify(),
            input_core_dims=[[], ['time']],
            output_core_dims=[['time']],
            exclude_dims={'time'},
        ).transpose(*run.dims).pint.quantify(run_units).assign_coords(time=run.time)
        run.time.attrs['units'] = units.s

    mnt = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
    mnt['trial_type'] = 'mnt-correct'

    # window: first event - t_pre to last event + t_post, so every kernel of every kept trial fits
    t_start = max(stim['onset'].min() - cfg_HRF_basis['t_pre'].magnitude, run.time.values[0])
    t_stop = min(stim['onset'].max() + cfg_HRF_basis['t_post'].magnitude, run.time.values[-1])
    run = run.sel(time=slice(t_start, t_stop))
    mnt = mnt[(mnt['onset'] >= t_start) & (mnt['onset'] <= t_stop)]

    ev_dm = glm.design_matrix.hrf_regressors(run, mnt, gauss_kernels)
    ev_dm.common = ev_dm.common.fillna(0)

    # reset time to 0 (regressors and onsets shift with it)
    t0 = run.time.values[0]
    run = run.assign_coords(time=run.time.values - t0)
    run.time.attrs['units'] = units.s
    ev_dm.common = ev_dm.common.assign_coords(time=run.time.values)
    all_runs.append(run)
    event_dms.append(ev_dm)
    mnt_onsets.append(mnt['onset'].values - t0)
    print(f"{run_key} (run {r_i}): {run.sizes['time']} samples, {len(mnt)} mnt-correct trials")

gauss_basis = gauss_kernels(all_runs[0]).transpose('time', 'component')  # (HRF lag x kernel)
B = gauss_basis.values
hrf_t = gauss_basis.time.values

#%% one design matrix: Gaussian kernels + per-run drift + per-run GSR
Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, event_dms)
# concatenate_runs_dms relabels the data as molar; the values are in micromolar (converted above)
event_names = list(dm_all.common.regressor.values)

drift_regressors = model.get_drift_legendre_regressors(runs_updated, cfg_GLM)
dm_all &= model.reduce(model.operator.and_, drift_regressors)
gsr = model.get_global_mean_regressor(runs_updated)
dm_all &= model.reduce(model.operator.and_, gsr)
dm_all.common = dm_all.common.fillna(0).sel(chromo=[select_chromo])

regressors = dm_all.common.regressor.values
nuis_names = [r for r in regressors if r not in event_names]
# GSR columns are in micromolar; scale each to unit std within its run so the N(0, 5) prior on b
# is on the same footing as the Legendre columns (in [-1, 1])
for r in nuis_names:
    if r.startswith('GS run'):
        col = dm_all.common.sel(regressor=r)
        dm_all.common.loc[dict(regressor=r)] = col / col.where(col != 0).std()

model.vis_dm(dm_all)

#%% model inputs
X_dm = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
X_ev = X_dm.sel(regressor=event_names).values         # (T, L)
N = X_dm.sel(regressor=nuis_names).values             # (T, M)
y = Y_all.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify().values  # micromolar
T = len(y)
run_starts = np.r_[0, np.cumsum([r.sizes['time'] for r in runs_updated])[:-1]]
run_offsets = [r.time.values[0] for r in runs_updated]  # concatenated time of each run's t = 0
fs = 1 / np.median(np.diff(all_runs[0].time.values))
if B.shape[1] != X_ev.shape[1]:
    raise ValueError(f"{B.shape[1]} Gaussian kernels but {X_ev.shape[1]} event regressors.")

y_mean, scale = y.mean(), y.std()
y_s = (y - y_mean) / scale
D = X_ev[None]                                         # (K=1, T, L)
w0, Q = canonical_prior_mean(B, hrf_t)

#%% AR order from the OLS residuals of the full design (events + nuisance)
X_full = np.column_stack([X_ev, N])
e_ols = y_s - X_full @ np.linalg.lstsq(X_full, y_s, rcond=None)[0]
P, ar_pvals = select_ar_order(e_ols, run_starts, p_start, pmax, p_alpha)
if P == pmax and ar_pvals[P] < p_alpha:
    print(f"warning: last lag still significant at pmax = {pmax}")
print(f"{subject} {select_parcel}: T = {T}, fs = {fs:.2f} Hz, L = {B.shape[1]}, M = {N.shape[1]}, P = {P}")

bayes_model = build_model(y_s, D, N, B, w0, Q, P, run_starts)

#%% sample
t_begin = time.time()
with bayes_model:
    if USE_GPU:
        # the model is compiled to JAX; the 4 chains are vmapped on the one GPU
        idata = pm.sample(draws=500, tune=500, chains=4, target_accept=0.9,
                          nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "vectorized"},
                          random_seed=1, progressbar=False)
    else:
        idata = pm.sample(draws=500, tune=500, chains=4, cores=4, target_accept=0.9,
                          nuts_sampler="pymc", random_seed=1, progressbar=False)
print(f"sampling time {time.time() - t_begin:.0f} s")

# ---- diagnostics ----
print("divergences:", int(idata.sample_stats["diverging"].sum()))
print(az.summary(idata, var_names=["a", "phi", "sigma", "nu", "tau"], round_to=3))

#%% posterior-mean fit pieces (standardized units)
post = idata.posterior.stack(s=("chain", "draw"))
beta_ev = (post["a"].values[0] * post["w"].values).mean(-1)   # E[a w]: (L,)
b_mean = post["b"].values.mean(-1)
phi_mean = post["phi"].values.mean(-1)

event_s = X_ev @ beta_ev                     # evoked part
nuis_s = N @ b_mean                          # drift + GSR
mu_s = event_s + nuis_s
e_pred_s = ar_one_step(y_s - mu_s, phi_mean, run_starts)  # autocorrelated noise predicted from the past

# back to micromolar: y - yhat and y_partial - yhat_event are both the innovations
yhat = (mu_s + e_pred_s) * scale + y_mean
y_partial = (y_s - nuis_s - e_pred_s) * scale
yhat_event = event_s * scale
hrf_uM = (post["a"].values[0] * post["hrf"].values).mean(-1) * scale  # a * h in micromolar
print(f"peak of a*h: {hrf_uM.max():.4g} uM at {hrf_t[hrf_uM.argmax()]:.1f} s")

#%% figures
if is_save:
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_save_path, exist_ok=True)

title = f"{subject} {select_parcel} {select_chromo} (mnt-correct)"
ev_times = np.sort(np.concatenate([ons + off for ons, off in zip(mnt_onsets, run_offsets)]))  # concatenated time
t_all = Y_all.time.values
plot_result(idata, hrf_t, B @ w0, t_all, y_s, mu_s, event_s, ev_times, P, fs, title,
            path=os.path.join(plot_dir, f'{subject}_{select_parcel}_posterior.png') if is_save else None)

# 20 s around the third mnt-correct event
t_ev = ev_times[plot_event_idx]
in_win = (t_all >= t_ev + plot_win[0]) & (t_all <= t_ev + plot_win[1])
t_rel = t_all[in_win] - t_ev
other_ev = ev_times[(ev_times >= t_ev + plot_win[0]) & (ev_times <= t_ev + plot_win[1]) & (ev_times != t_ev)] - t_ev

fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
axs[0].plot(t_rel, y[in_win], color='k', lw=1.5, label='y')
axs[0].plot(t_rel, yhat[in_win], color='#2a6fdb', lw=1.5, label='yhat (event + nuisance + AR)')
axs[0].set(ylabel=f'{select_chromo} (µM)', title='Full fit')
axs[1].plot(t_rel, y_partial[in_win], color='k', lw=1.5, label='y_partial (y − nuisance − AR noise)')
axs[1].plot(t_rel, yhat_event[in_win], color='#d1495b', lw=1.5, label='yhat_event')
axs[1].set(xlabel=f'Time from mnt-correct event #{plot_event_idx + 1} (s)', ylabel=f'{select_chromo} (µM)',
           title='Event part (nuisance and autocorrelated noise removed)')
for ax in axs:
    ax.axvline(0, color='#888888', ls='--', lw=1)
    for t_o in other_ev:
        ax.axvline(t_o, color='#bbbbbb', ls=':', lw=1)
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle(f"{title}, event at {t_ev:.1f} s (dotted: other mnt-correct onsets)")
if is_save:
    path = os.path.join(plot_dir, f'{subject}_{select_parcel}_event{plot_event_idx + 1}_fit.png')
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
plt.show()

#%% save posterior
if is_save:
    idata_path = os.path.join(data_save_path, f'{subject}_{model_name}_{select_parcel}_idata.nc')
    try:
        idata.to_netcdf(idata_path)          # needs netCDF4 or h5netcdf
        print(f"saved {idata_path}")
    except ValueError as err:
        print(f"posterior not saved: {err}")
