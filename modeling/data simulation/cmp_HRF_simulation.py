#%% Compare the Bayesian HRF posterior with the simulated ground truth
# Inputs:
# - ground truth from simulate_mnt-correct_HRF.py: <sim dir>/<subject>_sim_ground_truth.csv (A, s, onset shift per
#   simulated mnt-correct trial)
# - posterior from run_Bayes_image-space_gaussian-kernel_mnt-correct.py run with sim_name set:
#   <sim dir>/bayes/<subject>_<model_name>_<parcel>_idata.nc. Its constant_data group holds the HRF lag axis,
#   the y scale (standardized -> uM) and the image-space runs that were fit.
#
# Truth: the per-trial injected HRFs on the model's lag axis, locked to the labeled onset (so the onset jitter is
# in them), for the trials of the fitted runs. Their mean is the response the model should recover; its 95% CI is
# t-based across trials (uncertainty of the trial mean).
# Estimate: posterior of B h in uM, mean and 95% equal-tailed credible band.
#
# The model fits 0.02 Hz highpassed data, so the slow tail of the truth is slightly attenuated in what the model
# sees; the truth plotted here is the unfiltered injected HRF.
#%% load library
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import t as t_dist
from scipy.signal import butter, sosfiltfilt
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from params_setting import *
from sim_utils import true_trial_hrfs
import cedalion.models.glm as glm

#%% settings (match the Bayesian run)
subject = 'sub-723'
select_parcel = 'DorsAttnB_PostC_2_RH'
select_chromo = 'HbO'
sim_name = 'sim_mnt-correct_10pct-city_canonical_A0.0158_cv0.25_slogsd0.15_jit3_seed0'
run_keys = ['gradcpt1']
is_hp_fNIRS = True
l_cutoff = 0.02          # highpass (Hz) of the Bayesian run
# same as the Bayesian run; t_pre/t_post also set its trial window (first onset - t_pre to last onset + t_post)
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
is_smooth_penalty = False
is_hp_design = True      # main figure: the Bayes + ridge fit whose event regressors were highpassed like y
peak_win = (0.0, 18.0)   # lag window (s) for the peak amplitude and latency
is_save = True

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
smooth_flag = 'smooth' if is_smooth_penalty else 'noSmooth'
model_base = f'Bayes-HRF-AR_mnt-correct_gauss_1-stage_{"-".join(run_keys)}_{hp_flag}_{smooth_flag}'
sim_dir = os.path.join(project_path, 'derivatives', 'eeg', subject, 'simulation', sim_name)


def bayes_idata_path(is_ridge, is_hp_design):
    """Posterior file of run_Bayes_image-space_gaussian-kernel_mnt-correct.py for these two switches."""
    name = model_base + ('' if is_ridge else '_noRidge') + ('_hpDesign' if is_hp_design and is_hp_fNIRS else '')
    return os.path.join(sim_dir, 'bayes', f'{subject}_{name}_{select_parcel}_idata.nc')


model_name = model_base + ('_hpDesign' if is_hp_design and is_hp_fNIRS else '')
idata_path = bayes_idata_path(True, is_hp_design)
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, sim_name, model_name)

c_est, c_true = '#2a78d6', '#eb6834'   # validated categorical slots 1-2 (Bayes + ridge, truth)


#%% load posterior and ground truth
idata = az.from_netcdf(idata_path)
if 'constant_data' not in idata.groups():
    raise ValueError(f"{idata_path} has no constant_data (hrf_t, y_scale, run_idx). "
                     "Refit with the current run_Bayes_image-space_gaussian-kernel_mnt-correct.py.")
cd = idata.constant_data
if cd.attrs.get('sim_name') != sim_name or cd.attrs.get('parcel') != select_parcel:
    raise ValueError(f"idata is for sim '{cd.attrs.get('sim_name')}', parcel '{cd.attrs.get('parcel')}'.")
hrf_t = cd['hrf_t'].values
y_scale = float(cd['y_scale'])
run_idx = cd['run_idx'].values

hrf_post = idata.posterior['hrf'].stack(s=('chain', 'draw')).values.T * y_scale   # (draws, lag), uM
est_mean = hrf_post.mean(0)
est_lo, est_hi = np.percentile(hrf_post, [2.5, 97.5], 0)

truth = pd.read_csv(os.path.join(sim_dir, f'{subject}_sim_ground_truth.csv'))
truth = truth[truth['run_idx'].isin(run_idx)]
true_hrfs = true_trial_hrfs(truth, hrf_t)                                         # (trials, lag), uM
n_tr = len(true_hrfs)
true_mean = true_hrfs.mean(0)
h_ci = t_dist.ppf(0.975, n_tr - 1) * true_hrfs.std(0, ddof=1) / np.sqrt(n_tr)
true_lo, true_hi = true_mean - h_ci, true_mean + h_ci
print(f"{subject} {select_parcel}: {n_tr} simulated trials in runs {list(run_idx)}, "
      f"{hrf_post.shape[0]} posterior draws")


#%% metrics
in_pk = (hrf_t >= peak_win[0]) & (hrf_t <= peak_win[1])
pk_t = hrf_t[in_pk]

def peak(x):
    """Peak amplitude and latency of HRF(s) x (..., lag) within peak_win."""
    x = x[..., in_pk]
    return x.max(-1), pk_t[x.argmax(-1)]

est_pk_amp, est_pk_lat = peak(hrf_post)          # per draw
true_pk_amp, true_pk_lat = peak(true_mean)

diff = hrf_post - true_mean                      # per draw
summary = pd.Series({
    'n_trials': n_tr,
    'corr(post mean, true mean)': np.corrcoef(est_mean, true_mean)[0, 1],
    'RMSE (uM)': np.sqrt(np.mean((est_mean - true_mean) ** 2)),
    'RMSE / true peak': np.sqrt(np.mean((est_mean - true_mean) ** 2)) / true_pk_amp,
    'coverage: true mean in 95% CrI': np.mean((true_mean >= est_lo) & (true_mean <= est_hi)),
    'true peak (uM)': true_pk_amp,
    'post peak mean (uM)': est_pk_amp.mean(),
    'post peak 95% CrI lo': np.percentile(est_pk_amp, 2.5),
    'post peak 95% CrI hi': np.percentile(est_pk_amp, 97.5),
    'true peak latency (s)': true_pk_lat,
    'post latency median (s)': np.median(est_pk_lat),
    'post latency 95% CrI lo': np.percentile(est_pk_lat, 2.5),
    'post latency 95% CrI hi': np.percentile(est_pk_lat, 97.5),
})
print(summary.round(3).to_string())


#%% figure
def style(ax):
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True,
                        gridspec_kw={'width_ratios': [2, 1.4, 1, 1]})

a = axs[0]
a.fill_between(hrf_t, true_lo, true_hi, color=c_true, alpha=0.25, lw=0)
a.plot(hrf_t, true_mean, color=c_true, lw=2, label=f'Ground truth: trial mean, 95% CI (n = {n_tr})')
a.fill_between(hrf_t, est_lo, est_hi, color=c_est, alpha=0.2, lw=0)
a.plot(hrf_t, est_mean, color=c_est, lw=2, label='Bayes: posterior mean, 95% CrI')
a.axvline(0, color='#cccccc', lw=0.8, zorder=0)
a.set(xlabel='Lag (s)', ylabel='HbO (µM)', title='HRF: posterior vs ground truth')
a.legend(frameon=False, fontsize=9, loc='upper right')
style(a)

b = axs[1]
d_lo, d_hi = np.percentile(diff, [2.5, 97.5], 0)
b.fill_between(hrf_t, d_lo, d_hi, color=c_est, alpha=0.2, lw=0)
b.plot(hrf_t, diff.mean(0), color=c_est, lw=2)
b.axvline(0, color='#cccccc', lw=0.8, zorder=0)
b.set(xlabel='Lag (s)', ylabel='µM', title='Posterior − true mean, 95% CrI')
style(b)

c = axs[2]
c.hist(est_pk_amp, bins=40, color=c_est, alpha=0.8, edgecolor='white', lw=0.4)
c.axvline(true_pk_amp, color=c_true, lw=2, label='Ground truth')
c.set(xlabel='Peak amplitude (µM)', ylabel='Draws', title='Peak amplitude')
c.legend(frameon=False, fontsize=9)
style(c)

e = axs[3]
step = np.median(np.diff(pk_t))
e.hist(est_pk_lat, bins=np.arange(pk_t.min(), pk_t.max() + step, step * 2), color=c_est, alpha=0.8,
       edgecolor='white', lw=0.4)
e.axvline(true_pk_lat, color=c_true, lw=2, label='Ground truth')
lat_rng = np.r_[est_pk_lat, true_pk_lat]
e.set_xlim(lat_rng.min() - 1.5, lat_rng.max() + 1.5)
e.set(xlabel='Peak latency (s)', ylabel='Draws', title='Peak latency')
e.legend(frameon=False, fontsize=9, loc='upper left')
style(e)

fig.suptitle(f"{subject} {select_parcel} HbO, {'-'.join(run_keys)}: Bayesian HRF vs simulated truth   "
             f"(r = {summary['corr(post mean, true mean)']:.2f}, "
             f"coverage = {summary['coverage: true mean in 95% CrI']:.0%})")
if is_save:
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f'{subject}_{select_parcel}_cmp_HRF_truth.png')
    fig.savefig(path, dpi=150)
    summary.to_csv(os.path.join(plot_dir, f'{subject}_{select_parcel}_cmp_HRF_truth_summary.csv'))
    print(f"saved {path}")


#%% shrinkage diagnosis: refit the same design with simpler estimators
# Each step adds one piece of the Bayesian model, so the step where the amplitude drops is the cause:
#   OLS, injected only, no highpass : OLS of the noise-free injected HRFs, truncated like y but NOT highpassed.
#                           What the kernel basis and design alone leave of the truth.
#   OLS, injected only    : the same, highpassed like y. What the preprocessing and design leave of the truth
#                           (no noise at all); the step above -> this one is the loss from the highpass.
#   OLS                   : OLS on y. Unbiased under any noise; its CI ignores the autocorrelation (too narrow).
#   AR-GLS                : OLS after AR(P) prewhitening (P as in the Bayes fit; AR fit on the OLS residuals
#                           within runs, 2 Cochrane-Orcutt iterations). Gaussian, no prior on h.
#   Bayes, no ridge       : the joint Bayesian AR model, Student-t innovations, h ~ N(0, 10) fixed.
#   Bayes + ridge         : the full model, h ~ N(0, tau_h), tau_h learned.
# Row A uses the design as the Bayesian script built it without is_hp_design (event columns not filtered).
# Row B refits everything with the event columns highpassed like y (filtered over the full run, then truncated);
# its Bayes fits come from the Bayesian script run with is_hp_design = True. Row B should undo the highpass loss.
def lag_index(run_starts, T, P):
    """Times whose P-lag history lies inside one run, and their lags (as in the Bayesian script)."""
    bounds = list(run_starts) + [T]
    t_idx = np.concatenate([np.arange(s + P, stop) for s, stop in zip(bounds[:-1], bounds[1:])])
    return t_idx, t_idx[:, None] - np.arange(1, P + 1)[None, :]


def ols(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    s2 = r @ r / (len(y) - X.shape[1])
    return beta, s2 * np.linalg.pinv(X.T @ X)


def ar_gls(y, X, run_starts, P, n_iter=2):
    beta, cov = ols(y, X)
    t_idx, lag_idx = lag_index(run_starts, len(y), P)
    for _ in range(n_iter):
        e = y - X @ beta
        phi = np.linalg.lstsq(e[lag_idx], e[t_idx], rcond=None)[0]
        yw = y[t_idx] - y[lag_idx] @ phi
        Xw = X[t_idx] - np.einsum('nkc,k->nc', X[lag_idx], phi)
        beta, cov = ols(yw, Xw)
    return beta, cov


def kernel_fit_to_hrf(beta, cov, scale):
    """Kernel weights (first L entries) -> HRF mean, lo, hi (95%, Gaussian) in uM."""
    L = B.shape[1]
    m = B @ beta[:L] * scale
    sd = np.sqrt(np.einsum('ij,jk,ik->i', B, cov[:L, :L], B)) * scale
    return m, m - 1.96 * sd, m + 1.96 * sd


def posterior_to_hrf(path):
    if not os.path.exists(path):
        print(f"missing {path}")
        return None
    idt = az.from_netcdf(path)
    draws = idt.posterior['hrf'].stack(s=('chain', 'draw')).values.T * float(idt.constant_data['y_scale'])
    lo, hi = np.percentile(draws, [2.5, 97.5], 0)
    return dict(mean=draws.mean(0), lo=lo, hi=hi, draws=draws)


# full simulated runs (image-space run order), for the pieces the Bayesian run does not store
with open(os.path.join(sim_dir, f'{subject}_sim_injected_uM.pkl'), 'rb') as f:
    sim_injected = pickle.load(f)
with gzip.open(os.path.join(sim_dir, f'{subject}_sim_stims.pkl'), 'rb') as f:
    sim_stims = pickle.load(f)['stims']
with open(os.path.join(sim_dir, f'{subject}_sim_parcel_ts.pkl'), 'rb') as f:
    sim_runs = [r.sel(parcel=[select_parcel], chromo=[select_chromo]) for r in pickle.load(f)['parcel_ts']]
gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])


def hp_filter(x, t):
    """The Bayesian script's highpass: 4th-order Butterworth at l_cutoff, zero-phase, along axis 0."""
    fs = 1 / np.median(np.diff(t))
    return sosfiltfilt(butter(4, l_cutoff, btype='highpass', fs=fs, output='sos'), x, axis=0)


def trial_window(r):
    """Run r's samples in the Bayesian trial window: first event - t_pre to last event + t_post."""
    t, onset = sim_runs[r].time.values, sim_stims[r]['onset']
    t_start = max(onset.min() - cfg_HRF_basis['t_pre'].magnitude, t[0])
    t_stop = min(onset.max() + cfg_HRF_basis['t_post'].magnitude, t[-1])
    return (t >= t_start) & (t <= t_stop), t_start, t_stop


def injected_in_window(run_idx, is_hp):
    """Injected HRFs (uM) of the fitted runs, cut to the trial window and concatenated; highpassed over the
    full run first if is_hp (as the Bayesian script does to y)."""
    out = []
    for r in run_idx:
        inj = hp_filter(sim_injected[r], sim_runs[r].time.values) if is_hp else sim_injected[r]
        out.append(inj[trial_window(r)[0]])
    return np.concatenate(out)


def event_regressors_in_window(run_idx, is_hp):
    """mnt-correct Gaussian-kernel regressors of the fitted runs, built on the full run (as the Bayesian
    script builds them on the truncated run), highpassed over the full run if is_hp, cut to the trial window."""
    out = []
    for r in run_idx:
        run = sim_runs[r].pint.to('micromolar')
        run.time.attrs['units'] = units.s
        keep, t_start, t_stop = trial_window(r)
        stim = sim_stims[r]
        mnt = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
        mnt['trial_type'] = 'mnt-correct'
        mnt = mnt[(mnt['onset'] >= t_start) & (mnt['onset'] <= t_stop)]
        ev = glm.design_matrix.hrf_regressors(run, mnt, gauss_kernels).common.fillna(0)
        ev = ev.sel(chromo=select_chromo).transpose('time', 'regressor').values
        if is_hp:
            ev = hp_filter(ev, run.time.values)
        out.append(ev[keep])
    return np.concatenate(out)


for v in ('X_ev', 'N', 'y_s', 'run_starts', 'P', 'B', 'y_inj'):
    if v not in cd:
        raise ValueError(f"idata constant_data has no '{v}'. Refit with the current Bayesian script.")
y_s = cd['y_s'].values
N = cd['N'].values
B = cd['B'].values
P = int(cd['P'])
run_starts = cd['run_starts'].values

# rebuild the injected signal and the event regressors, and check both against what the Bayesian run saved
y_inj_check = injected_in_window(run_idx, is_hp_fNIRS)
if len(y_inj_check) != len(y_s) or not np.allclose(y_inj_check, cd['y_inj'].values):
    raise ValueError("Rebuilt injected signal does not match the Bayesian run's y_inj: "
                     "check l_cutoff, cfg_HRF_basis and run_keys against the Bayesian script.")
X_ev_raw = event_regressors_in_window(run_idx, is_hp=False)
X_ev_hp = event_regressors_in_window(run_idx, is_hp=True)
if not np.allclose(X_ev_hp if cd.attrs.get('hp_design', 0) else X_ev_raw, cd['X_ev'].values):
    raise ValueError("Rebuilt event regressors do not match the Bayesian run's X_ev: "
                     "check l_cutoff, cfg_HRF_basis and run_keys against the Bayesian script.")
designs = {'A': np.column_stack([X_ev_raw, N]),    # event columns as built (not filtered)
           'B': np.column_stack([X_ev_hp, N])}     # event columns highpassed like y
row_title = {'A': 'A. design not highpassed', 'B': 'B. design highpassed like y'}


def ols_noise_free(y, X):
    return dict(mean=B @ ols(y, X)[0][:B.shape[1]], lo=None, hi=None)


def band(beta_cov):
    return dict(zip(('mean', 'lo', 'hi'), kernel_fit_to_hrf(*beta_cov, y_scale)))


# fits[(row, estimator)] = dict(mean, lo, hi[, draws]) in uM
fits = {('A', 'OLS, injected only, no highpass'): ols_noise_free(injected_in_window(run_idx, is_hp=False), designs['A'])}
for row, X in designs.items():
    fits[(row, 'OLS, injected only')] = ols_noise_free(cd['y_inj'].values, X)
    fits[(row, 'OLS')] = band(ols(y_s, X))
    fits[(row, f'AR({P})-GLS')] = band(ar_gls(y_s, X, run_starts, P))
    for est, ridge in (('Bayes, no ridge', False), ('Bayes + ridge', True)):
        post_fit = posterior_to_hrf(bayes_idata_path(ridge, row == 'B'))
        if post_fit is not None:
            fits[(row, est)] = post_fit

recoverable_pk = peak(fits[('A', 'OLS, injected only')]['mean'])[0]
rows = {}
for (row, est), f in fits.items():
    pk_amp, pk_lat = peak(f['mean'])
    r = {'design': row_title[row], 'estimator': est,
         'peak (uM)': pk_amp,
         'peak / true peak': pk_amp / true_pk_amp,
         'peak / injected-only peak (A)': pk_amp / recoverable_pk,
         'peak latency (s)': pk_lat,
         'corr with truth': np.corrcoef(f['mean'], true_mean)[0, 1],
         'RMSE (uM)': np.sqrt(np.mean((f['mean'] - true_mean) ** 2))}
    if 'draws' in f:   # peak interval from the draws' own peaks
        a_d, _ = peak(f['draws'])
        r['peak 95% lo'], r['peak 95% hi'] = np.percentile(a_d, [2.5, 97.5])
    elif f['lo'] is not None:   # peak interval: the band at the peak lag
        i = np.flatnonzero(in_pk)[np.argmax(f['mean'][in_pk])]
        r['peak 95% lo'], r['peak 95% hi'] = f['lo'][i], f['hi'][i]
    if f['lo'] is not None:
        r['coverage of true mean'] = np.mean((true_mean >= f['lo']) & (true_mean <= f['hi']))
    rows[(row, est)] = r
diag = pd.DataFrame(rows.values(), index=pd.MultiIndex.from_tuples(rows.keys(), names=['row', 'estimator']))
print(f"\ntrue mean peak {true_pk_amp:.3f} uM at {true_pk_lat:.2f} s")
print(diag.drop(columns=['design', 'estimator']).round(3).to_string())

# figure: estimator columns, design rows (A, B), then all peak amplitudes side by side
est_order = ['OLS, injected only, no highpass', 'OLS, injected only', 'OLS', f'AR({P})-GLS',
             'Bayes, no ridge', 'Bayes + ridge']
fit_colors = {'OLS, injected only, no highpass': '#a3a29c', 'OLS, injected only': '#52514e', 'OLS': '#e87ba4',
              f'AR({P})-GLS': '#eda100', 'Bayes, no ridge': '#1baf7a', 'Bayes + ridge': c_est}
n_col = len(est_order)
fig = plt.figure(figsize=(3.6 * n_col, 14), constrained_layout=True)
gs = fig.add_gridspec(3, n_col, height_ratios=[1, 1, 1.15])
ax_first = None
for i_r, row in enumerate(('A', 'B')):
    for k, est in enumerate(est_order):
        if (row, est) not in fits:
            continue
        f, c = fits[(row, est)], fit_colors[est]
        a = fig.add_subplot(gs[i_r, k], sharey=ax_first)
        ax_first = ax_first or a
        a.fill_between(hrf_t, true_lo, true_hi, color=c_true, alpha=0.25, lw=0)
        a.plot(hrf_t, true_mean, color=c_true, lw=1.8, label='Ground truth')
        if f['lo'] is not None:
            a.fill_between(hrf_t, f['lo'], f['hi'], color=c, alpha=0.2, lw=0)
        a.plot(hrf_t, f['mean'], color=c, lw=2, label=est)
        a.axvline(0, color='#cccccc', lw=0.8, zorder=0)
        a.set(title=f"{row}{k + 1}. {est}\npeak = {diag.loc[(row, est), 'peak / true peak']:.0%} of truth",
              xlabel='Lag (s)', ylabel=f'{row_title[row]}\nHbO (µM)' if k == 0 or (row == 'B' and k == 1) else None)
        a.legend(frameon=False, fontsize=8, loc='upper right')
        style(a)

a = fig.add_subplot(gs[2, :])
order = [(row, est) for row in ('A', 'B') for est in est_order if (row, est) in fits]
ypos = np.arange(len(order))[::-1].astype(float)
ypos[[i for i, (row, _) in enumerate(order) if row == 'B']] -= 0.6   # gap between the two design rows
for y_, key in zip(ypos, order):
    r, c = diag.loc[key], fit_colors[key[1]]
    if np.isfinite(r.get('peak 95% lo', np.nan)):
        a.plot([r['peak 95% lo'], r['peak 95% hi']], [y_, y_], color=c, lw=2.5, solid_capstyle='round')
    a.plot(r['peak (uM)'], y_, 'o' if key[0] == 'A' else 's', ms=8, color=c, mec='white', mew=1.5)
    a.text(r['peak (uM)'], y_ + 0.2, f"{r['peak (uM)']:.3f} µM ({r['peak / true peak']:.0%})",
           ha='center', va='bottom', fontsize=8.5, color='#52514e')
a.axvline(true_pk_amp, color=c_true, lw=2, label=f'True mean peak ({true_pk_amp:.3f} µM)')
a.set_yticks(ypos, [f'{row}{est_order.index(est) + 1}. {est}' for row, est in order], fontsize=9)
a.set_ylim(ypos.min() - 0.6, ypos.max() + 0.8)
a.set(xlabel='Peak amplitude (µM), point = mean HRF peak, bar = 95% interval   (● design A, ■ design B)',
      title='Peak amplitude by estimator')
a.legend(frameon=False, fontsize=9, loc='upper left')
a.grid(axis='x', alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    a.spines[sp].set_visible(False)
fig.suptitle(f"{subject} {select_parcel} HbO, {'-'.join(run_keys)}: HRF amplitude by estimator, "
             f"design not highpassed (A) vs highpassed like y (B)")
if is_save:
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f'{subject}_{select_parcel}_cmp_HRF_truth_shrinkage.png')
    fig.savefig(path, dpi=150)
    diag.to_csv(os.path.join(plot_dir, f'{subject}_{select_parcel}_cmp_HRF_truth_shrinkage.csv'))
    print(f"saved {path}")
plt.show()

# %%
