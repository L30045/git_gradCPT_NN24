#%% Simulate mnt-correct HRFs on real fNIRS to validate the Bayesian HRF model
# Ground-truth data for run_Bayes_image-space_gaussian-kernel_mnt-correct.py.
#
# - Baseline: sub-723 image-space HbO of one parcel (DorsAttnB_PostC_2_RH), all runs. Only the
#   city-correct trials (trial_type 'city', response_code 1) are used as events.
# - In each run, a random 10% of the city-correct trials are relabeled as simulated mnt-correct
#   ('mnt', response_code 0). The real mnt trials are renamed 'mnt-orig' so a script that selects
#   mnt-correct by (trial_type == 'mnt') & (response_code == 0) sees only the simulated ones.
# - On each simulated trial, a double-gamma HRF is added to the raw parcel series:
#       h_i(t) = A_i * g(s_i * (t - onset_i - j_i / fs)),   g = double gamma with peak 1
#   g(t) = Gamma(t; a1) - ratio * Gamma(t; a2) scaled to a peak of 1 (sim_utils.double_gamma_hrf).
#   hrf_shape = 'fit_real': a1, a2, ratio are fitted to the real mnt-correct response (see Amplitude below);
#   hrf_shape = 'canonical': SPM's a1 = 6, a2 = 16, ratio = 1/6.
#   hrf_t_end (s): g is multiplied by a cosine taper (1 until hrf_t_end - hrf_taper, 0 from hrf_t_end on), so the
#   HRF is back at exactly 0 at hrf_t_end (at hrf_t_end / s_i for a time-scaled trial). The shape fit includes the
#   taper. None = untapered double gamma (the fitted undershoot then lasts until ~28 s).
#   A_i: peak amplitude (uM); s_i: time-scale factor, which sets the slope (rise slope ∝ A_i * s_i,
#   peak latency = t_peak / s_i); j_i: onset jitter in samples (uniform integer in [-J, J]).
#   The HRF is placed on the sample grid at (nearest sample to the labeled onset) + j_i.
# - Amplitude (amp_mean = 'match_real'): matched to the real mnt-correct response of this parcel. A (and, with
#   hrf_shape = 'fit_real', the shape a1, a2, ratio) is the least-squares fit of A * g (s = 1) to the real
#   (mnt-correct − city-correct) epoch mean, 0-18 s. The template is a unit HRF placed at every real mnt-correct
#   onset and passed through the same highpass, epoching and baseline correction as the data, so A is on the same
#   scale as the injected HRFs. The canonical shape alone could not fit the real response (its undershoot is
#   too small), which held A at about half the real peak.
# - Background shrink (bg_scale = c < 1): before the HRFs are added, the real signal under each simulated trial is
#   scaled down, so the injected HRF is less buried in the city-correct background. Only the fast part of the
#   series is scaled (above the l_cutoff highpass); the slow part (drift, offset) is kept, so no steps appear:
#       x_new(t) = x_slow(t) + m(t) * x_fast(t),   m(t) = 1 - (1 - c) * w(t)
#   w(t) is 1 within bg_win of any simulated onset, with cosine ramps of bg_taper s at the edges (max over trials,
#   so overlapping windows merge). c = 1 leaves the data unchanged.
# - Per-trial SNR (saved in the ground truth): snr = A_i / SD of the background in the window 0-18 s after the
#   labeled onset, where the background is the (shrunk) signal without any injected HRF, highpassed at l_cutoff
#   like the analysis; snr_db = 20 log10(snr).
# - Saved in the same format as the real inputs (stims list + {'parcel_ts': [...]}, molar), so the
#   Bayesian script can read the simulated files in place of the real ones. Per-trial ground truth
#   (A, s, jitter, peak latency, max slope) is saved alongside.
#%% load library
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import lognorm, t as t_dist
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from params_setting import *

#%% settings
subject = 'sub-723'
select_parcel = 'DorsAttnB_PostC_2_RH'
select_chromo = 'HbO'
mnt_frac = 0.10        # fraction of city-correct trials relabeled as mnt-correct (per run)
seed = 0
# HRF parameters
amp_mean = 'match_real'  # mean peak amplitude (uM), or 'match_real' to fit it to the real mnt-correct response
                         # (0.2 was the earlier hand-set value, ~5x the real response in sub-723)
amp_fit_win = (0.0, 18.0)  # lag window (s) of the amplitude (and shape) fit
hrf_shape = 'fit_real'   # 'fit_real': fit a1, a2, ratio of the double gamma to the real response; 'canonical': SPM
shape_bounds = {'a1': (3.0, 12.0), 'a2': (8.0, 30.0), 'ratio': (0.0, 1.5)}   # search range of the shape fit
hrf_t_end = 15.0       # HRF forced back to 0 at this lag (s); None = no taper
hrf_taper = 5.0        # length (s) of the cosine taper ending at hrf_t_end
amp_cv = 0.25          # amplitude sd / mean (truncated at > 0)
scale_log_sd = 0.15    # s ~ LogNormal(0, scale_log_sd); s > 1 = steeper, earlier peak
jitter_max = 3         # onset jitter, integer samples in [-jitter_max, jitter_max] (~±0.33 s at 9 Hz)
hrf_dur = 30.0         # HRF support (s), long enough for the undershoot to return to 0
# background shrink under the simulated trials (realism / SNR)
bg_scale = 1.0         # c: factor on the fast part of the real signal near each simulated trial (1 = no change)
bg_win = (-2.0, 18.0)  # window around each simulated onset (s) where the background is fully scaled
bg_taper = 2.0         # cosine ramp (s) on each side of bg_win
snr_win = (0.0, 18.0)  # window after the labeled onset (s) for the per-trial noise SD
# visualization
l_cutoff = 0.02        # highpass (Hz) for the epoch plots, same as the Bayesian script
epoch_win = (-2.0, 18.0)
n_examples = 4
is_save = True

rng = np.random.default_rng(seed)

# colors (validated categorical slots 1-2)
c_city, c_mnt, c_true = '#2a78d6', '#eb6834', '#1f1f1f'


#%% double-gamma HRF (shared with cmp_HRF_simulation.py)
from sim_utils import double_gamma_hrf, hrf_props, true_trial_hrfs, CANONICAL
from scipy.optimize import minimize


#%% load real data
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
stim_file = os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl')
ts_file = os.path.join(der_dir, subject,
                       f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl')
with gzip.open(stim_file, 'rb') as f:
    preproc = pickle.load(f)
all_stims = preproc['stims']
with open(ts_file, 'rb') as f:
    image_results = pickle.load(f)
raw_runs = image_results['parcel_ts']
if len(raw_runs) != len(all_stims):
    raise ValueError(f"{len(raw_runs)} parcel runs but {len(all_stims)} stim tables.")


#%% amplitude (and shape) matched to the real mnt-correct response
def real_mnt_epochs(stims, runs):
    """Real (mnt-correct − city-correct) epoch mean of the selected parcel (highpassed, baseline-corrected, trials
    pooled over runs), plus what is needed to build matching templates: per run (t, fs, sos, mnt onset samples)."""
    sums, n, run_info = {'mnt': 0.0, 'city': 0.0}, {'mnt': 0, 'city': 0}, []
    for stim, run in zip(stims, runs):
        t = run.time.values
        fs = 1 / np.median(np.diff(t))
        sos = butter(4, l_cutoff, btype='highpass', fs=fs, output='sos')
        x = sosfiltfilt(sos, run.sel(parcel=select_parcel, chromo=select_chromo)
                        .pint.to('micromolar').pint.dequantify().values)
        lags = np.arange(int(round(epoch_win[0] * fs)), int(round(epoch_win[1] * fs)) + 1)
        on = {'mnt': stim.loc[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0), 'onset'].values,
              'city': stim.loc[(stim['trial_type'] == 'city') & (stim['response_code'] == 1), 'onset'].values}
        i0 = {k: np.array([np.argmin(np.abs(t - o)) for o in v], dtype=int) for k, v in on.items()}
        for k in ['mnt', 'city']:
            idx = i0[k][:, None] + lags[None, :]
            idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(t))]
            ep = x[idx]
            sums[k] = sums[k] + (ep - ep[:, lags < 0].mean(1, keepdims=True)).sum(0)
            n[k] += len(idx)
        run_info.append((t, fs, sos, i0['mnt'], lags))
    return lags / fs, sums['mnt'] / n['mnt'] - sums['city'] / n['city'], run_info, n['mnt']


def template(shape, run_info):
    """Epoch mean of a unit-peak HRF series (one HRF per real mnt-correct onset, overlaps included) with the same
    highpass, epoching and baseline correction as the data."""
    total, n = 0.0, 0
    for t, fs, sos, i_mnt, lags in run_info:
        unit = np.zeros(len(t))
        n_hrf = int(np.ceil(hrf_dur * fs))
        g = double_gamma_hrf(np.arange(n_hrf) / fs, **shape, t_end=hrf_t_end, taper=hrf_taper)
        for i in i_mnt:
            m = min(n_hrf, len(t) - i)
            unit[i:i + m] += g[:m]
        unit = sosfiltfilt(sos, unit)
        idx = i_mnt[:, None] + lags[None, :]
        idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(t))]
        ep = unit[idx]
        total = total + (ep - ep[:, lags < 0].mean(1, keepdims=True)).sum(0)
        n += len(idx)
    return total / n


def fit_real_mnt(stims, runs, fit_shape, win=amp_fit_win):
    """Least-squares A (and shape, if fit_shape) of A * g fit to the real (mnt − city) epoch mean over win."""
    lag_t, d, run_info, n_real = real_mnt_epochs(stims, runs)
    m = (lag_t >= win[0]) & (lag_t <= win[1])

    def amp_sse(shape):
        h = template(shape, run_info)[m]
        A = float(d[m] @ h / (h @ h))
        return A, float(((d[m] - A * h) ** 2).sum())

    shape = dict(CANONICAL)
    if fit_shape:
        keys = list(shape_bounds)
        lo, hi = np.array([shape_bounds[k][0] for k in keys]), np.array([shape_bounds[k][1] for k in keys])
        def cost(p):
            p = np.clip(p, lo, hi)
            sh = dict(zip(keys, p))
            if sh['a2'] <= sh['a1'] + 2:          # undershoot must come after the peak
                return 1e9
            A, sse = amp_sse(sh)
            return sse if A > 0 else 1e9
        best = None
        for start in [[6, 16, 1 / 6], [6, 14, 0.5], [7, 18, 0.8], [5, 12, 0.35]]:
            r = minimize(cost, start, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-12, 'maxiter': 2000})
            best = r if best is None or r.fun < best.fun else best
        shape = dict(zip(keys, np.clip(best.x, lo, hi).astype(float)))
    A, sse = amp_sse(shape)
    fit = A * template(shape, run_info)
    r2 = 1 - sse / float(((d[m] - d[m].mean()) ** 2).sum())
    return A, shape, n_real, {'lag_t': lag_t, 'real': d, 'fit': fit, 'r2': r2}


if amp_mean == 'match_real':
    amp_mean, hrf_params, n_real, amp_fit = fit_real_mnt(all_stims, raw_runs, hrf_shape == 'fit_real')
    if amp_mean <= 0:
        raise ValueError(f"Fitted real mnt-correct amplitude is {amp_mean:.4f} uM (<= 0); set amp_mean by hand.")
    print(f"matched to the real mnt-correct response ({n_real} trials, {hrf_shape} shape): A = {amp_mean:.4f} uM, "
          f"a1 = {hrf_params['a1']:.2f}, a2 = {hrf_params['a2']:.2f}, ratio = {hrf_params['ratio']:.3f}, "
          f"R2 of the fit (0-18 s) = {amp_fit['r2']:.2f}")
else:
    hrf_params, amp_fit = dict(CANONICAL), None
t_peak, max_slope, undershoot = hrf_props(**hrf_params, t_end=hrf_t_end, taper=hrf_taper)
print(f"HRF shape: peak at {t_peak:.1f} s, undershoot {undershoot:.2f} x peak"
      + (f", back to 0 at {hrf_t_end:g} s" if hrf_t_end is not None else ''))
shape_tag = ('canonical' if hrf_shape == 'canonical' else 'dgfit') + (f'-end{hrf_t_end:g}' if hrf_t_end is not None else '')

sim_name = (f'sim_mnt-correct_{int(mnt_frac*100)}pct-city_{shape_tag}_A{amp_mean:.3g}_cv{amp_cv}'
            f'_slogsd{scale_log_sd}_jit{jitter_max}' + (f'_bg{bg_scale:g}' if bg_scale != 1 else '') + f'_seed{seed}')
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject, 'simulation', sim_name)
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, sim_name)


#%% simulate
def shrink_weight(t, onsets, win=bg_win, taper=bg_taper):
    """w(t) in [0, 1]: 1 within win of any onset, cosine ramps of `taper` s outside it, 0 elsewhere."""
    w = np.zeros(len(t))
    for o in onsets:
        d = t - o
        w_o = np.zeros(len(t))
        w_o[(d >= win[0]) & (d <= win[1])] = 1.0
        if taper > 0:
            pre = (d >= win[0] - taper) & (d < win[0])
            post = (d > win[1]) & (d <= win[1] + taper)
            w_o[pre] = 0.5 * (1 - np.cos(np.pi * (d[pre] - (win[0] - taper)) / taper))
            w_o[post] = 0.5 * (1 + np.cos(np.pi * (d[post] - win[1]) / taper))
        w = np.maximum(w, w_o)
    return w


sim_runs, sim_stims, truth_rows, bg_mult = [], [], [], []
for r_i, (stim, run) in enumerate(zip(all_stims, raw_runs)):
    run_units = run.pint.units
    t = run.time.values
    fs = 1 / np.median(np.diff(t))
    n_t = len(t)

    stim = stim.copy()
    stim['is_sim'] = False
    stim.loc[stim['trial_type'] == 'mnt', 'trial_type'] = 'mnt-orig'
    city_idx = stim.index[(stim['trial_type'] == 'city') & (stim['response_code'] == 1)]
    n_sim = int(round(mnt_frac * len(city_idx)))
    sim_idx = np.sort(rng.choice(city_idx, size=n_sim, replace=False))

    amp = np.abs(rng.normal(amp_mean, amp_cv * amp_mean, n_sim))
    s = np.exp(rng.normal(0.0, scale_log_sd, n_sim))
    jit = rng.integers(-jitter_max, jitter_max + 1, n_sim)

    # shrink the fast part of the real signal under the simulated trials (selected parcel and chromophore only)
    sos = butter(4, l_cutoff, btype='highpass', fs=fs, output='sos')
    x_uM = run.sel(parcel=select_parcel, chromo=select_chromo).pint.to('micromolar').pint.dequantify().values
    x_fast = sosfiltfilt(sos, x_uM)
    mult = 1.0 - (1.0 - bg_scale) * shrink_weight(t, stim.loc[sim_idx, 'onset'].values)
    bg_delta = (mult - 1.0) * x_fast                    # change of the background (uM)
    bg_hp = sosfiltfilt(sos, x_uM + bg_delta)           # background under the HRFs, highpassed like the analysis
    bg_mult.append(mult)

    injected = np.zeros(n_t)
    n_hrf = int(np.ceil(hrf_dur * fs))
    for k, idx in enumerate(sim_idx):
        onset = stim.at[idx, 'onset']
        i0 = int(np.argmin(np.abs(t - onset))) + jit[k]
        seg = np.arange(max(i0, 0), min(i0 + n_hrf, n_t))
        injected[seg] += amp[k] * double_gamma_hrf(s[k] * (seg - i0) / fs, **hrf_params, t_end=hrf_t_end, taper=hrf_taper)
        w_snr = (t >= onset + snr_win[0]) & (t <= onset + snr_win[1])
        noise_sd = float(bg_hp[w_snr].std(ddof=1))
        truth_rows.append(dict(run_idx=r_i, stim_idx=idx, onset=onset, onset_true=t[i0],
                               amp_uM=amp[k], time_scale=s[k], jitter_samples=jit[k],
                               peak_latency_s=t_peak / s[k], max_slope_uM_per_s=amp[k] * s[k] * max_slope,
                               hrf_a1=hrf_params['a1'], hrf_a2=hrf_params['a2'], hrf_ratio=hrf_params['ratio'],
                               hrf_t_end=np.nan if hrf_t_end is None else hrf_t_end, hrf_taper=hrf_taper,
                               bg_scale_at_onset=float(np.interp(onset, t, mult)),
                               noise_sd_uM=noise_sd, snr=amp[k] / noise_sd,
                               snr_db=20 * np.log10(amp[k] / noise_sd)))

    stim.loc[sim_idx, ['trial_type', 'response_code', 'is_sim']] = ['mnt', 0, True]

    # add the injected signal (uM -> run units) to the selected parcel and chromophore only
    vals = run.pint.dequantify().copy()
    inj_native = ((bg_delta + injected) * units.micromolar).to(run_units).magnitude
    vals.loc[dict(parcel=select_parcel, chromo=select_chromo)] += inj_native
    sim_run = vals.pint.quantify(run_units)
    sim_run.attrs['injected_uM'] = injected  # kept for plotting/checking

    sim_runs.append(sim_run)
    sim_stims.append(stim)
    print(f"run {r_i}: {len(city_idx)} city-correct, {n_sim} relabeled as simulated mnt-correct, "
          f"fs = {fs:.2f} Hz, background scaled by {bg_scale:g} on {np.mean(mult < 1) * 100:.0f}% of the run")

truth = pd.DataFrame(truth_rows)
print(truth[['amp_uM', 'time_scale', 'jitter_samples', 'peak_latency_s', 'max_slope_uM_per_s',
             'noise_sd_uM', 'snr', 'snr_db']].describe().round(4))


#%% save (same layout as the real inputs)
if is_save:
    os.makedirs(data_save_path, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    runs_to_save = []
    for r in sim_runs:
        r = r.copy()
        r.attrs.pop('injected_uM', None)
        runs_to_save.append(r)
    with gzip.open(os.path.join(data_save_path, f'{subject}_sim_stims.pkl'), 'wb') as f:
        pickle.dump({'stims': sim_stims}, f)
    with open(os.path.join(data_save_path, f'{subject}_sim_parcel_ts.pkl'), 'wb') as f:
        pickle.dump({'parcel_ts': runs_to_save}, f)
    with open(os.path.join(data_save_path, f'{subject}_sim_injected_uM.pkl'), 'wb') as f:
        pickle.dump([r.attrs['injected_uM'] for r in sim_runs], f)
    with open(os.path.join(data_save_path, f'{subject}_sim_bg_multiplier.pkl'), 'wb') as f:
        pickle.dump(bg_mult, f)     # m(t) per run: factor on the fast part of the background
    truth.to_csv(os.path.join(data_save_path, f'{subject}_sim_ground_truth.csv'), index=False)
    print(f"saved to {data_save_path}")


#%% epochs (highpassed, baseline-corrected) for visualization
def get_parcel_uM(run):
    return run.sel(parcel=select_parcel, chromo=select_chromo).pint.to('micromolar').pint.dequantify().values

def epoch(x, t, onsets, fs, win):
    lags = np.arange(int(round(win[0] * fs)), int(round(win[1] * fs)) + 1)
    idx = np.array([np.argmin(np.abs(t - o)) for o in onsets])[:, None] + lags[None, :]
    keep = (idx[:, 0] >= 0) & (idx[:, -1] < len(x))
    ep = x[idx[keep]]
    ep = ep - ep[:, lags < 0].mean(1, keepdims=True)
    return lags / fs, ep

ep = {'city-correct': [], 'mnt-correct': [], 'mnt-correct (injected only)': []}
hp_runs = []
for r_i, (stim, run) in enumerate(zip(sim_stims, sim_runs)):
    t = run.time.values
    fs = 1 / np.median(np.diff(t))
    sos = butter(4, l_cutoff, btype='highpass', fs=fs, output='sos')
    x_hp = sosfiltfilt(sos, get_parcel_uM(run))
    x0_hp = sosfiltfilt(sos, get_parcel_uM(raw_runs[r_i]))
    hp_runs.append((t, x_hp, x0_hp, run.attrs['injected_uM']))
    city_on = stim.loc[(stim['trial_type'] == 'city') & (stim['response_code'] == 1), 'onset'].values
    mnt_on = stim.loc[stim['is_sim'], 'onset'].values
    lag_t, e_c = epoch(x_hp, t, city_on, fs, epoch_win)
    _, e_m = epoch(x_hp, t, mnt_on, fs, epoch_win)
    _, e_inj = epoch(run.attrs['injected_uM'], t, mnt_on, fs, epoch_win)
    ep['city-correct'].append(e_c)
    ep['mnt-correct'].append(e_m)
    ep['mnt-correct (injected only)'].append(e_inj)
ep = {k: np.concatenate(v) for k, v in ep.items()}

# per-trial true HRFs on the epoch lag axis (aligned to the labeled onset, so jitter is included)
true_hrfs = true_trial_hrfs(truth, lag_t)


def mean_ci(x, level=0.95):
    m = x.mean(0)
    sem = x.std(0, ddof=1) / np.sqrt(len(x))
    h = t_dist.ppf(0.5 + level / 2, len(x) - 1) * sem
    return m, m - h, m + h


def style(ax):
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


#%% figure 0: fit of the injected HRF to the real mnt-correct response
if amp_fit is not None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(amp_fit['lag_t'], amp_fit['real'], color='#7b3fb5', lw=1.5, label='real mnt-correct − city-correct (epoch mean)')
    ax.plot(amp_fit['lag_t'], amp_fit['fit'], color=c_mnt, lw=2.2,
            label=f'fit: A·g after the same processing (A = {amp_mean:.4f} µM, R² = {amp_fit["r2"]:.2f})')
    t_g = np.linspace(0, epoch_win[1], 500)
    ax.plot(t_g, amp_mean * double_gamma_hrf(t_g, **hrf_params, t_end=hrf_t_end, taper=hrf_taper), color=c_true, ls='--', lw=1.5,
            label=f'injected HRF A·g (a1 = {hrf_params["a1"]:.1f}, a2 = {hrf_params["a2"]:.1f}, ratio = {hrf_params["ratio"]:.2f})')
    ax.axvspan(*amp_fit_win, color='#eeeeee', zorder=-1, lw=0)
    ax.set(xlabel='Lag from mnt-correct onset (s)', ylabel=f'{select_chromo} (µM)',
           title=f'{subject} {select_parcel}: HRF matched to the real response ({hrf_shape})')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    if is_save:
        fig.savefig(os.path.join(plot_dir, f'{subject}_{select_parcel}_HRF_fit_to_real.png'), dpi=150)


#%% figure 1: a stretch of the simulated series + single-trial examples
r_show = 0
t, x_hp, x0_hp, inj = hp_runs[r_show]
stim = sim_stims[r_show]
sim_on = stim.loc[stim['is_sim'], 'onset'].values
seg_start = sim_on[min(2, len(sim_on) - 1)] - 10
seg = (t >= seg_start) & (t <= seg_start + 90)

fig = plt.figure(figsize=(13, 9), constrained_layout=True)
gs = fig.add_gridspec(3, n_examples, height_ratios=[1.2, 1, 1])
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(t[seg], x0_hp[seg], color='#aaaaaa', lw=1, label='Original (highpassed)')
ax0.plot(t[seg], x_hp[seg], color=c_city, lw=1.2, label='Simulated (original + injected)')
ax0.plot(t[seg], inj[seg], color=c_mnt, lw=2, label='Injected HRFs')
for i, o in enumerate(sim_on[(sim_on >= t[seg][0]) & (sim_on <= t[seg][-1])]):
    ax0.axvline(o, color=c_mnt, ls='--', lw=0.8, label='Simulated mnt-correct onset' if i == 0 else None)
ax0.set(xlabel='Time in run (s)', ylabel=f'{select_chromo} (µM)',
        title=f'Run {r_show + 1}, 90 s of the simulated series')
ax0.legend(frameon=False, fontsize=9, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.08))
ax0.set_xlim(t[seg][0], t[seg][-1])
ax0.axhline(0, color='#cccccc', lw=0.8, zorder=0)
ax0.grid(alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    ax0.spines[sp].set_visible(False)

ex_c = rng.choice(len(ep['city-correct']), n_examples, replace=False)
ex_m = rng.choice(len(ep['mnt-correct']), n_examples, replace=False)
for k in range(n_examples):
    a = fig.add_subplot(gs[1, k])
    a.plot(lag_t, ep['city-correct'][ex_c[k]], color=c_city, lw=1.5)
    a.set(title=f'city-correct #{ex_c[k] + 1}', xlabel='Lag (s)', ylabel='µM' if k == 0 else None)
    style(a)
    b = fig.add_subplot(gs[2, k], sharey=a)
    b.plot(lag_t, ep['mnt-correct'][ex_m[k]], color=c_mnt, lw=1.5, label='Simulated data')
    b.plot(lag_t, true_hrfs[ex_m[k]], color=c_true, ls='--', lw=1.5, label='True HRF')
    row = truth.iloc[ex_m[k]]
    b.set(title=f'mnt-correct #{ex_m[k] + 1}\nA={row.amp_uM:.2f} µM, s={row.time_scale:.2f}, '
                f'jit={int(row.jitter_samples):+d}', xlabel='Lag (s)', ylabel='µM' if k == 0 else None)
    style(b)
    if k == 0:
        b.legend(frameon=False, fontsize=8)
fig.suptitle(f'{subject} {select_parcel} {select_chromo}: simulated trials '
             f'(highpassed {l_cutoff} Hz, baseline −2–0 s)')
if is_save:
    fig.savefig(os.path.join(plot_dir, f'{subject}_{select_parcel}_sim_examples.png'), dpi=150)


#%% figure 2: mean and 95% CI per trial type, plus the ground truth
fig, axs = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True,
                        gridspec_kw={'width_ratios': [2, 1, 1, 1]})
a = axs[0]
for name, c in [('city-correct', c_city), ('mnt-correct', c_mnt)]:
    m, lo, hi = mean_ci(ep[name])
    a.fill_between(lag_t, lo, hi, color=c, alpha=0.2, lw=0)
    a.plot(lag_t, m, color=c, lw=2, label=f'{name} (n = {len(ep[name])})')
m_true, lo_true, hi_true = mean_ci(true_hrfs)
a.plot(lag_t, m_true, color=c_true, ls='--', lw=1.5, label='True mean HRF (injected)')
a.set(xlabel='Lag (s)', ylabel=f'{select_chromo} (µM)', title='Epoch mean, 95% CI (t-based, across trials)')
a.legend(frameon=False, fontsize=9)
style(a)

axs[1].hist(truth.amp_uM, bins=20, color=c_mnt, alpha=0.8, edgecolor='white')
axs[1].set(xlabel='Peak amplitude A (µM)', ylabel='Trials', title='True amplitude')
axs[2].scatter(truth.peak_latency_s, truth.max_slope_uM_per_s, s=18, color=c_mnt, alpha=0.8,
               edgecolor='white', lw=0.5)
axs[2].set(xlabel='Peak latency (s)', ylabel='Max rise slope (µM/s)', title='True latency vs slope')
axs[3].hist(truth.snr_db, bins=20, color=c_mnt, alpha=0.8, edgecolor='white')
axs[3].axvline(truth.snr_db.median(), color=c_true, ls='--', lw=1.2, label=f'median {truth.snr_db.median():.1f} dB')
axs[3].set(xlabel='SNR = A / noise SD (dB)', ylabel='Trials',
           title=f'Per-trial SNR (background ×{bg_scale:g}, {snr_win[0]:g}–{snr_win[1]:g} s)')
axs[3].legend(frameon=False, fontsize=8)
for ax in axs[1:]:
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.suptitle(f'{subject} {select_parcel} {select_chromo}: city-correct vs simulated mnt-correct')
if is_save:
    fig.savefig(os.path.join(plot_dir, f'{subject}_{select_parcel}_sim_mean_CI.png'), dpi=150)
    print(f"figures saved to {plot_dir}")


#%% figure 3: the HRF at quantiles of the time-scale factor s
q_s = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
s_q = lognorm.ppf(q_s, scale_log_sd)
c_seq = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # one-hue ramp, light -> dark = small -> large s
t_h = np.linspace(-2, 25, 2701)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, gridspec_kw={'width_ratios': [2, 1]})
a = axs[0]
for s_, c, qq in zip(s_q, c_seq, q_s):
    h = double_gamma_hrf(s_ * t_h, **hrf_params, t_end=hrf_t_end, taper=hrf_taper)
    dh = np.gradient(h, t_h)
    a.plot(t_h, h, color=c, lw=2.6 if qq == 0.5 else 1.8,
           label=f's = {s_:.2f} ({qq*100:.1f}th pct): peak {t_h[h.argmax()]:.1f} s, max slope {dh.max():.2f}/s')
    a.plot(t_h[h.argmax()], h.max(), 'o', ms=7, color=c, mec='white', mew=1.5)
    i = dh.argmax()
    a.plot(t_h[i], h[i], 's', ms=6, color=c, mec='white', mew=1.2)
a.set(xlabel='Time from onset (s)', ylabel='g(s·t) (peak = 1, A = 1)',
      title=f'{shape_tag} double-gamma HRF at quantiles of s ~ LogNormal(0, {scale_log_sd})\n● peak   ■ steepest rise')
a.legend(frameon=False, fontsize=9, loc='upper right')
style(a)

b = axs[1]
s_grid = np.linspace(lognorm.ppf(0.0005, scale_log_sd), lognorm.ppf(0.9995, scale_log_sd), 500)
b.plot(s_grid, lognorm.pdf(s_grid, scale_log_sd), color=c_seq[2], lw=2)
for s_, c in zip(s_q, c_seq):
    b.axvline(s_, color=c, ls='--', lw=1.2)
b.set(xlabel='s', ylabel='Density', title='Distribution of s')
b.grid(alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    b.spines[sp].set_visible(False)
if is_save:
    fig.savefig(os.path.join(plot_dir, f'{shape_tag}_HRF_by_s_slogsd{scale_log_sd}.png'), dpi=150)
plt.show()
