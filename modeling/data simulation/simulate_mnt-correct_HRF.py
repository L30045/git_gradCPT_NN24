#%% Simulate mnt-correct HRFs on real fNIRS to validate the Bayesian HRF model
# Ground-truth data for run_Bayes_image-space_gaussian-kernel_mnt-correct.py.
#
# - Baseline: sub-723 image-space HbO of one parcel (DorsAttnB_PostC_2_RH), all runs. Only the
#   city-correct trials (trial_type 'city', response_code 1) are used as events.
# - In each run, a random 10% of the city-correct trials are relabeled as simulated mnt-correct
#   ('mnt', response_code 0). The real mnt trials are renamed 'mnt-orig' so a script that selects
#   mnt-correct by (trial_type == 'mnt') & (response_code == 0) sees only the simulated ones.
# - On each simulated trial, a canonical (SPM double-gamma) HRF is added to the raw parcel series:
#       h_i(t) = A_i * g(s_i * (t - onset_i - j_i / fs)),   g = double gamma with peak 1
#   A_i: peak amplitude (uM); s_i: time-scale factor, which sets the slope (rise slope ∝ A_i * s_i,
#   peak latency = t_peak / s_i); j_i: onset jitter in samples (uniform integer in [-J, J]).
#   The HRF is placed on the sample grid at (nearest sample to the labeled onset) + j_i.
# - Amplitude (amp_mean = 'match_real'): matched to the real mnt-correct response of this parcel. A is the
#   least-squares scale of a canonical HRF (s = 1) fit to the real (mnt-correct − city-correct) epoch mean,
#   0-18 s. The template is a unit HRF placed at every real mnt-correct onset and passed through the same
#   highpass, epoching and baseline correction as the data, so A is on the same scale as the injected HRFs.
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
amp_fit_win = (0.0, 18.0)  # lag window (s) of the amplitude fit
amp_cv = 0.25          # amplitude sd / mean (truncated at > 0)
scale_log_sd = 0.15    # s ~ LogNormal(0, scale_log_sd); s > 1 = steeper, earlier peak
jitter_max = 3         # onset jitter, integer samples in [-jitter_max, jitter_max] (~±0.33 s at 9 Hz)
hrf_dur = 30.0         # HRF support (s), long enough for the undershoot to return to 0
# visualization
l_cutoff = 0.02        # highpass (Hz) for the epoch plots, same as the Bayesian script
epoch_win = (-2.0, 18.0)
n_examples = 4
is_save = True

rng = np.random.default_rng(seed)

# colors (validated categorical slots 1-2)
c_city, c_mnt, c_true = '#2a78d6', '#eb6834', '#1f1f1f'


#%% canonical HRF (shared with cmp_HRF_simulation.py)
from sim_utils import canonical_hrf, true_trial_hrfs, T_PEAK, MAX_SLOPE


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


#%% amplitude matched to the real mnt-correct response
def fit_real_mnt_amplitude(stims, runs, win=amp_fit_win):
    """Least-squares A of a canonical HRF (s = 1) fit to the real (mnt-correct − city-correct) epoch mean of the
    selected parcel (highpassed, baseline-corrected, trials pooled over runs). The template is the epoch mean of a
    unit HRF series (one HRF per real mnt-correct onset, overlaps included) given the same processing."""
    sums = {'mnt': 0.0, 'city': 0.0, 'tmpl': 0.0}
    n = {'mnt': 0, 'city': 0}
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
        unit = np.zeros(len(t))
        n_hrf = int(np.ceil(hrf_dur * fs))
        for i in i0['mnt']:
            seg = np.arange(i, min(i + n_hrf, len(t)))
            unit[seg] += canonical_hrf((seg - i) / fs)
        unit = sosfiltfilt(sos, unit)
        for k, sig in [('mnt', x), ('city', x), ('tmpl', unit)]:
            idx = i0['city' if k == 'city' else 'mnt'][:, None] + lags[None, :]
            idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(t))]
            ep = sig[idx]
            sums[k] = sums[k] + (ep - ep[:, lags < 0].mean(1, keepdims=True)).sum(0)
            if k != 'tmpl':
                n[k] += len(idx)
    lag_t = lags / fs
    d = sums['mnt'] / n['mnt'] - sums['city'] / n['city']
    h = sums['tmpl'] / n['mnt']
    m = (lag_t >= win[0]) & (lag_t <= win[1])
    return float(d[m] @ h[m] / (h[m] @ h[m])), n['mnt']


if amp_mean == 'match_real':
    amp_mean, n_real = fit_real_mnt_amplitude(all_stims, raw_runs)
    if amp_mean <= 0:
        raise ValueError(f"Fitted real mnt-correct amplitude is {amp_mean:.4f} uM (<= 0); set amp_mean by hand.")
    print(f"amp_mean matched to the real mnt-correct response ({n_real} trials): {amp_mean:.4f} uM")

sim_name = (f'sim_mnt-correct_{int(mnt_frac*100)}pct-city_canonical_A{amp_mean:.3g}_cv{amp_cv}'
            f'_slogsd{scale_log_sd}_jit{jitter_max}_seed{seed}')
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject, 'simulation', sim_name)
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, sim_name)


#%% simulate
sim_runs, sim_stims, truth_rows = [], [], []
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

    injected = np.zeros(n_t)
    n_hrf = int(np.ceil(hrf_dur * fs))
    for k, idx in enumerate(sim_idx):
        onset = stim.at[idx, 'onset']
        i0 = int(np.argmin(np.abs(t - onset))) + jit[k]
        seg = np.arange(max(i0, 0), min(i0 + n_hrf, n_t))
        injected[seg] += amp[k] * canonical_hrf(s[k] * (seg - i0) / fs)
        truth_rows.append(dict(run_idx=r_i, stim_idx=idx, onset=onset, onset_true=t[i0],
                               amp_uM=amp[k], time_scale=s[k], jitter_samples=jit[k],
                               peak_latency_s=T_PEAK / s[k], max_slope_uM_per_s=amp[k] * s[k] * MAX_SLOPE))

    stim.loc[sim_idx, ['trial_type', 'response_code', 'is_sim']] = ['mnt', 0, True]

    # add the injected signal (uM -> run units) to the selected parcel and chromophore only
    vals = run.pint.dequantify().copy()
    inj_native = (injected * units.micromolar).to(run_units).magnitude
    vals.loc[dict(parcel=select_parcel, chromo=select_chromo)] += inj_native
    sim_run = vals.pint.quantify(run_units)
    sim_run.attrs['injected_uM'] = injected  # kept for plotting/checking

    sim_runs.append(sim_run)
    sim_stims.append(stim)
    print(f"run {r_i}: {len(city_idx)} city-correct, {n_sim} relabeled as simulated mnt-correct, "
          f"fs = {fs:.2f} Hz")

truth = pd.DataFrame(truth_rows)
print(truth[['amp_uM', 'time_scale', 'jitter_samples', 'peak_latency_s', 'max_slope_uM_per_s']].describe().round(3))


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
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True,
                        gridspec_kw={'width_ratios': [2, 1, 1]})
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
for ax in axs[1:]:
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.suptitle(f'{subject} {select_parcel} {select_chromo}: city-correct vs simulated mnt-correct')
if is_save:
    fig.savefig(os.path.join(plot_dir, f'{subject}_{select_parcel}_sim_mean_CI.png'), dpi=150)
    print(f"figures saved to {plot_dir}")


#%% figure 3: canonical HRF at quantiles of the time-scale factor s
q_s = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
s_q = lognorm.ppf(q_s, scale_log_sd)
c_seq = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # one-hue ramp, light -> dark = small -> large s
t_h = np.linspace(-2, 25, 2701)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, gridspec_kw={'width_ratios': [2, 1]})
a = axs[0]
for s_, c, qq in zip(s_q, c_seq, q_s):
    h = canonical_hrf(s_ * t_h)
    dh = np.gradient(h, t_h)
    a.plot(t_h, h, color=c, lw=2.6 if qq == 0.5 else 1.8,
           label=f's = {s_:.2f} ({qq*100:.1f}th pct): peak {t_h[h.argmax()]:.1f} s, max slope {dh.max():.2f}/s')
    a.plot(t_h[h.argmax()], h.max(), 'o', ms=7, color=c, mec='white', mew=1.5)
    i = dh.argmax()
    a.plot(t_h[i], h[i], 's', ms=6, color=c, mec='white', mew=1.2)
a.set(xlabel='Time from onset (s)', ylabel='g(s·t) (peak = 1, A = 1)',
      title=f'Canonical HRF at quantiles of s ~ LogNormal(0, {scale_log_sd})\n● peak   ■ steepest rise')
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
    fig.savefig(os.path.join(plot_dir, f'canonical_HRF_by_s_slogsd{scale_log_sd}.png'), dpi=150)
plt.show()
