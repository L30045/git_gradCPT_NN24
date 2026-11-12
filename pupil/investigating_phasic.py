#%% load library
"""
Phasic pupil responses to city-correct trials in gradCPT.

1. For each run, epoch city-correct trials (-8 s to +8 s around stimulus onset),
   stack them, and sort by smoothed VTC.
2. Test whether the pupil carries a rhythmic component at the stimulus rate
   (1/0.8 s = 1.25 Hz): per-trial power around 1.25 Hz vs VTC.
3. Test whether that rhythm is phase-locked to the stimulus train
   (inter-trial phase coherence vs a time-shift surrogate, pupil-vs-stimulus
   phase lag, and phase locking vs VTC).

Note: city trials are 0.8 s apart, so neighboring 16 s epochs share ~95% of
their samples. Trial-level p-values are therefore not independent; the
statistics below are done at the subject level.
"""
import numpy as np
import scipy as sp
import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
sys.path.append(os.path.join(git_path, 'modeling'))  # params_setting.py
from params_setting import *
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from pupil_labs import neon_recording as nr
from utils_eyetracking import preprocess_pupil

#%% parameters
baseline_length = 8      # sec before onset
epoch_length    = 8      # sec after onset
soa             = 0.8    # sec, stimulus onset asynchrony
f_stim          = 1 / soa
f_band          = (f_stim - 0.15, f_stim + 0.15)   # band for "power around 1.25 Hz"
f_noise_offset  = (0.25, 0.5)                       # neighbor bins (|f - f_stim|) for SNR
n_perm          = 200    # surrogates for the phase-locking null
vis_smooth      = 5      # % of trials used for the raster's moving average
n_vtc_bins      = 5      # VTC quantile bins for phase-locking vs VTC
rng             = np.random.default_rng(0)
max_subjects    = None   # set to an int for a quick test
save_dir        = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eye_tracking/phasic/'
os.makedirs(save_dir, exist_ok=True)

#%% functions
def smoothing_VTC_gaussian_array(vtc, L=20, alpha=2.5, truncate=4):
    """Gaussian smoothing of VTC (same kernel as cmp_vtc_pupil.py)."""
    sigma = (L - 1) / (2 * alpha)
    return sp.ndimage.gaussian_filter1d(vtc, sigma=sigma, truncate=truncate)

def _parse_eyetracking_readme(subj_snirf_raw):
    """Parse 'Eye-Tracking Folders' section -> {(task_lower, run_int): folder_name}."""
    readme_file = next(
        (f for f in sorted(os.listdir(subj_snirf_raw)) if f.upper().startswith("README")), None)
    if not readme_file:
        return {}
    with open(os.path.join(subj_snirf_raw, readme_file)) as _f:
        readme_text = _f.read()
    section_match = re.search(
        r"Eye-Tracking Folders\s*\n(.*?)(?:\n\n|\Z)", readme_text, re.IGNORECASE | re.DOTALL)
    if not section_match:
        return {}
    mapping = {}
    for line in section_match.group(1).splitlines():
        m = re.match(r"(\d{4}-\S+)\s+(\S+)\s+Run\s+(\d+)", line.strip(), re.IGNORECASE)
        if m:
            mapping[(m.group(2).lower(), int(m.group(3)))] = m.group(1)
    return mapping

def epoch_with_times(onsets, t_cont, x_cont, sfreq, t_pre, t_post):
    """
    Epoch a continuous signal around onsets.
    Returns epochs (n_trials, n_samples) and the exact sample times relative
    to each onset (n_trials, n_samples). Out-of-range epochs are NaN.
    """
    n_pre, n_post = int(round(sfreq * t_pre)), int(round(sfreq * t_post))
    n_samp = n_pre + n_post
    idx0 = np.searchsorted(t_cont, onsets)
    idx0 = np.clip(idx0, 1, len(t_cont) - 1)
    idx0 = np.where(np.abs(t_cont[idx0 - 1] - onsets) < np.abs(t_cont[idx0] - onsets), idx0 - 1, idx0)
    starts = idx0 - n_pre
    valid = (starts >= 0) & (starts + n_samp <= len(x_cont))
    ep = np.full((len(onsets), n_samp), np.nan)
    t_rel = np.tile((np.arange(n_samp) - n_pre) / sfreq, (len(onsets), 1))
    ii = starts[valid][:, None] + np.arange(n_samp)[None, :]
    ep[valid] = x_cont[ii]
    t_rel[valid] = t_cont[ii] - onsets[valid][:, None]
    return ep, t_rel

def fourier_at(ep, t_rel, f, taper):
    """
    Complex Fourier coefficient at frequency f with time origin at the
    stimulus onset (t=0). Each trial is linearly detrended and tapered.
    For x(t) = A cos(2*pi*f*t + phi), angle(X) = phi, i.e. phi = 0 means a
    pupil peak at stimulus onset.
    """
    x = sp.signal.detrend(ep, axis=1, type='linear')
    return np.sum(taper * x * np.exp(-2j * np.pi * f * t_rel), axis=1) * 2 / np.sum(taper)

def psd_trials(ep, sfreq, taper):
    """Single-trial power spectra (detrended, tapered)."""
    x = sp.signal.detrend(ep, axis=1, type='linear') * taper
    freqs = np.fft.rfftfreq(ep.shape[1], 1 / sfreq)
    pxx = np.abs(np.fft.rfft(x, axis=1)) ** 2 / (sfreq * np.sum(taper ** 2))
    pxx[:, 1:-1] *= 2
    return freqs, pxx

def stim_fourier_at(trial_onsets, all_onsets, t_pre, t_post, f):
    """
    Fourier coefficient of the stimulus impulse train (all stimuli, city + mnt)
    inside each trial's window, with the same Hann taper as the pupil.
    """
    out = np.zeros(len(trial_onsets), dtype=complex)
    T = t_pre + t_post
    for i, t0 in enumerate(trial_onsets):
        tk = all_onsets[(all_onsets >= t0 - t_pre) & (all_onsets < t0 + t_post)] - t0
        w = 0.5 - 0.5 * np.cos(2 * np.pi * (tk + t_pre) / T)
        out[i] = np.sum(w * np.exp(-2j * np.pi * f * tk))
    return out

def save_fig(fig, name):
    """Save a figure as PNG and SVG to save_dir."""
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(save_dir, f'{name}.{ext}'), dpi=300, bbox_inches='tight')

def itpc(phases):
    return np.abs(np.mean(np.exp(1j * phases)))

def rayleigh_test(phases):
    """Rayleigh test for non-uniformity (Zar 1999 approximation)."""
    n = len(phases)
    R = n * itpc(phases)
    z = R ** 2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return z, min(p, 1.0)

#%% epoch city-correct trials for each run
dirs = os.listdir(project_path)
subject_list = sorted([d for d in dirs if 'sub' in d])
if max_subjects is not None:
    subject_list = subject_list[:max_subjects]
physio_version = 'latest'

run_data = []          # one dict per run
missing_list = []
for subj in subject_list:
    subj_id = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    subj_snirf_raw = os.path.join(project_path, 'sourcedata', 'raw', subj, 'nirs')
    if not os.path.isdir(subj_neon_dir) or not os.path.isdir(subj_nirs):
        missing_list.append(f"{subj}: no Neon/nirs folder")
        continue
    neon_dirs_subj = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])
    mapping = _parse_eyetracking_readme(subj_snirf_raw) if os.path.isdir(subj_snirf_raw) else {}
    for run_id in range(1, 4):
        physio_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_{physio_version}.tsv")
        if not os.path.isfile(physio_file):
            physio_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
        event_file = os.path.join(subj_nirs, f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv")
        if not os.path.isfile(physio_file) or not os.path.isfile(event_file):
            missing_list.append(f"{subj} run-{run_id:02d}: no physio/event file")
            continue
        neon_data = pd.read_csv(physio_file, sep='\t')
        rec = None
        if neon_dirs_subj and ('gradcpt', run_id) in mapping:
            rec = nr.open(os.path.join(subj_neon_dir, mapping[('gradcpt', run_id)]))
        events_df = pd.read_csv(event_file, sep='\t')
        t_neon, pupil_d = preprocess_pupil(neon_data, rec=rec, detrend_order=2,
                                           is_rm_phasic=False, events_df=events_df)
        if t_neon is None:
            missing_list.append(f"{subj} run-{run_id:02d}: too much missing pupil data")
            continue
        t_neon = np.asarray(t_neon, dtype=float)
        pupil_d = np.asarray(pupil_d, dtype=float)
        sfreq = np.round(np.median(1 / np.diff(t_neon)))

        city_correct = ((events_df['trial_type'] == 'city') & (events_df['response_code'] != 0)).values
        vtc = smoothing_VTC_gaussian_array(events_df['VTC'].values, L=20)
        onsets = events_df['onset'].values[city_correct]
        ep, t_rel = epoch_with_times(onsets, t_neon, pupil_d, sfreq, baseline_length, epoch_length)
        keep = ~np.any(np.isnan(ep), axis=1)
        run_soa = np.median(np.diff(events_df['onset'].values))
        print(f"{subj} run-{run_id:02d}: sfreq={sfreq} Hz, SOA={run_soa:.4f} s, "
              f"{keep.sum()}/{len(keep)} city-correct epochs kept")
        run_data.append({
            'subj': subj, 'run': run_id, 'sfreq': sfreq, 'soa': run_soa,
            'epochs': ep[keep], 't_rel': t_rel[keep],
            'onsets': onsets[keep], 'vtc': vtc[city_correct][keep],
            'all_onsets': events_df['onset'].values,
            't_neon': t_neon, 'pupil_d': pupil_d,
        })
print('\n'.join(missing_list))
subjects = sorted({r['subj'] for r in run_data})
n_subjects = len(subjects)
assert len({r['sfreq'] for r in run_data}) == 1, 'Runs have different sampling rates'
sfreq = run_data[0]['sfreq']
n_samp = run_data[0]['epochs'].shape[1]
t_epoch = (np.arange(n_samp) - int(round(sfreq * baseline_length))) / sfreq
taper = sp.signal.windows.hann(n_samp, sym=False)

#%% Raster: city-correct epochs stacked and sorted by VTC (pooled across runs)
# Left: baseline-corrected pupil. Right: 0.5 Hz high-passed pupil to expose
# the stimulus-rate ripple. Rows are smoothed over vis_smooth % of trials.
b_hp, a_hp = sp.signal.butter(3, 0.5, btype='highpass', fs=sfreq)
all_ep, all_ep_hp, all_vtc = [], [], []
for r in run_data:
    bl = np.mean(r['epochs'][:, t_epoch < 0], axis=1, keepdims=True)
    all_ep.append(r['epochs'] - bl)
    # high-pass on the continuous run, then re-epoch, to avoid edge effects
    hp_cont = sp.signal.filtfilt(b_hp, a_hp, r['pupil_d'])
    ep_hp, _ = epoch_with_times(r['onsets'], r['t_neon'], hp_cont, sfreq, baseline_length, epoch_length)
    all_ep_hp.append(ep_hp)
    all_vtc.append(r['vtc'])
all_ep, all_ep_hp, all_vtc = np.vstack(all_ep), np.vstack(all_ep_hp), np.concatenate(all_vtc)
sort_idx = np.argsort(all_vtc)
win = max(1, int(len(sort_idx) * vis_smooth / 100))
smooth_rows = lambda a: sp.ndimage.uniform_filter1d(a, size=win, axis=0, mode='nearest')
rast = smooth_rows(all_ep[sort_idx])
rast_hp = smooth_rows(all_ep_hp[sort_idx])

fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 6, 6]}, sharey=True)
axes[0].plot(all_vtc[sort_idx], np.arange(len(sort_idx)), color='steelblue')
axes[0].set_xlabel('VTC (smoothed)')
axes[0].set_ylabel('Trial (sorted by VTC, low → high)')
axes[0].set_ylim(len(sort_idx), 0)
for ax, dat, ttl in zip(axes[1:], [rast, rast_hp], ['Baseline-corrected', 'High-passed (>0.5 Hz)']):
    vmax = np.nanpercentile(np.abs(dat), 99)
    im = ax.imshow(dat, aspect='auto', origin='upper', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   extent=[t_epoch[0], t_epoch[-1], len(sort_idx), 0], interpolation='nearest')
    for k in np.arange(-np.floor(baseline_length / soa), np.floor(epoch_length / soa) + 1):
        ax.axvline(k * soa, color='k', lw=0.3, alpha=0.4)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Time from city-correct onset (s)')
    ax.set_title(ttl)
    plt.colorbar(im, ax=ax, label='Pupil diameter (mm)')
fig.suptitle(f'City-correct epochs sorted by VTC (N={n_subjects} subjects, {len(sort_idx)} trials); '
             f'grey lines = stimulus onsets')
plt.tight_layout()
save_fig(fig, 'raster_city_correct_by_vtc')

# Grand-average (subject-level) high-passed epoch
fig, ax = plt.subplots(figsize=(10, 3.5))
subj_avg_hp = []
for s in subjects:
    eps = np.vstack([ep for ep, r in zip(np.split(all_ep_hp, np.cumsum([len(r['vtc']) for r in run_data])[:-1]), run_data)
                     if r['subj'] == s])
    subj_avg_hp.append(np.mean(eps, axis=0))
subj_avg_hp = np.array(subj_avg_hp)
m, se = subj_avg_hp.mean(0), subj_avg_hp.std(0, ddof=1) / np.sqrt(n_subjects)
ax.plot(t_epoch, m, color='k')
ax.fill_between(t_epoch, m - se, m + se, color='k', alpha=0.25)
for k in np.arange(-np.floor(baseline_length / soa), np.floor(epoch_length / soa) + 1):
    ax.axvline(k * soa, color='tab:red', lw=0.5, alpha=0.5)
ax.set_xlabel('Time from city-correct onset (s)')
ax.set_ylabel('High-passed pupil (mm)')
ax.set_title(f'Grand-average high-passed pupil (mean ± SEM across N={n_subjects} subjects); red = stimulus onsets')
plt.tight_layout()
save_fig(fig, 'grand_avg_highpassed_epoch')

#%% Spectral analysis: power around the stimulus frequency
# Per trial: band power in f_band, and SNR = power at f_stim / mean power in
# neighboring bins (removes the 1/f background and between-subject gain).
for r in run_data:
    freqs, pxx = psd_trials(r['epochs'], sfreq, taper)
    band = (freqs >= f_band[0]) & (freqs <= f_band[1])
    dist = np.abs(freqs - f_stim)
    noise = (dist >= f_noise_offset[0]) & (dist <= f_noise_offset[1])
    r['freqs'] = freqs
    r['pxx'] = pxx
    r['band_power'] = np.mean(pxx[:, band], axis=1)
    i_stim = np.argmin(dist)
    r['snr'] = pxx[:, i_stim] / np.mean(pxx[:, noise], axis=1)
    # evoked spectrum: spectrum of the run-average epoch
    _, pxx_ev = psd_trials(r['epochs'].mean(0, keepdims=True), sfreq, taper)
    r['pxx_evoked'] = pxx_ev[0]
freqs = run_data[0]['freqs']
print(f"Frequency resolution: {freqs[1]:.4f} Hz; f_stim bin = {freqs[np.argmin(np.abs(freqs - f_stim))]:.4f} Hz")

# Subject-level average spectra (total = mean of single-trial power; evoked = power of the average)
subj_psd = np.array([np.mean(np.vstack([r['pxx'] for r in run_data if r['subj'] == s]), 0) for s in subjects])
subj_psd_ev = np.array([np.mean([r['pxx_evoked'] for r in run_data if r['subj'] == s], 0) for s in subjects])
fmask = (freqs > 0) & (freqs <= 4)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, P, ttl in zip(axes, [subj_psd, subj_psd_ev], ['Total (single-trial) power', 'Evoked (trial-average) power']):
    lp = np.log10(P[:, fmask])
    m, se = lp.mean(0), lp.std(0, ddof=1) / np.sqrt(n_subjects)
    ax.plot(freqs[fmask], m, color='k')
    ax.fill_between(freqs[fmask], m - se, m + se, color='k', alpha=0.25)
    for h in range(1, 4):
        ax.axvline(h * f_stim, color='tab:red', ls='--', lw=0.8)
    ax.axvspan(*f_band, color='tab:red', alpha=0.1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log10 PSD (mm²/Hz)')
    ax.set_title(ttl)
fig.suptitle(f'Pupil spectrum of city-correct epochs (N={n_subjects}); red = 1.25 Hz and harmonics')
plt.tight_layout()
save_fig(fig, 'spectrum_city_correct')

#%% Scatter: power around 1.25 Hz vs VTC
# Pooled trials (colored by subject) + per-subject Spearman rho, tested
# across subjects (Fisher z, one-sample t-test).
metrics = [('band_power', f'log10 band power {f_band[0]:.2f}-{f_band[1]:.2f} Hz', True),
           ('snr', 'SNR at 1.25 Hz (vs neighbors)', True)]
cmap = plt.get_cmap('tab20', n_subjects)
fig, axes = plt.subplots(2, len(metrics), figsize=(6 * len(metrics), 9),
                         gridspec_kw={'height_ratios': [3, 1.3]})
for j, (key, label, logy) in enumerate(metrics):
    ax = axes[0, j]
    rhos = []
    for si, s in enumerate(subjects):
        v = np.concatenate([r['vtc'] for r in run_data if r['subj'] == s])
        y = np.concatenate([r[key] for r in run_data if r['subj'] == s])
        y = np.log10(y) if logy else y
        ax.scatter(v, y, s=3, alpha=0.3, color=cmap(si))
        rhos.append(sp.stats.spearmanr(v, y).statistic)
    rhos = np.array(rhos)
    z = np.arctanh(rhos)
    t_res = sp.stats.ttest_1samp(z, 0)
    v_all = np.concatenate([r['vtc'] for r in run_data])
    y_all = np.concatenate([r[key] for r in run_data])
    y_all = np.log10(y_all) if logy else y_all
    # binned mean across pooled trials
    edges = np.quantile(v_all, np.linspace(0, 1, 11))
    bi = np.clip(np.digitize(v_all, edges[1:-1]), 0, 9)
    ax.plot([v_all[bi == b].mean() for b in range(10)], [y_all[bi == b].mean() for b in range(10)],
            'k-o', ms=4, label='VTC-decile mean (pooled)')
    ax.set_xlabel('VTC (smoothed)')
    ax.set_ylabel(('log10 ' if logy and key == 'snr' else '') + label)
    ax.set_title(f'{label}\nmean subject ρ = {np.tanh(z.mean()):.3f}, '
                 f't({n_subjects - 1}) = {t_res.statistic:.2f}, p = {t_res.pvalue:.3g}')
    ax.legend(loc='upper right', fontsize=8)
    # per-subject rho strip
    axes[1, j].axvline(0, color='grey', lw=0.8)
    axes[1, j].scatter(rhos, rng.uniform(-0.3, 0.3, n_subjects), color=[cmap(i) for i in range(n_subjects)])
    axes[1, j].set_yticks([])
    axes[1, j].set_xlabel('Per-subject Spearman ρ (VTC vs metric)')
fig.suptitle(f'Pupil power at the stimulus rate vs VTC, city-correct trials (N={n_subjects})')
plt.tight_layout()
save_fig(fig, 'power_vs_vtc')

#%% Phase: is the 1.25 Hz rhythm locked to the stimulus?
# Phase convention: phi = angle of the Fourier coefficient with t = 0 at the
# trial onset. phi = 0 -> pupil peak at stimulus onset; the peak latency
# within a cycle is (-phi / (2*pi*f)) mod SOA.
# (a) ITPC of pupil phase across trials vs a null where each trial's
#     reference time is jittered by U(0, SOA) and the data re-epoched.
# (b) Pupil phase relative to the actual stimulus train (all stimuli in the
#     window), which absorbs small SOA jitter.
for r in run_data:
    f_r = 1 / r['soa']       # use the run's measured stimulus rate
    X = fourier_at(r['epochs'], r['t_rel'], f_r, taper)
    S = stim_fourier_at(r['onsets'], r['all_onsets'], baseline_length, epoch_length, f_r)
    r['X'] = X
    r['phase'] = np.angle(X)
    r['phase_rel'] = np.angle(X * np.conj(S))      # pupil phase - stimulus-train phase
    # null: re-epoch at randomly shifted references
    null = np.zeros(n_perm)
    for p in range(n_perm):
        shift = rng.uniform(0, r['soa'], len(r['onsets']))
        ep_s, t_s = epoch_with_times(r['onsets'] + shift, r['t_neon'], r['pupil_d'], sfreq,
                                     baseline_length, epoch_length)
        ok = ~np.any(np.isnan(ep_s), axis=1)
        null[p] = itpc(np.angle(fourier_at(ep_s[ok], t_s[ok], f_r, taper)))
    r['itpc'] = itpc(r['phase'])
    r['itpc_null'] = null

# subject-level summaries (pool runs)
subj_res = []
for s in subjects:
    rs = [r for r in run_data if r['subj'] == s]
    ph = np.concatenate([r['phase_rel'] for r in rs])
    n_tr = len(ph)
    # null ITPC averaged over runs weighted by trial count
    w = np.array([len(r['phase']) for r in rs]) / n_tr
    null = np.sum([wi * r['itpc_null'] for wi, r in zip(w, rs)], axis=0)
    obs = np.sum([wi * r['itpc'] for wi, r in zip(w, rs)])
    subj_res.append({'subj': s, 'n_trials': n_tr,
                     'itpc': obs, 'itpc_null_mean': null.mean(),
                     'p_perm': (np.sum(null >= obs) + 1) / (n_perm + 1),
                     'itpc_rel': itpc(ph),
                     'mean_phase_rel': np.angle(np.mean(np.exp(1j * ph)))})
subj_res = pd.DataFrame(subj_res)
subj_res['peak_latency_s'] = np.mod(-subj_res['mean_phase_rel'] / (2 * np.pi * f_stim), soa)
print(subj_res.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

t_itpc = sp.stats.ttest_rel(subj_res['itpc'], subj_res['itpc_null_mean'])
z_grp, p_grp = rayleigh_test(subj_res['mean_phase_rel'].values)
R_grp = itpc(subj_res['mean_phase_rel'].values)
grp_phase = np.angle(np.mean(np.exp(1j * subj_res['mean_phase_rel'].values)))
print(f"ITPC observed vs surrogate: t({n_subjects - 1}) = {t_itpc.statistic:.2f}, p = {t_itpc.pvalue:.3g}; "
      f"{(subj_res['p_perm'] < 0.05).sum()}/{n_subjects} subjects p_perm < 0.05")
print(f"Across-subject phase consistency: R = {R_grp:.3f}, Rayleigh z = {z_grp:.2f}, p = {p_grp:.3g}; "
      f"group phase = {np.degrees(grp_phase):.1f} deg "
      f"(peak {np.mod(-grp_phase / (2 * np.pi * f_stim), soa):.3f} s after stimulus onset)")

fig = plt.figure(figsize=(16, 4.8))
# (1) ITPC observed vs null per subject
ax = fig.add_subplot(1, 3, 1)
order = np.argsort(subj_res['itpc'].values)
for k, i in enumerate(order):
    ax.plot([k, k], [subj_res['itpc_null_mean'].iloc[i], subj_res['itpc'].iloc[i]], color='grey', lw=0.8)
ax.scatter(np.arange(n_subjects), subj_res['itpc'].values[order], color='k', s=18, label='Observed', zorder=3)
ax.scatter(np.arange(n_subjects), subj_res['itpc_null_mean'].values[order], color='tab:orange', s=18,
           label='Shift surrogate (mean)', zorder=3)
sig = subj_res['p_perm'].values[order] < 0.05
ax.scatter(np.arange(n_subjects)[sig], subj_res['itpc'].values[order][sig], facecolor='none',
           edgecolor='tab:red', s=60, label='p_perm < 0.05', zorder=4)
ax.set_xlabel('Subject (sorted)')
ax.set_ylabel('ITPC at 1.25 Hz')
ax.set_title(f'Inter-trial phase coherence\npaired t({n_subjects - 1}) = {t_itpc.statistic:.2f}, p = {t_itpc.pvalue:.3g}')
ax.legend(fontsize=8)
# (2) polar plot of pupil-vs-stimulus phase per subject
ax = fig.add_subplot(1, 3, 2, projection='polar')
for i, row in subj_res.iterrows():
    ax.plot([0, row['mean_phase_rel']], [0, row['itpc_rel']], color=cmap(i), lw=1.5)
ax.plot([0, grp_phase], [0, R_grp * subj_res['itpc_rel'].mean()], color='k', lw=3)
ax.set_title(f'Pupil phase re: stimulus train (0° = peak at onset)\n'
             f'Rayleigh across subjects z = {z_grp:.2f}, p = {p_grp:.3g}', fontsize=10)
# (3) pooled phase histogram (trials, run-relative)
ax = fig.add_subplot(1, 3, 3, projection='polar')
ph_all = np.concatenate([r['phase_rel'] for r in run_data])
ax.hist(ph_all, bins=36, density=True, color='steelblue', alpha=0.8)
ax.set_title('All trials pooled: pupil - stimulus phase', fontsize=10)
plt.tight_layout()
save_fig(fig, 'phase_locking_stimulus')

#%% Phase locking vs VTC
# Per subject: split trials into VTC quantile bins (equal trial counts, so
# ITPC bias is matched) and compute ITPC and phase-locked amplitude.
# Phase-locked amplitude per trial = Re(X * exp(-i*phi_subject)): the
# component of the 1.25 Hz response aligned with that subject's mean phase.
itpc_bins = np.full((n_subjects, n_vtc_bins), np.nan)
pla_bins = np.full((n_subjects, n_vtc_bins), np.nan)
rho_pla = []
for si, s in enumerate(subjects):
    rs = [r for r in run_data if r['subj'] == s]
    v = np.concatenate([r['vtc'] for r in rs])
    X = np.concatenate([r['X'] for r in rs])
    ph = np.concatenate([r['phase_rel'] for r in rs])
    # rotate X into the stimulus-train reference so runs can be pooled
    Xrel = np.abs(X) * np.exp(1j * ph)
    phi_s = np.angle(np.mean(np.exp(1j * ph)))
    pla = np.real(Xrel * np.exp(-1j * phi_s))
    rho_pla.append(sp.stats.spearmanr(v, pla).statistic)
    q = np.quantile(v, np.linspace(0, 1, n_vtc_bins + 1))
    b = np.clip(np.digitize(v, q[1:-1]), 0, n_vtc_bins - 1)
    for k in range(n_vtc_bins):
        if np.sum(b == k) > 1:
            itpc_bins[si, k] = itpc(ph[b == k])
            pla_bins[si, k] = np.mean(pla[b == k])
rho_pla = np.array(rho_pla)
t_pla = sp.stats.ttest_1samp(np.arctanh(rho_pla), 0)
# linear trend of ITPC across VTC bins per subject
slope_itpc = np.array([np.polyfit(np.arange(n_vtc_bins), row, 1)[0] for row in itpc_bins])
t_slope = sp.stats.ttest_1samp(slope_itpc, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
xb = np.arange(1, n_vtc_bins + 1)
for ax, M, ylab, ttl in zip(axes[:2], [itpc_bins, pla_bins],
                            ['ITPC at 1.25 Hz', 'Phase-locked amplitude (mm)'],
                            [f'ITPC vs VTC bin\nslope t({n_subjects - 1}) = {t_slope.statistic:.2f}, p = {t_slope.pvalue:.3g}',
                             'Phase-locked amplitude vs VTC bin']):
    for row in M:
        ax.plot(xb, row, color='grey', alpha=0.3, lw=0.8)
    m, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(n_subjects)
    ax.errorbar(xb, m, se, color='k', marker='o', capsize=3, lw=2)
    ax.set_xticks(xb)
    ax.set_xlabel('VTC quantile bin (1 = lowest VTC / in the zone)')
    ax.set_ylabel(ylab)
    ax.set_title(ttl)
axes[2].axvline(0, color='grey', lw=0.8)
axes[2].hist(rho_pla, bins=15, color='steelblue')
axes[2].set_xlabel('Per-subject Spearman ρ (VTC vs phase-locked amplitude)')
axes[2].set_ylabel('# subjects')
axes[2].set_title(f'mean ρ = {np.tanh(np.arctanh(rho_pla).mean()):.3f}, '
                  f't({n_subjects - 1}) = {t_pla.statistic:.2f}, p = {t_pla.pvalue:.3g}')
fig.suptitle(f'Stimulus phase locking of the 1.25 Hz pupil rhythm vs VTC (N={n_subjects})')
plt.tight_layout()
save_fig(fig, 'phase_locking_vs_vtc')
plt.show()
