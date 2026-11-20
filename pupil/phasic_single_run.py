#%% load library
"""
Single-subject, single-run version of investigating_phasic.py.

Question: within one run, is the phase of the 1.25 Hz (stimulus-rate) pupil
rhythm stable from trial to trial, or does it drift / jump?

For one subject and one run (set `subj`, `run_id` below):
1. Raster of city-correct epochs (-8 to +8 s), rows in trial order (time in
   run) and sorted by the covariate (smoothed VTC or EEG theta).
2. Spectrum of the run's epochs (total and evoked) around 1.25 Hz.
3. Per-trial phase relative to the stimulus train, plotted against time in
   the run, with a sliding circular mean and resultant length R. Two phase
   estimates are used:
     - long : the same 16 s Hann-tapered epoch as investigating_phasic.py.
              Neighbouring trials share ~95% of their samples, so this phase
              is smooth across trials by construction.
     - short: a `short_cycles` x SOA window centred on onset of the
              high-passed pupil. Much less overlap, so trial-to-trial
              variability is visible.
4. Inter-trial phase coherence vs a time-shift surrogate (same as the group
   script, now at the trial level), plus a Rayleigh test on a subset of
   non-overlapping trials (independent windows).
5. Phase stability vs time separation: mean cos(phi_i - phi_j) for pairs of
   trials binned by |t_i - t_j|. Flat and > 0 = phase locked for the whole
   run; decaying to 0 = phase drifts.
6. Power and phase locking vs the covariate (trial-level; p-values are not
   independent because of the overlap, see note in investigating_phasic.py).
7. Whole-run 1.25 Hz pupil power (Welch PSD of the continuous pupil) and the
   run's behavior (city/mnt correct/incorrect counts, RT mean and CV), saved as one
   row per run for the across-run comparison in summary_phasic_single_run.ipynb.

Phase convention (same as investigating_phasic.py): phi = 0 means a pupil
peak at stimulus onset; peak latency within a cycle = (-phi / (2*pi*f)) mod SOA.
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
subj            = 'sub-723'
run_id          = 1
physio_version  = 'latest'
baseline_length = 8      # sec before onset
epoch_length    = 8      # sec after onset
soa             = 0.8    # sec, nominal SOA (the run's measured SOA is used for phase)
f_stim          = 1 / soa
f_band          = (f_stim - 0.15, f_stim + 0.15)   # band for "power around 1.25 Hz"
f_noise_offset  = (0.25, 0.5)                       # neighbor bins (|f - f_stim|) for SNR
f_hp            = 0.5    # Hz, high-pass for the raster and the short-window phase
short_cycles    = 2      # length of the short phase window in stimulus cycles
n_slide         = 20     # trials in the sliding circular mean
n_perm          = 500    # surrogates for the phase-locking null
n_cov_bins      = 5      # covariate quantile bins
dt_bin          = 5      # sec, bin width for phase stability vs time separation
covariate       = 'vtc'  # 'vtc' or 'theta'
theta_feature   = 'fz_theta_win'
theta_zscore    = True
rng             = np.random.default_rng(0)
output_mode     = 'show'   # 'save': write PNG/SVG to save_dir; 'show': display interactively (plt.show), nothing saved
# command line (ignored in interactive/IPython sessions):
#   python phasic_single_run.py [subj] [run_id] [covariate] [output_mode]
#   e.g. python phasic_single_run.py sub-723 1 vtc save
if 'ipykernel' not in sys.modules and len(sys.argv) > 1:
    _args = sys.argv[1:] + [None] * 4
    subj        = _args[0] or subj
    run_id      = int(_args[1]) if _args[1] else run_id
    covariate   = _args[2] or covariate
    output_mode = _args[3] or output_mode
    if output_mode == 'save':
        import matplotlib
        matplotlib.use('Agg')
save_dir        = os.path.join(project_path, 'derivatives', 'eye_tracking', 'phasic', subj)
eeg_feat_dir    = os.path.join(project_path, 'derivatives', 'eeg')
assert covariate in ('vtc', 'theta'), covariate
assert output_mode in ('save', 'show'), output_mode
if output_mode == 'save':
    os.makedirs(save_dir, exist_ok=True)
cov_name  = covariate if covariate == 'vtc' else theta_feature
cov_label = 'VTC (smoothed)' if covariate == 'vtc' else \
            f"{theta_feature}{' (z within run)' if theta_zscore else ''}"
cov_short = 'VTC' if covariate == 'vtc' else 'theta'
cov_low   = 'lowest VTC / in the zone' if covariate == 'vtc' else 'lowest theta'
tag       = f'{subj}_run-{run_id:02d}'

#%% functions (same as investigating_phasic.py)
def smoothing_VTC_gaussian_array(vtc, L=20, alpha=2.5, truncate=4):
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
    """Epochs (n_trials, n_samples) and exact sample times re: onset. Out-of-range epochs are NaN."""
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
    """Complex Fourier coefficient at f, t = 0 at onset (detrended, tapered)."""
    x = sp.signal.detrend(ep, axis=1, type='linear')
    return np.sum(taper * x * np.exp(-2j * np.pi * f * t_rel), axis=1) * 2 / np.sum(taper)

def psd_trials(ep, sfreq, taper):
    x = sp.signal.detrend(ep, axis=1, type='linear') * taper
    freqs = np.fft.rfftfreq(ep.shape[1], 1 / sfreq)
    pxx = np.abs(np.fft.rfft(x, axis=1)) ** 2 / (sfreq * np.sum(taper ** 2))
    pxx[:, 1:-1] *= 2
    return freqs, pxx

def stim_fourier_at(trial_onsets, all_onsets, t_pre, t_post, f):
    """Fourier coefficient of the stimulus impulse train in each trial's (Hann-tapered) window."""
    out = np.zeros(len(trial_onsets), dtype=complex)
    T = t_pre + t_post
    for i, t0 in enumerate(trial_onsets):
        tk = all_onsets[(all_onsets >= t0 - t_pre) & (all_onsets < t0 + t_post)] - t0
        w = 0.5 - 0.5 * np.cos(2 * np.pi * (tk + t_pre) / T)
        out[i] = np.sum(w * np.exp(-2j * np.pi * f * tk))
    return out

def itpc(phases):
    phases = phases[np.isfinite(phases)]
    return np.abs(np.mean(np.exp(1j * phases)))

def circ_mean(phases):
    return np.angle(np.mean(np.exp(1j * phases)))

def rayleigh_test(phases):
    """Rayleigh test for non-uniformity (Zar 1999 approximation)."""
    n = len(phases)
    R = n * itpc(phases)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return R ** 2 / n, min(p, 1.0)

def non_overlapping(onsets, min_gap):
    """Greedy subset of trials whose onsets are at least min_gap apart."""
    keep, last = [], -np.inf
    for i, t in enumerate(onsets):
        if t - last >= min_gap:
            keep.append(i)
            last = t
    return np.array(keep, dtype=int)

def phase_rel(ep, t_rel, onsets, all_onsets, t_pre, t_post, f):
    """Pupil phase minus stimulus-train phase, plus the raw Fourier coefficient."""
    taper = sp.signal.windows.hann(ep.shape[1], sym=False)
    X = fourier_at(ep, t_rel, f, taper)
    S = stim_fourier_at(onsets, all_onsets, t_pre, t_post, f)
    return np.angle(X * np.conj(S)), X

def save_fig(fig, name):
    """Save (and close) the figure in 'save' mode; leave it open for plt.show() in 'show' mode."""
    if output_mode != 'save':
        return
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(save_dir, f'{tag}_{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

#%% load the run
subj_id = subj.replace('sub-', '')
subj_nirs = os.path.join(project_path, subj, 'nirs')
subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
subj_snirf_raw = os.path.join(project_path, 'sourcedata', 'raw', subj, 'nirs')
stem = f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}"
physio_file = os.path.join(subj_nirs, f"{stem}_recording-eyetracking_physio_{physio_version}.tsv")
if not os.path.isfile(physio_file):
    physio_file = os.path.join(subj_nirs, f"{stem}_recording-eyetracking_physio.tsv")
event_file = os.path.join(subj_nirs, f"{stem}_events.tsv")
assert os.path.isfile(physio_file) and os.path.isfile(event_file), f'{tag}: no physio/event file'

neon_data = pd.read_csv(physio_file, sep='\t')
events_df = pd.read_csv(event_file, sep='\t')
mapping = _parse_eyetracking_readme(subj_snirf_raw) if os.path.isdir(subj_snirf_raw) else {}
rec = nr.open(os.path.join(subj_neon_dir, mapping[('gradcpt', run_id)])) \
    if ('gradcpt', run_id) in mapping else None
t_neon, pupil_d = preprocess_pupil(neon_data, rec=rec, detrend_order=2,
                                   is_rm_phasic=False, events_df=events_df)
assert t_neon is not None, f'{tag}: too much missing pupil data'
t_neon, pupil_d = np.asarray(t_neon, float), np.asarray(pupil_d, float)
sfreq = np.round(np.median(1 / np.diff(t_neon)))
b_hp, a_hp = sp.signal.butter(3, f_hp, btype='highpass', fs=sfreq)
pupil_hp = sp.signal.filtfilt(b_hp, a_hp, pupil_d)

all_onsets = events_df['onset'].values
run_soa = np.median(np.diff(all_onsets))
f_r = 1 / run_soa
# BIDS response_code: city 1 = correct, -1 = omission; mnt 0 = correct withhold, -2 = commission
city_correct = ((events_df['trial_type'] == 'city') & (events_df['response_code'] == 1)).values
if covariate == 'vtc':
    cov_all = smoothing_VTC_gaussian_array(events_df['VTC'].values, L=20)
else:
    feat_df = pd.read_csv(os.path.join(eeg_feat_dir, subj, 'features', f'{subj}_EEG_trial_features.csv.gz'))
    f_run = feat_df[feat_df['run'] == run_id].sort_values('trial_idx')
    assert len(f_run) == len(events_df), f'{tag}: EEG features do not match events'
    cov_all = f_run[theta_feature].values.astype(float)
    if theta_zscore:
        cov_all = (cov_all - np.nanmean(cov_all)) / np.nanstd(cov_all)

onsets = all_onsets[city_correct]
ep, t_rel = epoch_with_times(onsets, t_neon, pupil_d, sfreq, baseline_length, epoch_length)
keep = ~np.any(np.isnan(ep), axis=1)
ep, t_rel, onsets = ep[keep], t_rel[keep], onsets[keep]
cov = cov_all[city_correct][keep]
ep_hp, _ = epoch_with_times(onsets, t_neon, pupil_hp, sfreq, baseline_length, epoch_length)
n_tr, n_samp = ep.shape
t_epoch = (np.arange(n_samp) - int(round(sfreq * baseline_length))) / sfreq
t_run = onsets - all_onsets[0]          # time of each trial in the run
taper = sp.signal.windows.hann(n_samp, sym=False)
print(f"{tag}: sfreq = {sfreq} Hz, SOA = {run_soa:.4f} s, {n_tr}/{len(keep)} city-correct epochs kept, "
      f"{np.isfinite(cov).sum()} with a {cov_short} value")

#%% 1. Raster: trial order and sorted by the covariate
order_cov = np.argsort(cov)
order_cov = order_cov[np.isfinite(cov[order_cov])]
fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 6, 1, 6]})
vmax = np.nanpercentile(np.abs(ep_hp), 99)
for k, (idx, ylab, yv, xlab) in enumerate([
        (np.arange(n_tr), 'Trial (in order, start → end of run)', t_run, 'Time in run (s)'),
        (order_cov, f'Trial (sorted by {cov_short}, low → high)', cov[order_cov], cov_label)]):
    axes[2 * k].plot(yv, np.arange(len(idx)), color='steelblue')
    axes[2 * k].set_ylim(len(idx), 0)
    axes[2 * k].set_xlabel(xlab)
    axes[2 * k].set_ylabel(ylab)
    ax = axes[2 * k + 1]
    im = ax.imshow(ep_hp[idx], aspect='auto', origin='upper', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   extent=[t_epoch[0], t_epoch[-1], len(idx), 0], interpolation='nearest')
    for j in np.arange(-np.floor(baseline_length / run_soa), np.floor(epoch_length / run_soa) + 1):
        ax.axvline(j * run_soa, color='k', lw=0.3, alpha=0.4)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_yticks([])
    ax.set_xlabel('Time from city-correct onset (s)')
    plt.colorbar(im, ax=ax, label=f'High-passed (>{f_hp} Hz) pupil (mm)')
fig.suptitle(f'{tag}: city-correct epochs ({n_tr} trials, unsmoothed); grey lines = stimulus onsets. '
             'Vertical stripes that stay aligned down the rows = stable phase')
plt.tight_layout()
save_fig(fig, 'raster')

#%% 2. Spectrum
freqs, pxx = psd_trials(ep, sfreq, taper)
_, pxx_ev = psd_trials(ep.mean(0, keepdims=True), sfreq, taper)
dist = np.abs(freqs - f_stim)
band = (freqs >= f_band[0]) & (freqs <= f_band[1])
noise = (dist >= f_noise_offset[0]) & (dist <= f_noise_offset[1])
band_power = np.mean(pxx[:, band], axis=1)
snr = pxx[:, np.argmin(dist)] / np.mean(pxx[:, noise], axis=1)
fmask = (freqs > 0) & (freqs <= 4)
fig, ax = plt.subplots(figsize=(8, 4))
lp = np.log10(pxx[:, fmask])
m, se = lp.mean(0), lp.std(0, ddof=1) / np.sqrt(n_tr)
ax.plot(freqs[fmask], m, color='k', label='Total (mean single-trial)')
ax.fill_between(freqs[fmask], m - se, m + se, color='k', alpha=0.25)
ax.plot(freqs[fmask], np.log10(pxx_ev[0, fmask]), color='tab:blue', label='Evoked (trial average)')
for h in range(1, 4):
    ax.axvline(h * f_r, color='tab:red', ls='--', lw=0.8)
ax.axvspan(*f_band, color='tab:red', alpha=0.1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('log10 PSD (mm²/Hz)')
ax.set_title(f'{tag}: pupil spectrum of city-correct epochs; red = {f_r:.3f} Hz and harmonics')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'spectrum')

#%% 3. Per-trial phase across the run (long and short windows)
short_half = short_cycles * run_soa / 2
ph_long, X_long = phase_rel(ep, t_rel, onsets, all_onsets, baseline_length, epoch_length, f_r)
ep_s, t_s = epoch_with_times(onsets, t_neon, pupil_hp, sfreq, short_half, short_half)
ph_short, _ = phase_rel(ep_s, t_s, onsets, all_onsets, short_half, short_half, f_r)

def sliding_circ(ph, n):
    """Centered sliding circular mean and resultant length over n trials."""
    z = np.exp(1j * ph)
    zs = np.convolve(z, np.ones(n) / n, mode='same')
    valid = np.zeros(len(ph), bool)
    valid[n // 2: len(ph) - n // 2] = True
    return np.where(valid, np.angle(zs), np.nan), np.where(valid, np.abs(zs), np.nan)

# trial-level surrogate null: re-epoch at references jittered by U(0, SOA)
null_long = np.full((n_perm, n_tr), np.nan)
null_short = np.full((n_perm, n_tr), np.nan)
for p in range(n_perm):
    shift = rng.uniform(0, run_soa, n_tr)
    e1, t1 = epoch_with_times(onsets + shift, t_neon, pupil_d, sfreq, baseline_length, epoch_length)
    ok = ~np.any(np.isnan(e1), axis=1)
    null_long[p, ok] = np.angle(fourier_at(e1[ok], t1[ok], f_r, taper))
    e2, t2 = epoch_with_times(onsets + shift, t_neon, pupil_hp, sfreq, short_half, short_half)
    ok = ~np.any(np.isnan(e2), axis=1)
    null_short[p, ok] = np.angle(fourier_at(e2[ok], t2[ok], f_r, sp.signal.windows.hann(e2.shape[1], sym=False)))

phase_sets = {
    'long':  {'ph': ph_long,  'null': null_long,  'win': baseline_length + epoch_length,
              'label': f'{baseline_length + epoch_length:.0f} s window'},
    'short': {'ph': ph_short, 'null': null_short, 'win': 2 * short_half,
              'label': f'{2 * short_half:.1f} s window ({short_cycles} cycles, high-passed)'},
}
# ITPC note: the surrogate is compared with ITPC of the raw (onset-referenced)
# phase, as in investigating_phasic.py; phase re: stimulus train is used for plots.
taper_s = sp.signal.windows.hann(ep_s.shape[1], sym=False)
raw_ph = {'long': np.angle(X_long), 'short': np.angle(fourier_at(ep_s, t_s, f_r, taper_s))}
for name, d in phase_sets.items():
    d['itpc'] = itpc(raw_ph[name])
    d['itpc_null'] = np.array([itpc(row) for row in d['null']])
    d['p_perm'] = (np.sum(d['itpc_null'] >= d['itpc']) + 1) / (n_perm + 1)
    d['mean'] = circ_mean(d['ph'])
    d['R'] = itpc(d['ph'])
    d['slide_mean'], d['slide_R'] = sliding_circ(d['ph'], n_slide)
    d['indep'] = non_overlapping(onsets, d['win'])
    d['ray_z'], d['ray_p'] = rayleigh_test(d['ph'][d['indep']])
    lat = np.mod(-d['mean'] / (2 * np.pi * f_r), run_soa)
    print(f"[{name:5s}] ITPC = {d['itpc']:.3f} vs surrogate {d['itpc_null'].mean():.3f} "
          f"(p_perm = {d['p_perm']:.3g}); phase re: stimulus = {np.degrees(d['mean']):.1f} deg "
          f"(peak {lat * 1000:.0f} ms after onset), R = {d['R']:.3f}; "
          f"non-overlapping trials: n = {len(d['indep'])}, Rayleigh z = {d['ray_z']:.2f}, p = {d['ray_p']:.3g}")
    d['peak_lat'] = lat
summary_df = pd.DataFrame([
    {'window': name, 'label': d['label'], 'n_trials': n_tr, 'soa': run_soa,
     'ITPC': d['itpc'], 'ITPC_surrogate_mean': d['itpc_null'].mean(), 'p_perm': d['p_perm'],
     'phase_re_stim_deg': np.degrees(d['mean']), 'peak_latency_ms': d['peak_lat'] * 1000, 'R': d['R'],
     'n_non_overlapping': len(d['indep']), 'rayleigh_z': d['ray_z'], 'rayleigh_p': d['ray_p']}
    for name, d in phase_sets.items()])
if output_mode == 'save':
    summary_df.to_csv(os.path.join(save_dir, f'{tag}_phase_summary.csv'), index=False)

fig = plt.figure(figsize=(17, 8))
for row, (name, d) in enumerate(phase_sets.items()):
    # phase vs time in run
    ax = fig.add_subplot(2, 3, 3 * row + 1)
    ax.scatter(t_run, np.degrees(d['ph']), s=6, color='steelblue', alpha=0.6, label='single trial')
    ax.scatter(t_run[d['indep']], np.degrees(d['ph'][d['indep']]), s=18, facecolor='none',
               edgecolor='tab:red', label='non-overlapping subset')
    sm = np.degrees(d['slide_mean'])
    jump = np.r_[False, np.abs(np.diff(sm)) > 180]        # break line at ±180 wraps
    ax.plot(t_run, np.where(jump, np.nan, sm), color='k', lw=2, label=f'sliding mean ({n_slide} trials)')
    ax.axhline(np.degrees(d['mean']), color='k', ls=':', lw=1)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('Time in run (s)')
    ax.set_ylabel('Pupil phase re: stimulus (deg)')
    ax.set_title(f'{name}: {d["label"]}')
    ax.legend(fontsize=7, loc='lower right')
    ax2 = ax.twinx()
    ax2.plot(t_run, d['slide_R'], color='tab:orange', lw=1)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('sliding R', color='tab:orange')
    # polar histogram
    ax = fig.add_subplot(2, 3, 3 * row + 2, projection='polar')
    ax.hist(d['ph'], bins=36, density=True, color='steelblue', alpha=0.8)
    ax.plot([0, d['mean']], [0, d['R'] * ax.get_ylim()[1]], color='k', lw=2.5)
    ax.set_yticklabels([])
    ax.set_title(f'phase re: stimulus, mean = {np.degrees(d["mean"]):.0f}°, R = {d["R"]:.2f}\n'
                 f'Rayleigh (n = {len(d["indep"])} non-overlapping) p = {d["ray_p"]:.3g}', fontsize=9)
    # ITPC vs surrogate
    ax = fig.add_subplot(2, 3, 3 * row + 3)
    ax.hist(d['itpc_null'], bins=30, color='tab:orange', alpha=0.7, label='shift surrogate')
    ax.axvline(d['itpc'], color='k', lw=2, label='observed')
    ax.set_xlabel('ITPC at stimulus rate')
    ax.set_ylabel('# surrogates')
    ax.set_title(f'ITPC = {d["itpc"]:.3f}, p_perm = {d["p_perm"]:.3g}')
    ax.legend(fontsize=8)
fig.suptitle(f'{tag}: phase of the {f_r:.3f} Hz pupil rhythm across trials (0° = peak at stimulus onset)')
plt.tight_layout()
save_fig(fig, 'phase_across_trials')

#%% 4. Phase stability vs time separation between trials
# For every trial pair, cos(phi_i - phi_j); averaged in bins of |t_i - t_j|.
# Pairs closer than the window length share data, so the long-window curve is
# trivially high there (shaded).
dt_pairs = np.abs(t_run[:, None] - t_run[None, :])
iu = np.triu_indices(n_tr, k=1)
edges = np.arange(0, dt_pairs.max() + dt_bin, dt_bin)
centers = edges[:-1] + dt_bin / 2
fig, ax = plt.subplots(figsize=(9, 4))
for (name, d), c in zip(phase_sets.items(), ['k', 'tab:blue']):
    cs = np.cos(d['ph'][:, None] - d['ph'][None, :])[iu]
    bi = np.digitize(dt_pairs[iu], edges) - 1
    prof = np.array([cs[bi == k].mean() if np.any(bi == k) else np.nan for k in range(len(centers))])
    ax.plot(centers, prof, '-o', ms=3, color=c, label=d['label'])
    # expected level if every trial's phase came from the same distribution: R^2
    ax.axhline(d['R'] ** 2, color=c, ls='--', lw=0.8, label=f"R² = {d['R'] ** 2:.2f} ({name})")
    d['stab_prof'] = prof
ax.axvspan(0, baseline_length + epoch_length, color='grey', alpha=0.1)
ax.axhline(0, color='grey', lw=0.8)
ax.set_xlabel('Time between trials (s)')
ax.set_ylabel('mean cos(Δphase)')
ax.set_title(f'{tag}: phase similarity vs trial separation. Level > 0 = phase locked; '
             'decay with separation = drift\n(dashed = R², the flat level for a stable phase; '
             'grey = long-window overlap, trivially high)', fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'phase_stability_vs_lag')

#%% 5. Power and phase locking vs the covariate (trial level)
ok = np.isfinite(cov)
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
for ax, y, lab in zip(axes[:2], [np.log10(band_power), np.log10(snr)],
                      [f'log10 band power {f_band[0]:.2f}-{f_band[1]:.2f} Hz', 'log10 SNR at 1.25 Hz']):
    rho = sp.stats.spearmanr(cov[ok], y[ok])
    ax.scatter(cov[ok], y[ok], s=6, alpha=0.5, c=t_run[ok], cmap='viridis')
    ax.set_xlabel(cov_label)
    ax.set_ylabel(lab)
    ax.set_title(f'Spearman ρ = {rho.statistic:.3f} (trial p = {rho.pvalue:.2g}, not independent)', fontsize=9)
q = np.quantile(cov[ok], np.linspace(0, 1, n_cov_bins + 1))
bins = np.full(n_tr, -1)
bins[ok] = np.clip(np.digitize(cov[ok], q[1:-1]), 0, n_cov_bins - 1)
xb = np.arange(1, n_cov_bins + 1)
for (name, d), c in zip(phase_sets.items(), ['k', 'tab:blue']):
    it = [itpc(raw_ph[name][bins == k]) for k in range(n_cov_bins)]
    # null per bin from the shift surrogate (onset-referenced phases)
    itn = np.array([[itpc(row[bins == k]) for k in range(n_cov_bins)] for row in d['null']])
    axes[2].plot(xb, it, '-o', color=c, label=d['label'])
    axes[2].fill_between(xb, *np.nanpercentile(itn, [2.5, 97.5], axis=0), color=c, alpha=0.12)
    # phase-locked amplitude (long window only has meaningful amplitude in mm)
    if name == 'long':
        Xrel = np.abs(X_long) * np.exp(1j * d['ph'])
        pla = np.real(Xrel * np.exp(-1j * d['mean']))
        axes[3].errorbar(xb, [pla[bins == k].mean() for k in range(n_cov_bins)],
                         [pla[bins == k].std(ddof=1) / np.sqrt(np.sum(bins == k)) for k in range(n_cov_bins)],
                         color='k', marker='o', capsize=3)
        rho_pla = sp.stats.spearmanr(cov[ok], pla[ok]).statistic
for ax in axes[2:]:
    ax.set_xticks(xb)
    ax.set_xlabel(f'{cov_short} quantile bin (1 = {cov_low})')
axes[2].set_ylabel('ITPC at stimulus rate')
axes[2].set_title('ITPC per bin (bands: shift surrogate 95%)', fontsize=9)
axes[2].legend(fontsize=7)
axes[3].set_ylabel('Phase-locked amplitude (mm)')
axes[3].set_title(f'Long window, mean ± SEM; Spearman ρ = {rho_pla:.3f}', fontsize=9)
fig.suptitle(f'{tag}: 1.25 Hz pupil rhythm vs {cov_short} (scatter color = time in run)')
plt.tight_layout()
save_fig(fig, f'vs_{cov_name}')

#%% 6. Whole-run 1.25 Hz power and the run's behavioral performance (one row per run)
# Welch PSD of the continuous pupil over the task period (first onset to last
# onset + SOA), 16 s Hann segments with 50% overlap (0.0625 Hz resolution).
# Compared across runs (and subjects) in summary_phasic_single_run.ipynb.
task = (t_neon >= all_onsets[0]) & (t_neon <= all_onsets[-1] + run_soa)
nper = int(round(sfreq * (baseline_length + epoch_length)))
f_w, p_w = sp.signal.welch(sp.signal.detrend(pupil_d[task]), fs=sfreq, window='hann',
                           nperseg=nper, noverlap=nper // 2)
dist_w = np.abs(f_w - f_r)
band_w = (f_w >= f_band[0]) & (f_w <= f_band[1])
noise_w = (dist_w >= f_noise_offset[0]) & (dist_w <= f_noise_offset[1])
broad_w = (f_w >= f_hp) & (f_w <= 4)
run_power = {
    'run_power_at_f':  p_w[np.argmin(dist_w)],                        # mm^2/Hz at the run's stimulus rate
    'run_band_power':  p_w[band_w].mean(),                            # mean PSD in f_band
    'run_snr':         p_w[np.argmin(dist_w)] / p_w[noise_w].mean(),  # peak / neighbouring bins
    'run_rel_power':   p_w[band_w].sum() / p_w[broad_w].sum(),        # fraction of 0.5-4 Hz power
}
# behavior (BIDS response_code): city 1 = hit, -1 = miss; mnt 0 = correct withhold, -2 = false alarm
code = events_df['response_code'].values
is_city = (events_df['trial_type'] == 'city').values
is_mnt = (events_df['trial_type'] == 'mnt').values
cc, ci = is_city & (code == 1), is_city & (code != 1)    # city correct / incorrect (omission)
mc, mi = is_mnt & (code == 0), is_mnt & (code != 0)      # mnt correct / incorrect (commission)
rt = events_df['reaction_time'].values.astype(float)
rt_cc = rt[cc]
rt_cc = rt_cc[np.isfinite(rt_cc) & (rt_cc > 0)]
behavior = {
    'n_trials':          len(events_df),
    'n_city':            int(is_city.sum()),
    'n_mnt':             int(is_mnt.sum()),
    'n_city_correct':    int(cc.sum()),
    'n_city_incorrect':  int(ci.sum()),
    'n_mnt_correct':     int(mc.sum()),
    'n_mnt_incorrect':   int(mi.sum()),
    'city_incorrect_rate': ci.sum() / max(is_city.sum(), 1),
    'mnt_incorrect_rate':  mi.sum() / max(is_mnt.sum(), 1),
    'accuracy':          (cc.sum() + mc.sum()) / len(events_df),
    'rt_mean':           rt_cc.mean() if len(rt_cc) else np.nan,           # city-correct RT (s)
    'rt_cv':             rt_cc.std(ddof=1) / rt_cc.mean() if len(rt_cc) > 1 else np.nan,  # SD / mean
}
run_df = pd.DataFrame([{'subj': subj, 'run': run_id, 'soa': run_soa, 'f_r': f_r,
                        **run_power, **behavior}])
print(f"{tag}: whole-run power at {f_r:.3f} Hz = {run_power['run_power_at_f']:.3g} mm²/Hz, "
      f"SNR = {run_power['run_snr']:.2f}; city_correct = {behavior['n_city_correct']}, "
      f"city_incorrect = {behavior['n_city_incorrect']}, mnt_correct = {behavior['n_mnt_correct']}, "
      f"mnt_incorrect = {behavior['n_mnt_incorrect']}")
if output_mode == 'save':
    run_df.to_csv(os.path.join(save_dir, f'{tag}_run_power_behavior.csv'), index=False)

fig, ax = plt.subplots(figsize=(6, 3.5))
fm = (f_w > 0) & (f_w <= 4)
ax.semilogy(f_w[fm], p_w[fm], color='k')
for h in range(1, 4):
    ax.axvline(h * f_r, color='tab:red', ls='--', lw=0.8)
ax.axvspan(*f_band, color='tab:red', alpha=0.1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (mm²/Hz)')
ax.set_title(f"{tag}: whole-run pupil PSD (Welch, 16 s); SNR at {f_r:.2f} Hz = {run_power['run_snr']:.1f}\n"
             f"city correct {behavior['n_city_correct']}, mnt correct {behavior['n_mnt_correct']}, "
             f"mnt incorrect {behavior['n_mnt_incorrect']}", fontsize=9)
plt.tight_layout()
save_fig(fig, 'run_psd')
if output_mode == 'show':
    plt.show()
