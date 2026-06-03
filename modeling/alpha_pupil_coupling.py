#%% Load libraries
import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import mne
import re
from scipy.ndimage import gaussian_filter1d

git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
sys.path.append('/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking')

from params_setting import project_path
try:
    from pupil_labs import neon_recording as nr
except ImportError:
    nr = None
from utils_eyetracking import preprocess_pupil

#%% Parameters
EEG_DERIV_DIR = os.path.join(project_path, 'derivatives', 'eeg')
ALPHA_BAND = (8, 12)          # Hz – alpha band
PUPIL_HF_BAND = (0.2, 1.0)   # Hz – high-frequency pupil component (Montefusco-Siegmund 2022)
POSTERIOR_CH = ['pz', 'oz']   # occipital/parietal channels available in this dataset
PEAK_MIN_DIST_S = 3.0         # seconds – minimum separation between pupil peaks/troughs
EPOCH_TMIN = -1.0             # seconds before peak/trough
EPOCH_TMAX = 1.0              # seconds after peak/trough
XCORR_LAG_S = 1.0             # ±lag shown in cross-correlation plot

#%% Helper functions

def get_eeg_fif(eeg_subj_dir, subj_id, run_id):
    """Find preprocessed EEG fif (task name casing varies across subjects)."""
    for task_str in ['gradCPT', 'GradCPT']:
        path = os.path.join(eeg_subj_dir,
            f'sub-{subj_id}_task-{task_str}_run-{run_id:02d}_preproc_eeg.fif')
        if os.path.isfile(path):
            return path
    return None


def get_physio_file(subj_nirs, subj_id, run_id):
    """Return the most recent eyetracking physio TSV, checking multiple naming conventions."""
    candidates = [
        f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260423.tsv',
        f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260311_correct_idx.tsv',
        f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv',
    ]
    for fname in candidates:
        path = os.path.join(subj_nirs, fname)
        if os.path.isfile(path):
            return path
    return None


def compute_eeg_nirs_offset(eeg_ev_file, nirs_ev_file):
    """Return t_offset such that nirs_time = eeg_time + t_offset.

    Computed by matching first stimulus onset in both event tables.
    """
    eeg_ev = pd.read_csv(eeg_ev_file, sep='\t')
    nirs_ev = pd.read_csv(nirs_ev_file, sep='\t')
    return nirs_ev['onset'].values[0] - eeg_ev['onset'].values[0]


def alpha_envelope(raw, channels, band=ALPHA_BAND):
    """Extract mean alpha-band amplitude envelope across posterior channels."""
    avail = [c for c in channels if c in raw.ch_names]
    if not avail:
        return None, None
    r = raw.copy().pick(avail).filter(band[0], band[1], method='fir', verbose=False)
    env = np.abs(sp.signal.hilbert(r.get_data(), axis=1))  # (n_ch, n_times)
    return raw.times, np.mean(env, axis=0)


def bandpass_pupil_hf(t, signal, band=PUPIL_HF_BAND):
    """Bandpass-filter pupil signal to the high-frequency band."""
    fs = 1.0 / np.median(np.diff(t))
    sos = sp.signal.butter(4, band, btype='bandpass', fs=fs, output='sos')
    return sp.signal.sosfiltfilt(sos, signal)


def find_peaks_troughs(signal, t, min_dist_s=PEAK_MIN_DIST_S):
    """Return indices of peaks and troughs with minimum temporal separation."""
    fs = 1.0 / np.median(np.diff(t))
    min_dist = max(1, int(min_dist_s * fs))
    peak_idx, _ = sp.signal.find_peaks(signal, distance=min_dist)
    trough_idx, _ = sp.signal.find_peaks(-signal, distance=min_dist)
    return peak_idx, trough_idx


def epoch_signal(signal, t, event_times, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX):
    """Epoch `signal` around each event time. Normalises each epoch to its own mean."""
    dt = np.median(np.diff(t))
    n_samples = int(round((tmax - tmin) / dt))
    t_ep = np.linspace(tmin, tmax, n_samples)
    i_tmin = int(round(tmin / dt))

    epochs = []
    for t_ev in event_times:
        i_center = int(np.argmin(np.abs(t - t_ev)))
        i_start = i_center + i_tmin
        i_end = i_start + n_samples
        if i_start < 0 or i_end > len(signal):
            continue
        ep = signal[i_start:i_end].copy()
        ep = ep - np.mean(ep)          # zero-mean normalisation (whole-epoch baseline)
        epochs.append(ep)
    return t_ep, np.array(epochs) if epochs else (t_ep, np.empty((0, n_samples)))


def cross_correlate(a, b, t, max_lag_s=XCORR_LAG_S):
    """Pearson cross-correlation between a and b at integer-sample lags."""
    n = len(a)
    a_z = (a - np.mean(a)) / (np.std(a) + 1e-30)
    b_z = (b - np.mean(b)) / (np.std(b) + 1e-30)
    xcorr = np.correlate(a_z, b_z, mode='full') / n
    dt = np.median(np.diff(t))
    lags_s = np.arange(-(n - 1), n) * dt
    mask = np.abs(lags_s) <= max_lag_s
    return lags_s[mask], xcorr[mask]


#%% Main: load data and compute per-run coupling
# subjects that have both preprocessed EEG derivatives and eyetracking physio
SUBJECTS = [670, 673, 719, 721, 723, 726, 727, 730, 733, 746, 751]

run_results = {}

for subj_id in SUBJECTS:
    subj = f'sub-{subj_id}'
    eeg_subj_dir = os.path.join(EEG_DERIV_DIR, subj)
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')

    snirf_files = sorted([f for f in os.listdir(subj_nirs) if f.endswith('.snirf')])
    neon_dirs = []
    if os.path.isdir(subj_neon_dir):
        neon_dirs = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])

    for run_id in range(1, 4):
        fif_file   = get_eeg_fif(eeg_subj_dir, subj_id, run_id)
        physio_file = get_physio_file(subj_nirs, subj_id, run_id)
        eeg_ev_file  = os.path.join(eeg_subj_dir, f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv')
        nirs_ev_file = os.path.join(subj_nirs,    f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_events.tsv')

        missing = [f for f in [fif_file, physio_file, eeg_ev_file, nirs_ev_file] if not f or not os.path.isfile(f)]
        if missing:
            continue

        print(f'Processing {subj} run-{run_id:02d} ...')

        # ── EEG: alpha envelope ────────────────────────────────────────────
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        t_eeg, alpha_env = alpha_envelope(raw, POSTERIOR_CH, ALPHA_BAND)
        if t_eeg is None:
            print(f'  No posterior channels found, skipping.')
            continue

        # ── Pupil: load, blink-remove, detrend, downsample ────────────────
        neon_data = pd.read_csv(physio_file, sep='\t')
        snirf_name = f'sub-{subj_id}_task-gradCPT_run-{run_id:02d}_nirs.snirf'
        neon_idx = snirf_files.index(snirf_name) if snirf_name in snirf_files else run_id - 1
        rec = None
        if nr is not None and neon_dirs and neon_idx < len(neon_dirs):
            try:
                rec = nr.open(os.path.join(subj_neon_dir, neon_dirs[neon_idx]))
            except Exception:
                pass
        events_df = pd.read_csv(nirs_ev_file, sep='\t')
        t_pupil_nirs, pupil_d = preprocess_pupil(
            neon_data, rec=rec, detrend_order=2,
            is_rm_phasic=False, events_df=events_df
        )
        if t_pupil_nirs is None:
            print(f'  Too much missing pupil data, skipping.')
            continue

        # ── Align timelines: convert pupil NIRS time → EEG time ───────────
        t_offset = compute_eeg_nirs_offset(eeg_ev_file, nirs_ev_file)
        t_pupil = t_pupil_nirs - t_offset

        # Restrict to the shared EEG time range
        mask = (t_pupil >= t_eeg[0]) & (t_pupil <= t_eeg[-1])
        t_pupil = t_pupil[mask]
        pupil_d = pupil_d[mask]
        if len(t_pupil) < 200:
            continue

        # ── High-frequency pupil (0.2–1 Hz) ───────────────────────────────
        pupil_hf = bandpass_pupil_hf(t_pupil, pupil_d, PUPIL_HF_BAND)

        # ── Interpolate alpha envelope onto pupil time grid ────────────────
        alpha_pupil_t = np.interp(t_pupil, t_eeg, alpha_env)

        # ── Pearson correlation (whole run) ───────────────────────────────
        r, p = sp.stats.pearsonr(alpha_pupil_t, pupil_hf)

        # ── Pupil peaks / troughs ─────────────────────────────────────────
        peak_idx, trough_idx = find_peaks_troughs(pupil_hf, t_pupil, PEAK_MIN_DIST_S)
        t_peaks   = t_pupil[peak_idx]
        t_troughs = t_pupil[trough_idx]

        # ── Epoch alpha around peaks and troughs ──────────────────────────
        t_ep, alpha_ep_peak   = epoch_signal(alpha_pupil_t, t_pupil, t_peaks,   EPOCH_TMIN, EPOCH_TMAX)
        _,    alpha_ep_trough = epoch_signal(alpha_pupil_t, t_pupil, t_troughs, EPOCH_TMIN, EPOCH_TMAX)

        # ── Cross-correlation ─────────────────────────────────────────────
        lags, xcorr = cross_correlate(alpha_pupil_t, pupil_hf, t_pupil, XCORR_LAG_S)

        run_results[(subj, f'run-{run_id:02d}')] = dict(
            subj=subj, run=f'run-{run_id:02d}',
            r=r, p=p,
            t_ep=t_ep,
            alpha_ep_peak=alpha_ep_peak,
            alpha_ep_trough=alpha_ep_trough,
            lags=lags,
            xcorr=xcorr,
            n_peaks=len(t_peaks),
            n_troughs=len(t_troughs),
            t_pupil=t_pupil,
            pupil_d=pupil_d,
            pupil_hf=pupil_hf,
            alpha_pupil_t=alpha_pupil_t,
            nirs_ev_file=nirs_ev_file,
        )
        print(f'  r={r:.3f} p={p:.2e}  n_peaks={len(t_peaks)}  n_troughs={len(t_troughs)}')

print(f'\nTotal runs processed: {len(run_results)}')


#%% Aggregate across subjects: one value per subject (mean across runs)
from collections import defaultdict

subj_r = defaultdict(list)
for (subj, run), res in run_results.items():
    subj_r[subj].append(res['r'])

subj_mean_r = {s: np.mean(v) for s, v in subj_r.items()}
all_r = list(subj_mean_r.values())
t_stat, p_group = sp.stats.ttest_1samp(all_r, 0)
print(f'\nGroup-level alpha–pupil correlation: mean r={np.mean(all_r):.3f}, '
      f't={t_stat:.2f}, p={p_group:.4f}, N={len(all_r)} subjects')


#%% Plot 1: per-subject correlation bar chart
fig, ax = plt.subplots(figsize=(8, 4))
subj_labels = list(subj_mean_r.keys())
bars = ax.bar(subj_labels, [subj_mean_r[s] for s in subj_labels],
              color=['steelblue' if v >= 0 else 'crimson' for v in subj_mean_r.values()])
ax.axhline(0, color='k', linewidth=0.8)
ax.axhline(np.mean(all_r), color='darkorange', linestyle='--', linewidth=1.5,
           label=f'Group mean r={np.mean(all_r):.3f}')
ax.set_ylabel('Pearson r (alpha–pupil HF)')
ax.set_title(f'Alpha–pupil coupling per subject  (p={p_group:.4f})')
ax.set_xticklabels(subj_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


#%% Plot 2: alpha power epoch (mean ± SEM) around pupil peaks and troughs
# aggregate across all runs / subjects
all_ep_peak, all_ep_trough = [], []
for res in run_results.values():
    if len(res['alpha_ep_peak']) > 0:
        all_ep_peak.append(np.mean(res['alpha_ep_peak'], axis=0))
    if len(res['alpha_ep_trough']) > 0:
        all_ep_trough.append(np.mean(res['alpha_ep_trough'], axis=0))

if all_ep_peak and all_ep_trough:
    t_ep_ref = next(iter(run_results.values()))['t_ep']
    n = min(len(all_ep_peak), len(all_ep_trough))
    t_ep_ref = t_ep_ref[:min(len(t_ep_ref), all_ep_peak[0].shape[0])]

    arr_peak   = np.array(all_ep_peak)[:, :len(t_ep_ref)]
    arr_trough = np.array(all_ep_trough)[:, :len(t_ep_ref)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for arr, label, color in [(arr_peak, 'Pupil peak', 'steelblue'),
                               (arr_trough, 'Pupil trough', 'crimson')]:
        mean = arr.mean(axis=0)
        sem  = arr.std(axis=0) / np.sqrt(len(arr))
        ax.plot(t_ep_ref, mean, color=color, label=f'{label} (N={len(arr)})')
        ax.fill_between(t_ep_ref, mean - sem, mean + sem, alpha=0.25, color=color)
    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event (peak/trough)')
    ax.set_xlabel('Time relative to pupil event (s)')
    ax.set_ylabel('Alpha envelope (normalised)')
    ax.set_title('Alpha power around pupil peaks and troughs (cross-subject mean ± SEM)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


#%% Plot 3: mean cross-correlation (alpha vs pupil HF) across runs / subjects
all_xcorr, all_lags = [], None
for res in run_results.values():
    all_xcorr.append(res['xcorr'])
    if all_lags is None:
        all_lags = res['lags']

if all_xcorr and all_lags is not None:
    n_trim = min(len(x) for x in all_xcorr)
    arr_xc = np.array([x[:n_trim] for x in all_xcorr])
    lags_trim = all_lags[:n_trim]
    mean_xc = arr_xc.mean(axis=0)
    sem_xc  = arr_xc.std(axis=0) / np.sqrt(len(arr_xc))

    peak_lag = lags_trim[np.argmax(mean_xc)]
    print(f'\nCross-correlation peak lag: {peak_lag*1000:.0f} ms '
          f'(positive = alpha leads pupil)')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags_trim * 1000, mean_xc, color='steelblue')
    ax.fill_between(lags_trim * 1000, mean_xc - sem_xc, mean_xc + sem_xc,
                    alpha=0.25, color='steelblue', label='SEM')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(peak_lag * 1000, color='darkorange', linestyle='--', linewidth=1.5,
               label=f'Peak lag = {peak_lag*1000:.0f} ms')
    ax.set_xlabel('Lag (ms)  [positive = alpha leads pupil]')
    ax.set_ylabel('Correlation (r)')
    ax.set_title(f'Cross-correlation: alpha power vs high-frequency pupil  (N={len(arr_xc)} runs)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%% Fig. 1 (D)
"""
Mean amplitude of the high-frequency component of the pupil dynamic versus mean alpha power and the corresponding linear fit (red) averaged across all subjects
"""
# Epoch both signals into 1 s windows with 25% overlap (Montefusco-Siegmund 2022 methods)
EP_LEN_S  = 1.0    # epoch duration (s)
EP_STEP_S = 0.75   # step = 75% of epoch → 25% overlap
N_DECILES  = 10

all_decile_pupil = []   # one (N_DECILES,) array per run
all_decile_alpha  = []

for res in run_results.values():
    t_p   = res['t_pupil']
    p_hf  = res.get('pupil_hf') if res.get('pupil_hf') is not None \
            else bandpass_pupil_hf(t_p, res['pupil_d'], PUPIL_HF_BAND)
    a_env = res.get('alpha_pupil_t')
    if a_env is None:
        continue  # re-run the main loop to populate alpha_pupil_t

    dt      = float(np.median(np.diff(t_p)))
    ep_len  = int(round(EP_LEN_S  / dt))
    ep_step = int(round(EP_STEP_S / dt))

    p_means, a_means = [], []
    i = 0
    while i * ep_step + ep_len <= len(t_p):
        s = i * ep_step
        e = s + ep_len
        p_means.append(np.mean(p_hf[s:e]))
        a_means.append(np.mean(a_env[s:e]))
        i += 1

    if len(p_means) < N_DECILES:
        continue

    p_means = np.array(p_means)
    a_means = np.array(a_means)

    # assign each epoch to one of N_DECILES bins by pupil amplitude
    decile_edges = np.percentile(p_means, np.linspace(0, 100, N_DECILES + 1))
    d_pupil = np.full(N_DECILES, np.nan)
    d_alpha  = np.full(N_DECILES, np.nan)
    for d in range(N_DECILES):
        lo, hi = decile_edges[d], decile_edges[d + 1]
        mask = (p_means >= lo) & (p_means <= hi) if d == N_DECILES - 1 \
               else (p_means >= lo) & (p_means < hi)
        if mask.sum() > 0:
            d_pupil[d] = np.mean(p_means[mask])
            d_alpha[d]  = np.mean(a_means[mask])

    all_decile_pupil.append(d_pupil)
    all_decile_alpha.append(d_alpha)

# group mean across runs (NaN-safe)
mean_pupil = np.nanmean(all_decile_pupil, axis=0)
mean_alpha  = np.nanmean(all_decile_alpha, axis=0)

# linear fit and R²
slope, intercept, r_val, p_val, _ = sp.stats.linregress(mean_pupil, mean_alpha)
fit_x = np.linspace(mean_pupil.min(), mean_pupil.max(), 200)
fit_y = slope * fit_x + intercept

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(mean_pupil, mean_alpha, color='k', s=50, zorder=3)
ax.plot(fit_x, fit_y, color='red', linewidth=2,
        label=f'R²={r_val**2:.4f}  p={p_val:.3f}')
ax.set_xlabel('Mean pupil diameter (norm.)')
ax.set_ylabel('Alpha amplitude (a.u.)')
ax.set_title('Mean HF pupil amplitude vs mean alpha amplitude\n'
             '(decile means averaged across all runs)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%% Mean pupil diameter in 800 ms post-stimulus window per trial type
# Discovers all subjects with physio + events files — does NOT require EEG derivatives.
PUPIL_WINDOW_S = 0.8   # seconds after stimulus onset

CONDITIONS = {
    'mnt_correct':   lambda df: (df['trial_type'] == 'mnt')  & (df['response_code'] == 0),
    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt')  & (df['response_code'] != 0),
    'city_correct':  lambda df: (df['trial_type'] == 'city') & (df['response_code'] != 0),
    'city_incorrect':lambda df: (df['trial_type'] == 'city') & (df['response_code'] == 0),
}

# discover all subjects that have at least one gradCPT physio + events pair
all_subjects = sorted([d for d in os.listdir(project_path) if d.startswith('sub-')])

from collections import defaultdict
subj_pupil_mean = {cond: defaultdict(list) for cond in CONDITIONS}

for subj in all_subjects:
    subj_id  = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    if not os.path.isdir(subj_nirs):
        continue

    neon_dirs_s = []
    if os.path.isdir(subj_neon_dir):
        neon_dirs_s = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])

    for run_id in range(1, 4):
        physio_file  = get_physio_file(subj_nirs, subj_id, run_id)
        nirs_ev_file = os.path.join(subj_nirs,
                           f'{subj}_task-gradCPT_run-{run_id:02d}_events.tsv')
        if not physio_file or not os.path.isfile(nirs_ev_file):
            continue

        neon_data = pd.read_csv(physio_file, sep='\t')
        events_df = pd.read_csv(nirs_ev_file, sep='\t')

        rec = None
        if nr is not None and neon_dirs_s and run_id - 1 < len(neon_dirs_s):
            try:
                rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_s[run_id - 1]))
            except Exception:
                pass

        # pupil and timestamps stay in NIRS time — same reference as events onsets
        t_p, pupil_d_s = preprocess_pupil(
            neon_data, rec=rec, detrend_order=2,
            is_rm_phasic=False, events_df=events_df
        )
        if t_p is None:
            continue
        t_p = np.array(t_p)

        for cond, selector in CONDITIONS.items():
            mask_cond = selector(events_df).values
            for onset in events_df['onset'].values[mask_cond]:
                win_mask = (t_p >= onset) & (t_p < onset + PUPIL_WINDOW_S)
                if win_mask.sum() < 3:
                    continue
                subj_pupil_mean[cond][subj].append(np.mean(pupil_d_s[win_mask]))

# Average trials within each subject
subj_cond_mean = {
    cond: {s: np.mean(vals) for s, vals in subj_dict.items()}
    for cond, subj_dict in subj_pupil_mean.items()
}

# Print group summary
print(f'\nMean pupil diameter in first {int(PUPIL_WINDOW_S*1000)} ms after stimulus onset:')
for cond, sdict in subj_cond_mean.items():
    vals = list(sdict.values())
    print(f'  {cond:20s}: mean={np.mean(vals):.4f}  SD={np.std(vals):.4f}  N={len(vals)}')

# Plot: bar chart with individual subject points
cond_labels = ['mnt_correct', 'mnt_incorrect', 'city_correct', 'city_incorrect']
cond_colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
cond_display = ['mnt\nCorrect', 'mnt\nIncorrect', 'city\nCorrect', 'city\nIncorrect']

fig, ax = plt.subplots(figsize=(7, 4))
for i, (cond, color, label) in enumerate(zip(cond_labels, cond_colors, cond_display)):
    vals = list(subj_cond_mean[cond].values())
    if not vals:
        continue
    mean_v = np.mean(vals)
    sem_v  = np.std(vals) / np.sqrt(len(vals))
    ax.bar(i, mean_v, color=color, alpha=0.7, yerr=sem_v, capsize=4,
           error_kw=dict(linewidth=1.5))
    ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.15, 0.15, len(vals)),
               vals, color=color, s=30, zorder=3, alpha=0.8)

ax.set_xticks(range(len(cond_labels)))
ax.set_xticklabels(cond_display, fontsize=9)
ax.set_ylabel('Mean pupil diameter (mm)')
ax.set_title(f'Mean pupil diameter: first {int(PUPIL_WINDOW_S*1000)} ms after stimulus onset')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


#%% Resting-state baseline: mean pupil diameter per subject
def get_rs_physio_file(subj_nirs, subj_id):
    """Find the most recent resting-state eyetracking physio TSV."""
    candidates = [
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio_20260423.tsv',
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio_20260311_correct_idx.tsv',
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio.tsv',
    ]
    for fname in candidates:
        path = os.path.join(subj_nirs, fname)
        if os.path.isfile(path):
            return path
    return None


baseline_pupil = {}   # subj → mean RS pupil diameter

# same dynamic discovery — any subject with an RS physio file
for subj in all_subjects:
    subj_id  = subj.replace('sub-', '')
    subj_nirs = os.path.join(project_path, subj, 'nirs')
    rs_file = get_rs_physio_file(subj_nirs, subj_id)
    if rs_file is None:
        print(f'{subj}: no RS physio file, skipping baseline')
        continue

    neon_data_rs = pd.read_csv(rs_file, sep='\t')
    t_rs, pupil_rs = preprocess_pupil(
        neon_data_rs, rec=None, detrend_order=2,
        is_rm_phasic=False, events_df=None
    )
    if t_rs is None:
        print(f'{subj}: RS pupil preprocessing failed')
        continue

    baseline_pupil[subj] = np.mean(pupil_rs)
    print(f'{subj}: RS baseline pupil = {baseline_pupil[subj]:.4f} mm')

print(f'\nBaseline available for {len(baseline_pupil)} subjects: {list(baseline_pupil.keys())}')


#%% Stats: paired t-test (condition − baseline) with FDR correction
# restrict to subjects that have both a baseline and task data for that condition
print(f'\n{"Condition":<20}  {"mean diff":>10}  {"SD":>8}  {"N":>4}  {"t":>7}  {"p_raw":>10}  {"p_fdr":>10}  sig')

stat_rows = []
for cond in cond_labels:
    subjs_both = [s for s in subj_cond_mean[cond] if s in baseline_pupil]
    if len(subjs_both) < 3:
        continue
    task_vals = np.array([subj_cond_mean[cond][s] for s in subjs_both])
    base_vals = np.array([baseline_pupil[s]        for s in subjs_both])
    diff      = task_vals - base_vals
    t_val, p_val = sp.stats.ttest_1samp(diff, 0)
    stat_rows.append(dict(cond=cond, diff=diff, mean_diff=diff.mean(),
                          sd=diff.std(), n=len(diff), t=t_val, p=p_val,
                          subjs=subjs_both))

# FDR correction across conditions
p_raw = np.array([r['p'] for r in stat_rows])
_, p_fdr = mne.stats.fdr_correction(p_raw, alpha=0.05)
for row, p_f in zip(stat_rows, p_fdr):
    row['p_fdr'] = p_f
    sig = '*' if p_f < 0.05 else ('~' if row['p'] < 0.05 else '')
    print(f"{row['cond']:<20}  {row['mean_diff']:>10.4f}  {row['sd']:>8.4f}"
          f"  {row['n']:>4}  {row['t']:>7.3f}  {row['p']:>10.4f}  {p_f:>10.4f}  {sig}")


#%% Plot: condition vs baseline (difference plot with SEM and significance)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ── Left panel: raw pupil values (task + baseline) ────────────────────────
ax = axes[0]
x_task = np.arange(len(cond_labels))
legend_handles = []
for i, (cond, color, label) in enumerate(zip(cond_labels, cond_colors, cond_display)):
    subjs_both = [s for s in subj_cond_mean[cond] if s in baseline_pupil]
    if not subjs_both:
        continue
    n_subj = len(subjs_both)
    task_vals = np.array([subj_cond_mean[cond][s] for s in subjs_both])
    base_vals = np.array([baseline_pupil[s]        for s in subjs_both])
    # draw paired lines
    for tv, bv in zip(task_vals, base_vals):
        ax.plot([i - 0.15, i + 0.15], [bv, tv], color='gray', linewidth=0.6,
                alpha=0.5, zorder=1)
    ax.scatter(np.full(n_subj, i - 0.15), base_vals,
               color='gray', s=25, zorder=2, alpha=0.7)
    h = ax.scatter(np.full(n_subj, i + 0.15), task_vals,
                   color=color, s=25, zorder=2, alpha=0.9,
                   label=f'{label.replace(chr(10), " ")} (N={n_subj})')
    legend_handles.append(h)
    # condition mean + SEM
    ax.errorbar(i + 0.15, task_vals.mean(),
                yerr=task_vals.std() / np.sqrt(n_subj),
                fmt='s', color=color, markersize=7, capsize=4, zorder=3)
    ax.errorbar(i - 0.15, base_vals.mean(),
                yerr=base_vals.std() / np.sqrt(n_subj),
                fmt='s', color='gray', markersize=7, capsize=4, zorder=3)

# add a single grey entry for the shared baseline
import matplotlib.lines as mlines
baseline_handle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=5, label=f'RS baseline (N={len(baseline_pupil)})')
ax.legend(handles=legend_handles + [baseline_handle], fontsize=7, loc='best')
ax.set_xticks(x_task)
ax.set_xticklabels(cond_display, fontsize=9)
ax.set_ylabel('Mean pupil diameter (mm)')
ax.set_title('Task (colour) vs RS baseline (grey)')
ax.grid(axis='y', alpha=0.3)

# ── Right panel: difference (condition − baseline) ────────────────────────
ax = axes[1]
for i, row in enumerate(stat_rows):
    cond_i   = cond_labels.index(row['cond'])
    color    = cond_colors[cond_i]
    label    = cond_display[cond_i]
    diff     = row['diff']
    mean_d   = row['mean_diff']
    sem_d    = row['sd'] / np.sqrt(row['n'])
    ax.bar(i, mean_d, color=color, alpha=0.7, yerr=sem_d, capsize=4,
           error_kw=dict(linewidth=1.5))
    ax.scatter(np.full(len(diff), i) + np.random.uniform(-0.15, 0.15, len(diff)),
               diff, color=color, s=28, zorder=3, alpha=0.85)
    # significance annotation
    sig_str = ('*' if row['p_fdr'] < 0.05
               else ('~' if row['p'] < 0.05 else 'ns'))
    ax.text(i, mean_d + sem_d + 0.003, sig_str,
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(range(len(stat_rows)))
ax.set_xticklabels(
    [f"{cond_display[cond_labels.index(r['cond'])]}\n(N={r['n']})" for r in stat_rows],
    fontsize=9)
ax.set_ylabel('Pupil difference (task − baseline, mm)')
ax.set_title(f'Pupil vs RS baseline  (* FDR<0.05, ~ uncorr<0.05)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% pupil diameter and alpha power coupling in Resting state
"""
Reproduction of Montefusco-Siegmund et al. (2022) Figure 1D:
  "Mean amplitude of the high-frequency component of the pupil dynamic versus
   mean alpha power and the corresponding linear fit (red) averaged across all subjects."

Pipeline
--------
1. Discover subjects
   - Find all subjects that have both:
       (a) a preprocessed resting-state EEG file
           sub-<id>_task-Rest_run-01_preproc_eeg.fif  (in derivatives/eeg/)
       (b) a resting-state eyetracking physio file
           sub-<id>_task-RS_run-01_recording-eyetracking_physio_20260423.tsv  (in <subj>/nirs/)

2. Load preprocessed EEG
   - Read the fif file with mne.io.read_raw_fif (preload=True).

3. Extract alpha envelope  (Pz, Oz channels; editable via RS_POSTERIOR_CH)
   - Bandpass filter EEG at 8–12 Hz (FIR).
   - Apply the Hilbert transform → instantaneous amplitude envelope.
   - Average envelope across selected channels.

4. Load and preprocess pupil data
   - Load the BIDS eyetracking physio TSV.
   - Open the matching Neon recording directory (sorted snirf index → sorted neon dir index)
     to obtain blink timestamps for interpolation.
   - Call preprocess_pupil():
       • Interpolate blink periods (linear interpolation over ±16 ms windows).
       • Polynomial detrend (order controlled by detrend_order parameter).
       • Lowpass filter and downsample to 60 Hz.
   - Z-score pupil diameter within subject (zero mean, unit variance).

5. Align EEG and pupil timelines
   - No task events exist for the resting state; both modalities start simultaneously.
   - Shift pupil timestamps so t = 0 matches the start of the recording
     (t_pupil = t_pupil_nirs − min(t_pupil_nirs)).
   - Restrict to the EEG time range.

6. Compute per-epoch features  (window = 1 s, step = 250 ms)
   - Bandpass-filter pupil at 0.2–1 Hz (high-frequency component; Butterworth order 4).
   - Epoch both the alpha envelope and the HF pupil signal using the same sliding window.
   - Per epoch: compute mean alpha amplitude and mean HF pupil amplitude.
   - Reject epochs whose mean alpha amplitude > 1e-4 (artifact threshold).

7. Plot Fig 1D — one subplot per subject
   - Each dot represents one epoch (x = mean HF pupil amplitude, y = mean alpha amplitude).
   - Overlay a red linear fit (scipy.stats.linregress); report R² and p in the subplot title.
   - x-axis fixed to [−1.5, 1.5] (z-scored pupil units) for comparability across subjects.

Differences from Montefusco-Siegmund et al. (2022)
----------------------------------------------------
- EEG channels: the original study used 8 posterior channels (P3, Pz, PO3, O1, Oz, O2,
  PO4, P4) from a 32-electrode BioSemi ActiveTwo system. This dataset has only 2 posterior
  channels available (Pz, Oz), so the alpha envelope is averaged over those two only.

- Recording duration: the original study limited recordings to 60 s to avoid hippus. This
  dataset's resting-state sessions are ~6 minutes long. The longer duration provides more
  epochs but may introduce slow pupil oscillations (hippus) that were intentionally excluded
  in the original study.

- Visualization: the original Fig 1D shows a single scatter plot of 10 decile means averaged
  across all 16 subjects with one red linear fit. Here each subject is shown in a separate
  subplot with epoch-level dots, making individual variability visible rather than collapsing
  it into a group summary.

- Pupil normalization: the original study normalized pupil diameter by referencing to the
  whole recording mean (dividing by the session mean). Here a z-score (zero mean, unit
  variance) is applied within each subject, which additionally accounts for between-subject
  differences in pupil diameter variance.

- Blink interpolation: the original study used a MATLAB smooth function (robust local
  regression, 0.1% window) after linear interpolation. Here blink periods are detected from
  the Pupil Labs Neon blink timestamps and linearly interpolated; a Butterworth lowpass
  filter is then applied at 30 Hz rather than MATLAB's robust regression smoother.
"""

# ── RS-specific parameters ─────────────────────────────────────────────────
RS_POSTERIOR_CH = ['pz', 'oz']   # editable channel selection
RS_EP_LEN_S     = 1.0            # epoch window (s)
RS_EP_STEP_S    = 0.75           # step between epochs (s)  — 250 ms per pipeline spec
RS_N_DECILES    = 10


def get_rest_eeg_fif(eeg_subj_dir, subj_id):
    """Find RS preprocessed EEG, tolerating naming variations across subjects."""
    for fname in [
        f'sub-{subj_id}_task-Rest_run-01_preproc_eeg.fif',
        f'sub-{subj_id}_task_Rest_run-01_preproc_eeg.fif',   # sub-751
    ]:
        path = os.path.join(eeg_subj_dir, fname)
        if os.path.isfile(path):
            return path
    return None


def get_rs_physio_file(subj_nirs, subj_id):
    """Return the RS eyetracking physio TSV (most-recent version first)."""
    candidates = [
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio_20260423.tsv',
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio_20260311_correct_idx.tsv',
        f'sub-{subj_id}_task-RS_run-01_recording-eyetracking_physio.tsv',
    ]
    for fname in candidates:
        path = os.path.join(subj_nirs, fname)
        if os.path.isfile(path):
            return path
    return None


# ── Step 1: discover subjects with both RS EEG and RS physio ──────────────
all_subj_dirs = sorted([d for d in os.listdir(project_path) if d.startswith('sub-')])

rs_subj_data = []   # list of dicts: {subj, p_means, a_means}

for subj in all_subj_dirs:
    subj_id  = subj.replace('sub-', '')
    eeg_dir  = os.path.join(EEG_DERIV_DIR, subj)
    nirs_dir = os.path.join(project_path, subj, 'nirs')

    # Step 1: locate files
    fif_file    = get_rest_eeg_fif(eeg_dir, subj_id)
    physio_file = get_rs_physio_file(nirs_dir, subj_id)
    if not fif_file or not physio_file:
        continue

    physio_tmp = pd.read_csv(physio_file, sep='\t')
    duration_min_physio = (physio_tmp['timestamps'].max() - physio_tmp['timestamps'].min()) / 60
    print(f'Processing {subj} RS ...  duration={duration_min_physio:.2f} min')

    # Step 2: load EEG and find trigger onset in EEG time
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

    eeg_trigger_t = None
    if 'Trigger' in raw.ch_names:
        trig_data = raw.copy().pick('Trigger').get_data()[0]
        thresh    = trig_data.max() / 2
        crossings = np.where((trig_data[:-1] < thresh) & (trig_data[1:] >= thresh))[0]
        if len(crossings):
            eeg_trigger_t = float(raw.times[crossings[0]])
            print(f'  EEG trigger: {len(crossings)} crossing(s), '
                  f'first at t={eeg_trigger_t:.3f} s (EEG time)')
        else:
            print(f'  EEG trigger: Trigger channel present but no crossing found')
    else:
        print(f'  EEG trigger: no Trigger channel in raw')

    # Step 5.1–5.4: alpha envelope on posterior channels via Hilbert transform
    t_eeg, alpha_env = alpha_envelope(raw, RS_POSTERIOR_CH, ALPHA_BAND)
    if t_eeg is None:
        print(f'  No posterior channels found, skipping.')
        continue

    # Step 3: load and preprocess pupil
    neon_data = pd.read_csv(physio_file, sep='\t')

    # locate the matching Neon recording directory for blink removal
    subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    neon_dirs_rs  = sorted([d for d in os.listdir(subj_neon_dir)
                            if re.match(r'\d{4}-', d)]) if os.path.isdir(subj_neon_dir) else []
    snirf_files_rs = sorted([f for f in os.listdir(nirs_dir) if f.endswith('.snirf')])
    rs_snirf_name  = f'sub-{subj_id}_task-RS_run-01_nirs.snirf'
    neon_idx_rs    = snirf_files_rs.index(rs_snirf_name) if rs_snirf_name in snirf_files_rs else 0

    # load RS snirf and check for trigger in the digital aux channel
    rs_snirf_path = os.path.join(nirs_dir, rs_snirf_name)
    rs_onset_nirs = None
    if os.path.isfile(rs_snirf_path):
        import h5py
        with h5py.File(rs_snirf_path, 'r') as sf:
            try:
                digital = sf['nirs/aux1/dataTimeSeries'][()].flatten()
                t_aux   = sf['nirs/aux1/time'][()]
                onsets  = np.where(np.diff(digital) > 0)[0]
                if len(onsets):
                    rs_onset_nirs = float(t_aux[onsets[0]])
                    print(f'  RS trigger: {len(onsets)} onset(s) found, '
                          f'first at {rs_onset_nirs:.3f} s (NIRS time)')
                else:
                    print(f'  RS trigger: digital channel present but no onset found')
            except KeyError:
                print(f'  RS trigger: no digital aux channel in snirf')

    rec = None
    if nr is not None and neon_dirs_rs and neon_idx_rs < len(neon_dirs_rs):
        try:
            rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_rs[neon_idx_rs]))
        except Exception:
            pass

    t_pupil_nirs, pupil_d = preprocess_pupil(
        neon_data, rec=rec, detrend_order=None,
        is_rm_phasic=False, events_df=None
    )
    if t_pupil_nirs is None:
        print(f'  Too much missing pupil data, skipping.')
        continue

    # z-score pupil diameter within subject
    pupil_d = (pupil_d - np.nanmean(pupil_d)) / (np.nanstd(pupil_d) + 1e-30)

    # Step 4: align using trigger — skip if either trigger is missing
    if rs_onset_nirs is None or eeg_trigger_t is None:
        print(f'  Missing trigger (NIRS={rs_onset_nirs}, EEG={eeg_trigger_t}), skipping.')
        continue

    # t_pupil_eeg = t_pupil_nirs - rs_onset_nirs + eeg_trigger_t
    t_pupil = np.array(t_pupil_nirs) - rs_onset_nirs + eeg_trigger_t

    mask = (t_pupil >= t_eeg[0]) & (t_pupil <= t_eeg[-1])
    t_pupil = t_pupil[mask]
    pupil_d = pupil_d[mask]
    if len(t_pupil) < 200:
        print(f'  Too little overlapping data, skipping.')
        continue

    # Step 5.5: high-frequency pupil component (0.2–1 Hz)
    pupil_hf = bandpass_pupil_hf(t_pupil, pupil_d, PUPIL_HF_BAND)

    # Interpolate alpha envelope onto pupil time grid
    alpha_on_pupil = np.interp(t_pupil, t_eeg, alpha_env)

    # Step 5.3–5.5: epoch both signals (1 s window, 250 ms step)
    dt      = float(np.median(np.diff(t_pupil)))
    ep_len  = int(round(RS_EP_LEN_S  / dt))
    ep_step = int(round(RS_EP_STEP_S / dt))

    p_means, a_means, ep_t_centers = [], [], []
    i = 0
    while i * ep_step + ep_len <= len(t_pupil):
        s = i * ep_step
        e = s + ep_len
        t_center = t_pupil[s + ep_len // 2]
        p_means.append(np.mean(pupil_hf[s:e]))
        a_means.append(np.mean(alpha_on_pupil[s:e]))
        ep_t_centers.append(t_center)
        i += 1

    p_means     = np.array(p_means)
    a_means     = np.array(a_means)
    ep_t_centers = np.array(ep_t_centers)

    # discard epochs in the first and last 30 s of the session
    edge_mask = (ep_t_centers >= 30.0) & (ep_t_centers <= t_pupil[-1] - 30.0)
    p_means   = p_means[edge_mask]
    a_means   = a_means[edge_mask]
    print(f'  {(~edge_mask).sum()} edge epochs discarded (first/last 30 s)')

    # reject epochs with mean alpha power > 1e-4 (likely artifacts)
    keep = a_means <= 1e-4
    n_rejected = (~keep).sum()
    p_means = p_means[keep]
    a_means = a_means[keep]

    if len(p_means) < 2:
        print(f'  Too few epochs after rejection ({len(p_means)}), skipping.')
        continue

    # use actual aligned/clipped duration rather than raw physio length
    duration_min_actual = (t_pupil[-1] - t_pupil[0]) / 60

    rs_subj_data.append(dict(
        subj=subj,
        p_means=p_means,
        a_means=a_means,
        duration_min=duration_min_actual,
        t_pupil=t_pupil,
        pupil_hf=pupil_hf,
    ))
    print(f'  {len(p_means)} epochs kept, {n_rejected} rejected')

# ── Step 6: Fig 1D — one subplot per subject ─────────────────────────────
if not rs_subj_data:
    print('No RS subjects could be processed.')
else:
    n_subj = len(rs_subj_data)
    n_cols = int(np.ceil(np.sqrt(n_subj)))
    n_rows = int(np.ceil(n_subj / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, res in enumerate(rs_subj_data):
        ax = axes[idx // n_cols][idx % n_cols]
        p  = res['p_means']
        a  = res['a_means']

        slope, intercept, r_val, p_val, _ = sp.stats.linregress(p, a)
        fit_x = np.linspace(p.min(), p.max(), 200)

        ax.scatter(p, a, color='k', s=8, alpha=0.5, zorder=2)
        ax.plot(fit_x, slope * fit_x + intercept, color='red', linewidth=1.5, zorder=3)
        ax.set_title(f"{res['subj']}  (duration_actual={res['duration_min']:.1f} min)\n"
                     f"R²={r_val**2:.3f}  p={p_val:.3f}", fontsize=9)
        ax.set_xlabel('Mean pupil HF amp.', fontsize=8)
        ax.set_ylabel('Mean alpha amp.', fontsize=8)
        ax.set_xlim(-1.5, 1.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for idx in range(n_subj, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle('Fig 1D — HF pupil amplitude vs alpha amplitude per epoch\n'
                 f'Resting State  (N={n_subj} subjects, 1 dot = 1 epoch)',
                 fontsize=11)
    plt.tight_layout()
    plt.show()

# ── HF pupil time series — 3×3 grid ──────────────────────────────────────
if rs_subj_data:
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), squeeze=False)

    for idx in range(9):
        ax = axes[idx // 3][idx % 3]
        if idx < len(rs_subj_data):
            res  = rs_subj_data[idx]
            t_s  = res['t_pupil']
            mask = (t_s >= 30.0) & (t_s <= t_s[-1] - 30.0)
            pupil_smooth = gaussian_filter1d(res['pupil_hf'], sigma=200)
            ax.plot(t_s[mask]/60, pupil_smooth[mask], color='steelblue', linewidth=0.6)
            ax.set_title(f"{res['subj']}  (duration_actual={res['duration_min']:.1f} min)", fontsize=9)
            ax.set_xlabel('Time (min)', fontsize=8)
            ax.set_ylabel('Pupil HF (z)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    fig.suptitle('High-frequency pupil signal (0.2–1 Hz, z-scored) — Resting State',
                 fontsize=11)
    plt.tight_layout()
    plt.show()
