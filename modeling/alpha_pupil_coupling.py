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

#%% Mean pupil diameter in 800 ms post-stimulus window per trial type
PUPIL_WINDOW_S = 0.8   # seconds after stimulus onset

CONDITIONS = {
    'mnt_correct':   lambda df: (df['trial_type'] == 'mnt')  & (df['response_code'] == 0),
    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt')  & (df['response_code'] != 0),
    'city_correct':  lambda df: (df['trial_type'] == 'city') & (df['response_code'] != 0),
    'city_incorrect':lambda df: (df['trial_type'] == 'city') & (df['response_code'] == 0),
}

# subj_pupil_mean[cond] = list of per-subject mean values (averaged over trials and runs)
from collections import defaultdict
subj_pupil_mean = {cond: defaultdict(list) for cond in CONDITIONS}

for (subj, run), res in run_results.items():
    t_p  = res['t_pupil']
    pd_  = res['pupil_d']
    evdf = pd.read_csv(res['nirs_ev_file'], sep='\t')

    # convert NIRS event onsets → EEG time using the already-known offset
    # (t_pupil is already in EEG time after alignment in the main loop)
    eeg_ev_file = os.path.join(EEG_DERIV_DIR, subj,
        f'{subj}_task-gradCPT_{run}_events.tsv')
    nirs_first  = evdf['onset'].values[0]
    eeg_first   = pd.read_csv(eeg_ev_file, sep='\t')['onset'].values[0]
    t_off       = nirs_first - eeg_first          # nirs_time = eeg_time + t_off
    onsets_eeg  = evdf['onset'].values - t_off    # trial onsets in EEG time

    for cond, selector in CONDITIONS.items():
        mask_cond = selector(evdf).values
        for onset_eeg in onsets_eeg[mask_cond]:
            win_mask = (t_p >= onset_eeg) & (t_p < onset_eeg + PUPIL_WINDOW_S)
            if win_mask.sum() < 3:
                continue
            subj_pupil_mean[cond][subj].append(np.mean(pd_[win_mask]))

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

for subj_id in SUBJECTS:
    subj = f'sub-{subj_id}'
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
