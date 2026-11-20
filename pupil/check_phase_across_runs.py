#%% load library
"""
Why is the 1.25 Hz pupil rhythm's phase (relative to the stimulus) not
consistent across runs of the same subject? (follow-up to investigating_phasic.py)

For one subject:
1. Spaghetti plot of the high-passed pupil around city-correct onsets, one
   subplot per run (3 x 1).
2. Cycle-folded waveform: high-passed pupil averaged over every stimulus
   (city + mnt) within one 0.8 s cycle, all runs overlaid. A timing offset
   between runs shows up as a horizontal shift of this waveform.
3. Phase over time within each run (sliding windows): a constant phase means
   a fixed per-run offset; a steady drift means clock drift.
4. Independent sync check: cross-correlate the Neon IMU acceleration (stored
   in the aligned physio TSV) with the fNIRS accelerometer. If the per-run
   alignment (single head-nod peak, see
   code/eyetracking/neon_to_bids_gradCPT_add_nodding.py) is off, the residual
   lag should match the pupil-phase-implied offset.

Sign convention: a positive lag / offset means the pupil (Neon) signal
appears LATER in fNIRS time than it should.
"""
import numpy as np
import scipy as sp
import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'modeling'))  # params_setting.py
from params_setting import *
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from pupil_labs import neon_recording as nr
from utils_eyetracking import preprocess_pupil
from cedalion import io as cio

#%% parameters
subj            = 'sub-727'
runs            = [1, 2, 3]
physio_version  = 'latest'
baseline_length = 8       # sec before onset
epoch_length    = 8       # sec after onset
f_hp            = 0.5     # Hz, high-pass to isolate the stimulus-rate rhythm
win_phase       = 60      # sec, sliding window for phase over time
step_phase      = 15      # sec
win_xcorr       = 60      # sec, sliding window for accelerometer cross-correlation
max_lag         = 1.5     # sec, lag search range
fs_acc          = 100     # Hz, common grid for accelerometer comparison
save_dir        = os.path.join(project_path, 'derivatives', 'eye_tracking', 'phasic', subj)
os.makedirs(save_dir, exist_ok=True)

#%% functions
def save_fig(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(save_dir, f'{subj}_{name}.{ext}'), dpi=300, bbox_inches='tight')

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

def epoch(onsets, t_cont, x_cont, sfreq, t_pre, t_post):
    n_pre, n_post = int(round(sfreq * t_pre)), int(round(sfreq * t_post))
    idx0 = np.clip(np.searchsorted(t_cont, onsets), 1, len(t_cont) - 1)
    idx0 = np.where(np.abs(t_cont[idx0 - 1] - onsets) < np.abs(t_cont[idx0] - onsets), idx0 - 1, idx0)
    starts = idx0 - n_pre
    valid = (starts >= 0) & (starts + n_pre + n_post <= len(x_cont))
    ii = starts[valid][:, None] + np.arange(n_pre + n_post)[None, :]
    return x_cont[ii], valid

def cycle_coeffs(stim_onsets, soa, t_cont, x_cont):
    """
    Per-stimulus complex amplitude of the stimulus-rate rhythm:
    c_k = mean over [t_k, t_k + SOA) of x(t) * exp(-i*2*pi*(t - t_k)/SOA).
    angle(c_k) = 0 means the pupil peaks at stimulus onset.
    Also returns the cycle-folded waveform on a common phase grid.
    """
    n_bins = 48
    grid = np.linspace(0, soa, n_bins, endpoint=False)
    coeffs, folded = [], []
    for tk in stim_onsets:
        m = (t_cont >= tk) & (t_cont < tk + soa)
        if m.sum() < 10:
            coeffs.append(np.nan + 0j); continue
        tau = t_cont[m] - tk
        coeffs.append(np.mean(x_cont[m] * np.exp(-2j * np.pi * tau / soa)))
        folded.append(np.interp(grid, tau, x_cont[m]))
    return np.array(coeffs), grid, np.array(folded)

def bandpass(x, fs, lo, hi, order=3):
    b, a = sp.signal.butter(order, [lo, hi], btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, x)

def xcorr_lag(a, b, fs, max_lag):
    """Lag (s) maximizing corr(a(t), b(t - lag)); positive = a later than b."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n_lag = int(max_lag * fs)
    lags = np.arange(-n_lag, n_lag + 1)
    cc = np.array([np.mean(a[max(0, L):len(a) + min(0, L)] * b[max(0, -L):len(b) - max(0, L)])
                   for L in lags])
    i = np.argmax(cc)
    return lags[i] / fs, cc[i], lags / fs, cc

#%% load runs
subj_id = subj.replace('sub-', '')
subj_nirs = os.path.join(project_path, subj, 'nirs')
subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
mapping = _parse_eyetracking_readme(os.path.join(project_path, 'sourcedata', 'raw', subj, 'nirs'))

run_data = {}
for run_id in runs:
    stem = f"sub-{subj_id}_task-gradCPT_run-{run_id:02d}"
    physio_file = os.path.join(subj_nirs, f"{stem}_recording-eyetracking_physio_{physio_version}.tsv")
    physio_json = physio_file.replace('.tsv', '.json')
    events_df = pd.read_csv(os.path.join(subj_nirs, f"{stem}_events.tsv"), sep='\t')
    neon_data = pd.read_csv(physio_file, sep='\t')
    rec = nr.open(os.path.join(subj_neon_dir, mapping[('gradcpt', run_id)])) \
        if ('gradcpt', run_id) in mapping else None
    meta = pd.read_json(physio_json, typ='series') if os.path.isfile(physio_json) else {}

    t_neon, pupil_d = preprocess_pupil(neon_data, rec=rec, detrend_order=2,
                                       is_rm_phasic=False, events_df=events_df)
    t_neon, pupil_d = np.asarray(t_neon, float), np.asarray(pupil_d, float)
    sfreq = np.round(np.median(1 / np.diff(t_neon)))
    b, a = sp.signal.butter(3, f_hp, btype='highpass', fs=sfreq)
    pupil_hp = sp.signal.filtfilt(b, a, pupil_d)

    all_onsets = events_df['onset'].values
    soa = np.median(np.diff(all_onsets))
    city_correct = ((events_df['trial_type'] == 'city') & (events_df['response_code'] == 1)).values
    ep, valid = epoch(all_onsets[city_correct], t_neon, pupil_hp, sfreq, baseline_length, epoch_length)
    coeffs, grid, folded = cycle_coeffs(all_onsets, soa, t_neon, pupil_hp)

    run_data[run_id] = {
        'events': events_df, 'neon': neon_data, 'meta': meta, 'soa': soa, 'sfreq': sfreq,
        't': t_neon, 'pupil_hp': pupil_hp, 'epochs': ep, 'all_onsets': all_onsets,
        'coeffs': coeffs, 'grid': grid, 'folded': folded,
        'snirf': os.path.join(subj_nirs, f"{stem}_nirs.snirf"),
    }
    ok = ~np.isnan(coeffs)
    m = np.nanmean(coeffs[ok] / np.abs(coeffs[ok]))
    run_data[run_id]['phase'] = np.angle(m)
    run_data[run_id]['plv'] = np.abs(m)
    print(f"{subj} run-{run_id:02d}: SOA = {soa:.4f} s, {len(ep)} city-correct epochs, "
          f"alignment offset = {meta.get('AlignmentOffset_s', np.nan):.3f} s, "
          f"stimulus-rate phase = {np.degrees(np.angle(m)):.1f} deg, PLV = {np.abs(m):.2f}")

ref_phase = np.angle(np.mean([np.exp(1j * d['phase']) for d in run_data.values()]))
for run_id, d in run_data.items():
    dphi = np.angle(np.exp(1j * (d['phase'] - ref_phase)))
    d['dt_pupil'] = -dphi / (2 * np.pi / d['soa'])
    print(f"run-{run_id:02d}: pupil-phase-implied offset from subject mean = {d['dt_pupil'] * 1000:.0f} ms "
          f"(mod {d['soa'] * 1000:.0f} ms)")

#%% 1. Spaghetti plot: high-passed pupil around city-correct onsets, one run per row
t_ep = (np.arange(run_data[runs[0]]['epochs'].shape[1]) - int(round(run_data[runs[0]]['sfreq'] * baseline_length))) \
       / run_data[runs[0]]['sfreq']
ylim = np.nanpercentile(np.abs(np.vstack([d['epochs'] for d in run_data.values()])), 99.5)
fig, axes = plt.subplots(len(runs), 1, figsize=(12, 3 * len(runs)), sharex=True, sharey=True)
for ax, (run_id, d) in zip(np.atleast_1d(axes), run_data.items()):
    ax.plot(t_ep, d['epochs'].T, color='tab:blue', lw=0.3, alpha=0.05)
    ax.plot(t_ep, d['epochs'].mean(0), color='k', lw=1.5, label=f"mean (n={len(d['epochs'])})")
    for k in np.arange(-np.floor(baseline_length / d['soa']), np.floor(epoch_length / d['soa']) + 1):
        ax.axvline(k * d['soa'], color='tab:red', lw=0.5, alpha=0.5)
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel('High-passed pupil (mm)')
    ax.set_title(f"run-{run_id:02d}: phase = {np.degrees(d['phase']):.0f}°, "
                 f"offset from subject mean = {d['dt_pupil'] * 1000:.0f} ms")
    ax.legend(loc='upper right', fontsize=8)
np.atleast_1d(axes)[-1].set_xlabel('Time from city-correct onset (s)')
fig.suptitle(f'{subj}: high-passed (>{f_hp} Hz) pupil, city-correct trials; red = stimulus onsets')
plt.tight_layout()
save_fig(fig, 'spaghetti_runs')

# Same, zoomed to two stimulus cycles around onset so the ripple's position is visible
fig, axes = plt.subplots(len(runs), 1, figsize=(8, 3 * len(runs)), sharex=True, sharey=True)
zoom = (t_ep >= -0.8) & (t_ep <= 1.6)
ylim_z = np.nanpercentile(np.abs(np.vstack([d['epochs'][:, zoom] for d in run_data.values()])), 99)
for ax, (run_id, d) in zip(np.atleast_1d(axes), run_data.items()):
    ax.plot(t_ep[zoom], d['epochs'][:, zoom].T, color='tab:blue', lw=0.3, alpha=0.08)
    ax.plot(t_ep[zoom], d['epochs'][:, zoom].mean(0), color='k', lw=2)
    for k in (-1, 0, 1, 2):
        ax.axvline(k * d['soa'], color='tab:red', lw=0.8)
    ax.set_ylim(-ylim_z, ylim_z)
    ax.set_ylabel('High-passed pupil (mm)')
    ax.set_title(f"run-{run_id:02d}")
np.atleast_1d(axes)[-1].set_xlabel('Time from city-correct onset (s)')
fig.suptitle(f'{subj}: zoom on -0.8 to 1.6 s')
plt.tight_layout()
save_fig(fig, 'spaghetti_runs_zoom')

#%% 2. Cycle-folded waveform (all stimuli), runs overlaid
fig, ax = plt.subplots(figsize=(7, 4))
colors = plt.get_cmap('viridis')(np.linspace(0, 0.85, len(runs)))
for c, (run_id, d) in zip(colors, run_data.items()):
    m = np.nanmean(d['folded'], 0)
    se = np.nanstd(d['folded'], 0) / np.sqrt(len(d['folded']))
    g = np.r_[d['grid'], d['grid'] + d['soa']] * 1000      # show two cycles
    ax.plot(g, np.r_[m, m], color=c, lw=2, label=f"run-{run_id:02d} ({d['dt_pupil'] * 1000:+.0f} ms)")
    ax.fill_between(g, np.r_[m - se, m - se], np.r_[m + se, m + se], color=c, alpha=0.2)
ax.axvline(0, color='tab:red', lw=0.8); ax.axvline(run_data[runs[0]]['soa'] * 1000, color='tab:red', lw=0.8)
ax.set_xlabel('Time from stimulus onset (ms), two cycles shown')
ax.set_ylabel('High-passed pupil (mm)')
ax.set_title(f'{subj}: pupil folded on the stimulus cycle (mean ± SEM over stimuli)')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'cycle_folded')

#%% 3. Phase over time within each run
fig, ax = plt.subplots(figsize=(10, 4))
for c, (run_id, d) in zip(colors, run_data.items()):
    on, cf = d['all_onsets'], d['coeffs']
    centers = np.arange(on[0] + win_phase / 2, on[-1] - win_phase / 2, step_phase)
    ph = []
    for tc in centers:
        m = (on >= tc - win_phase / 2) & (on < tc + win_phase / 2) & ~np.isnan(cf)
        ph.append(np.angle(np.mean(cf[m] / np.abs(cf[m]))))
    lat = np.mod(-np.unwrap(np.array(ph)) / (2 * np.pi / d['soa']), d['soa']) * 1000
    d['phase_time'] = (centers - on[0], np.array(ph))
    ax.plot(centers - on[0], lat, '-o', ms=3, color=c, label=f'run-{run_id:02d}')
ax.set_xlabel('Time from first stimulus (s)')
ax.set_ylabel('Pupil peak latency within cycle (ms)')
ax.set_ylim(0, run_data[runs[0]]['soa'] * 1000)
ax.set_title(f'{subj}: phase of the stimulus-rate rhythm over time ({win_phase} s windows). '
             'Flat = fixed offset; sloped = clock drift')
ax.legend(fontsize=8)
plt.tight_layout()
save_fig(fig, 'phase_over_time')

#%% 4. Sync check: Neon IMU vs fNIRS accelerometer
# Uses the Neon acceleration columns already placed on fNIRS time in the
# physio TSV and the fNIRS ACCEL_*_1_1 aux channels (as in the alignment script).
fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 7), squeeze=False)
for j, (run_id, d) in enumerate(run_data.items()):
    rec_s = cio.read_snirf(d['snirf'])[0]
    acc = {}
    for ax_name in 'XYZ':
        key = f'ACCEL_{ax_name}_1_1' if f'ACCEL_{ax_name}_1_1' in rec_s.aux_ts else f'ACCEL_{ax_name}'
        acc[ax_name] = rec_s.aux_ts[key].mean(dim='aux_channel').values
    t_s = rec_s.aux_ts[key].time.values.astype(float)
    mag_s = np.sqrt(acc['X'] ** 2 + acc['Y'] ** 2 + acc['Z'] ** 2)
    nd = d['neon'].dropna(subset=['acceleration_x'])
    t_n = nd['timestamps'].values
    mag_n = np.sqrt(nd['acceleration_x'] ** 2 + nd['acceleration_y'] ** 2 + nd['acceleration_z'] ** 2).values

    t0, t1 = max(t_s[0], t_n[0]), min(t_s[-1], t_n[-1])
    grid = np.arange(t0, t1, 1 / fs_acc)
    a_n = bandpass(np.interp(grid, t_n, mag_n), fs_acc, 0.2, 4)
    a_s = bandpass(np.interp(grid, t_s, np.nan_to_num(mag_s, nan=np.nanmean(mag_s))), fs_acc, 0.2, 4)

    lag_all, r_all, lags, cc = xcorr_lag(a_n, a_s, fs_acc, max_lag)
    d['lag_acc'] = lag_all
    axes[0, j].plot(lags * 1000, cc, color='k')
    axes[0, j].axvline(lag_all * 1000, color='tab:red', ls='--')
    axes[0, j].set_title(f'run-{run_id:02d}: whole-run lag = {lag_all * 1000:.0f} ms (r = {r_all:.2f})')
    axes[0, j].set_xlabel('Lag of Neon vs fNIRS accel (ms)')
    axes[0, j].set_ylabel('Correlation')

    centers, lag_w, r_w = [], [], []
    for ws in np.arange(grid[0], grid[-1] - win_xcorr, win_xcorr / 2):
        m = (grid >= ws) & (grid < ws + win_xcorr)
        L, r, _, _ = xcorr_lag(a_n[m], a_s[m], fs_acc, max_lag)
        centers.append(ws + win_xcorr / 2); lag_w.append(L); r_w.append(r)
    sc = axes[1, j].scatter(centers, np.array(lag_w) * 1000, c=r_w, cmap='viridis', vmin=0, vmax=1)
    axes[1, j].axhline(lag_all * 1000, color='tab:red', ls='--', lw=0.8)
    axes[1, j].set_xlabel('fNIRS time (s)')
    axes[1, j].set_ylabel('Window lag (ms)')
    axes[1, j].set_title(f'{win_xcorr} s windows (color = peak r)')
    print(f"run-{run_id:02d}: accel lag (whole run) = {lag_all * 1000:.0f} ms, r = {r_all:.2f}; "
          f"window lags median = {np.median(lag_w) * 1000:.0f} ms, IQR = "
          f"{np.subtract(*np.percentile(lag_w, [75, 25])) * 1000:.0f} ms")
plt.colorbar(sc, ax=axes[1, -1], label='peak r')
fig.suptitle(f'{subj}: residual sync lag between Neon IMU and fNIRS accelerometer '
             '(positive = Neon later)')
plt.tight_layout()
save_fig(fig, 'accel_sync_check')

# Compare: pupil-phase-implied offsets vs accelerometer residual lags (both relative to run mean)
lag_mean = np.mean([d['lag_acc'] for d in run_data.values()])
print('\nrun   pupil offset (ms, mod SOA)   accel lag - mean (ms)')
for run_id, d in run_data.items():
    print(f"{run_id:>3}   {d['dt_pupil'] * 1000:>10.0f}                 {(d['lag_acc'] - lag_mean) * 1000:>10.0f}")
plt.show()
