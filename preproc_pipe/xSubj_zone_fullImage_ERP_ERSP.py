"""
Cross-subject (and single-subject) in-the-zone vs out-of-the-zone ERP / ERSP.

Time-lock = stimulus onset + full_image_delay.
  full_image_delay = 0   -> transition point (100% Nth-1 / 0% Nth), as in Solis-Vivanco/Lavie 2024
  full_image_delay = 0.4 -> image Nth fully present (Yamashita t_n)

Reproduces:
  - Solis-Vivanco, ..., Lavie (bioRxiv 2024) Fig. 2A: TF power difference around the
    transition point, plus theta / beta band time courses (in vs out).
  - Yamashita TF figure: in-zone minus out-of-zone log power, -0.8 to 0.8 s,
    with significant clusters outlined.

ERSP: mne tfr_array_multitaper with the same adaptive window as Lavie
(2 cycles per frequency, dT = 2/f) and time_bandwidth = 2 -> a single DPSS taper
(close to their single Hanning taper). Epochs are padded so every window lies inside
real data; artifact rejection is only applied inside the analysis window.

Zone split: per subject, in-zone = smoothed VTC < the subject's median VTC
(median over all trials in all runs).

--preset lavie: reproduce Solis-Vivanco/Lavie 2024 as closely as our data allow
  - zone: trial-wise |z RT| within run (Go trials), in <= 1 SD, out > 1 SD
  - re-reference to the average of all EEG channels (here fz/cz/pz/oz only)
  - 1.6 s segments (-0.8..0.8 s), no padding, mean voltage removed over the whole segment
  - Hanning-tapered FFT (FieldTrip mtmconvol style), window dT = 2/f, 3-40 Hz, 1 Hz steps;
    time points whose window does not fit inside the segment are NaN (edge cone)
  - power log-transformed per trial, then averaged across trials (mean of log)
  - theta 4-7 Hz, beta 15-30 Hz; figures show out - in (their sign)

Results: xSubj_results/zone_ERP_ERSP_lock<delay>ms<variant>/
  group figures in the root, single-subject figures in single_subj/,
  per-subject results (pickles) in subj_cache/
"""
#%% load library
import os
import gzip
import pickle
import time
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_multitaper
from mne.stats import permutation_cluster_1samp_test
from tqdm import tqdm
from utils import data_save_path, get_eeg_preproc_dir, tsv_to_events

#%% parameters
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--zone', default='vtc_median', choices=['vtc_median', 'lavie'],
                 help="vtc_median: in = smoothed VTC < subject median (all trial types). "
                      "lavie: trial-wise |z RT| (z within run over Go trials with a response); "
                      "in = |z|<=1, out = |z|>1. Only Go (city) trials can be labelled.")
_ap.add_argument('--norm', default='none', choices=['none', 'broadband'],
                 help="broadband: subtract the mean log10 power over the whole TF map "
                      "(3-40 Hz x analysis window), separately per condition / trial, "
                      "to remove a broadband power offset before comparing zones.")
_ap.add_argument('--preset', default='none', choices=['none', 'lavie'],
                 help="lavie: zone=lavie, average reference, Hanning FFT (2/f), 1.6 s segments "
                      "with segment demean, mean of per-trial log power, plotted as out - in.")
_args, _ = _ap.parse_known_args()
preset = _args.preset
zone_method = 'lavie' if preset == 'lavie' else _args.zone
norm_method = 'none' if preset == 'lavie' else _args.norm
tf_method   = 'hanning' if preset == 'lavie' else 'multitaper'
eeg_ref     = 'average' if preset == 'lavie' else 'mastoid'     # mastoid = as preprocessed
trial_avg   = 'meanlog' if preset == 'lavie' else 'logmean'     # group input: mean(log P) vs log(mean P)
diff_sign   = -1 if preset == 'lavie' else 1                    # plot out - in (Lavie) or in - out
diff_txt    = 'out-of-zone $-$ in-zone' if diff_sign < 0 else 'in-zone $-$ out-of-zone'

is_save_fig = True
if is_save_fig:
    matplotlib.use('Agg')

ch_names = ['fz', 'cz', 'pz', 'oz']
zone_events = ['city_correct', 'mnt_correct']   # Lavie used Go (city) trials
if zone_method == 'lavie':
    zone_events = ['city_correct']              # NoGo trials have no RT -> no trial-wise label
full_image_delay = 0.0                          # time-lock = onset + delay (0 = transition point)
onset_to_full = 0.4                             # image fully present 400 ms after onset (for plot markers)
analysis_window = (-0.8, 0.8)                   # relative to time-lock
epoch_reject_crit = dict(eeg=100e-6)            # applied only inside analysis_window
min_trials_per_zone = 10                        # per subject / channel / zone

# ERSP: window length = n_cycles / f, same as Solis-Vivanco/Lavie 2024 (dT = 2/f;
# 667 ms @ 3 Hz, 200 ms @ 10 Hz, 50 ms @ 40 Hz).
tf_freqs = np.arange(3, 41, 1.0)                # 3-40 Hz in 1 Hz steps, as in the paper
tf_n_cycles = 2.0
tf_time_bandwidth = 2.0                         # -> single DPSS taper; spectral bandwidth = 2/window = f
tf_decim = 5                                    # 500 Hz -> 100 Hz TF time resolution
win_len = tf_n_cycles / tf_freqs                # seconds, per frequency
pad = float(np.ceil(win_len.max() / 2 * 10) / 10)  # half of the longest window, rounded up
if preset == 'lavie':
    pad = 0.0                                   # Lavie: 1.6 s segments, edge windows -> NaN

bands = {'theta': (4, 7), 'beta': (15, 30)} if preset == 'lavie' else \
        {'theta': (4, 8), 'beta': (15, 30)}
alpha = 0.05
n_permutations = 2000

ep_tmin = analysis_window[0] - pad
ep_tmax = analysis_window[1] + pad

lock_tag = f"lock{full_image_delay * 1000:.0f}ms"
tf_tag = f"cyc{tf_n_cycles:g}_TW{tf_time_bandwidth:g}" if tf_method == 'multitaper' else \
         f"hann{tf_n_cycles:g}cyc"
variant_tag = ''
if preset != 'none':
    variant_tag += f'_preset-{preset}'
elif zone_method != 'vtc_median':
    variant_tag += f'_zone-{zone_method}'
if norm_method != 'none':
    variant_tag += f'_norm-{norm_method}'
results_dir = os.path.join(data_save_path, "xSubj_results", f"zone_ERP_ERSP_{lock_tag}{variant_tag}")
cache_dir   = os.path.join(results_dir, "subj_cache")
single_dir  = os.path.join(results_dir, "single_subj")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(single_dir, exist_ok=True)
is_overwrite_cache = False

# plot markers (relative to time-lock)
t_onset = -full_image_delay
t_full  = onset_to_full - full_image_delay
time_label = ('Time from stimulus onset (transition point) (s)' if full_image_delay == 0
              else f'Time from onset + {full_image_delay * 1000:.0f} ms (s)')
marker_txt = 'dotted: stimulus onset, dashed: image fully present'

norm_label = '' if norm_method == 'none' else ' (broadband-normalised)'
tf_desc = (f'multitaper {tf_n_cycles:g} cycles, TW={tf_time_bandwidth:g}' if tf_method == 'multitaper'
           else f'Hanning FFT, {tf_n_cycles:g} cycles (dT=2/f), mean of log power')
print(f"TF window length: {win_len.max():.2f} s @ {tf_freqs[0]:.0f} Hz -> "
      f"{win_len.min():.3f} s @ {tf_freqs[-1]:.0f} Hz; padding = {pad:.1f} s; {lock_tag}; "
      f"zone={zone_method}; norm={norm_method}; tf={tf_method}; ref={eeg_ref}; avg={trial_avg}")
print("Results ->", results_dir)


def norm_offset(logpow):
    """Broadband offset to subtract: mean log10 power over (freq, time), per leading index."""
    if norm_method == 'broadband':
        return logpow.mean(axis=(-2, -1), keepdims=True)
    return np.zeros(logpow.shape[:-2] + (1, 1))


def tfr_hanning(data, sfreq, freqs, n_cycles, decim):
    """FieldTrip mtmconvol-style power: one Hanning taper of length n_cycles/f per frequency,
    FFT/convolution, evaluated every `decim` samples. Time points whose window does not lie
    fully inside the segment are NaN.  data: (n_ep, n_t) -> (n_ep, n_freq, ceil(n_t/decim))."""
    from scipy.signal import fftconvolve
    n_ep, n_t = data.shape
    t_idx = np.arange(0, n_t, decim)
    out = np.full((n_ep, len(freqs), len(t_idx)), np.nan)
    for fi, f in enumerate(freqs):
        N = int(round(n_cycles / f * sfreq))
        k = np.arange(1, N + 1)
        taper = 0.5 * (1 - np.cos(2 * np.pi * k / (N + 1)))       # MATLAB hanning(N)
        taper /= np.linalg.norm(taper)
        kern = taper * np.exp(-2j * np.pi * f * np.arange(N) / sfreq)
        y = fftconvolve(data, kern[::-1][None, :], mode='valid')    # y[:, j] = window starting at j
        start = t_idx - N // 2
        ok = (start >= 0) & (start <= n_t - N)
        out[:, fi, ok] = np.abs(y[:, start[ok]]) ** 2
    return out


def nan_safe_t(X):
    """One-sample t over axis 0; 0 where undefined (NaN edge cone)."""
    X = np.nan_to_num(X)
    sd = X.std(0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = X.mean(0) / (sd / np.sqrt(len(X)))
    return np.where(sd > 0, t, 0.0)


def lavie_abs_z(on_events, reaction_time, lookup):
    """Trial-wise |z RT| within a run over Go (city) trials with a response; NaN otherwise."""
    rt = np.asarray(reaction_time, dtype=float)
    is_go = np.isin(on_events[:, 2], [lookup['city_correct'], lookup['city_incorrect']])
    ok = is_go & (rt > 0)
    absz = np.full(len(rt), np.nan)
    absz[ok] = np.abs((rt[ok] - rt[ok].mean()) / rt[ok].std(ddof=1))
    return absz


def save_or_show_fig(fig, save_path):
    if is_save_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def add_markers(ax, scale=1.0):
    for x, ls in ((t_onset, ':'), (t_full, '--')):
        if analysis_window[0] <= x <= analysis_window[1]:
            ax.axvline(x * scale, color='k', ls=ls, lw=1)


def fdr(p):
    return multipletests(np.nan_to_num(p, nan=1.0).ravel(), alpha=alpha,
                         method='fdr_bh')[0].reshape(np.shape(p))


#%% per-subject processing
def epoch_locked(EEG, events, event_id, ev_name, ev_vtc):
    """Epoch around onset + full_image_delay; return epochs and VTC of kept epochs."""
    sfreq = EEG.info['sfreq']
    ev = events.copy()
    ev[:, 0] = ev[:, 0] + int(round(full_image_delay * sfreq))
    first = int(np.floor(ep_tmin * sfreq))
    last  = int(np.ceil(ep_tmax * sfreq))
    valid = (ev[:, 0] + first >= 0) & (ev[:, 0] + last < len(EEG.times))
    ev, ev_vtc = ev[valid], ev_vtc[valid]
    if len(ev) == 0:
        return None, None
    epochs = mne.Epochs(EEG, ev, event_id={ev_name: event_id},
                        tmin=ep_tmin, tmax=ep_tmax, picks='eeg',
                        baseline=(None, None) if preset == 'lavie' else None,   # segment demean
                        reject=epoch_reject_crit,
                        reject_tmin=analysis_window[0], reject_tmax=analysis_window[1],
                        detrend=None if preset == 'lavie' else 1, preload=True, verbose=False)
    keep = np.array([len(x) == 0 for x in epochs.drop_log])
    if keep.sum() == 0:
        return None, None
    return epochs, ev_vtc[keep]


def zone_compare(a, b):
    """Welch t-test across trials (axis 0): returns mean/sem of each and FDR mask."""
    _, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return dict(m_in=a.mean(0), se_in=a.std(0, ddof=1) / np.sqrt(len(a)),
                m_out=b.mean(0), se_out=b.std(0, ddof=1) / np.sqrt(len(b)),
                sig=fdr(p))


def process_subject(key_name):
    cache_path = os.path.join(cache_dir, f"{key_name}_zone_{lock_tag}{variant_tag}_{tf_tag}_dec{tf_decim}.pkl")
    if os.path.exists(cache_path) and not is_overwrite_cache:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    gz_path = os.path.join(data_save_path, key_name, f"{key_name}_preprocessed_dict.pkl.gz")
    with gzip.open(gz_path, 'rb') as f:
        EEG_dict = pickle.load(f)["EEG"]

    # ---- epoch every run, keep VTC of every trial for the zone threshold
    all_vtc = []
    run_epochs = {ev: [] for ev in zone_events}      # list of (epochs, vtc)
    sfreq = None
    for run_name, EEG in EEG_dict.items():
        if 'gradcpt' not in run_name:
            continue
        sfreq = EEG.info['sfreq']
        if eeg_ref == 'average':
            EEG = EEG.copy().set_eeg_reference('average', ch_type='eeg', verbose=False)
        run_id = int(run_name.split('cpt')[-1])
        event_file = os.path.join(get_eeg_preproc_dir(key_name),
                                  f"{key_name}_task-gradCPT_run-{run_id:02d}_events.tsv")
        events, lookup, vtc_list, rt_list = tsv_to_events(event_file, sfreq)
        n_trials = len(events) // 2                   # first half = stimulus onsets
        on_events, on_vtc = events[:n_trials], np.asarray(vtc_list[:n_trials])
        all_vtc.append(on_vtc)
        # per-trial zone score: lower = more in the zone
        score = on_vtc if zone_method == 'vtc_median' else \
                lavie_abs_z(on_events, rt_list[:n_trials], lookup)
        for ev in zone_events:
            m = (on_events[:, 2] == lookup[ev]) & ~np.isnan(score)
            if m.sum() == 0:
                continue
            epochs, score_keep = epoch_locked(EEG, on_events[m], lookup[ev], ev, score[m])
            if epochs is not None:
                run_epochs[ev].append((epochs, score_keep))
    # in-zone: vtc_median -> score < median VTC ; lavie -> |z RT| <= 1
    thres = np.median(np.concatenate(all_vtc)) if zone_method == 'vtc_median' else 1.0
    is_in = (lambda s: s < thres) if zone_method == 'vtc_median' else (lambda s: s <= thres)

    out = dict(thres=thres, times=None, tf_times=None, res={})
    for ev in zone_events:
        out['res'][ev] = {}
        for ch in ch_names:
            parts = [(e.get_data(picks=[ch])[:, 0, :], v) for e, v in run_epochs[ev]
                     if ch in e.ch_names]
            if len(parts) == 0:
                continue
            data = np.concatenate([p[0] for p in parts])          # (n_ep, n_times)
            in_zone = is_in(np.concatenate([p[1] for p in parts]))
            times = run_epochs[ev][0][0].times
            n_in, n_out = int(in_zone.sum()), int((~in_zone).sum())
            r = dict(n_in=n_in, n_out=n_out)
            if min(n_in, n_out) >= min_trials_per_zone:
                awin = (times >= analysis_window[0] - 1e-9) & (times <= analysis_window[1] + 1e-9)
                # ERP
                erp = data[:, awin]
                r['erp_in'], r['erp_out'] = erp[in_zone].mean(0), erp[~in_zone].mean(0)
                r['ss_erp'] = zone_compare(erp[in_zone], erp[~in_zone])
                # single-trial power (n_ep, n_freq, n_tf_times)
                if tf_method == 'hanning':
                    pw = tfr_hanning(data, sfreq, tf_freqs, tf_n_cycles, tf_decim)
                else:
                    pw = tfr_array_multitaper(data[:, None, :], sfreq=sfreq,
                                              freqs=tf_freqs, n_cycles=tf_n_cycles,
                                              time_bandwidth=tf_time_bandwidth,
                                              output='power', decim=tf_decim,
                                              n_jobs=1, verbose=False)[:, 0]
                tf_times = times[::tf_decim]
                tf_awin = (tf_times >= analysis_window[0] - 1e-9) & \
                          (tf_times <= analysis_window[1] + 1e-9)
                pw = pw[..., tf_awin]
                # group-level input: log10 of trial-averaged power (logmean) or trial-average of
                # log10 power (meanlog, Lavie); optionally broadband-normalised per condition
                for zone, mask in (('in', in_zone), ('out', ~in_zone)):
                    lpm = np.log10(pw[mask]).mean(0) if trial_avg == 'meanlog' else \
                          np.log10(pw[mask].mean(0))
                    r[f'logpow_{zone}'] = lpm - norm_offset(lpm)
                # single-subject stats on single-trial log power (normalised per trial)
                lp = np.log10(pw)
                trial_off = norm_offset(lp)                          # (n_ep, 1, 1)
                lp = lp - trial_off
                t_map, p_map = stats.ttest_ind(lp[in_zone], lp[~in_zone], axis=0, equal_var=False)
                r['ss_tf'] = dict(t=t_map, sig=fdr(p_map))
                r['ss_band'] = {}
                for bname, (lo, hi) in bands.items():
                    bm = (tf_freqs >= lo) & (tf_freqs <= hi)
                    bp = (np.log10(pw[:, bm]).mean(1) if trial_avg == 'meanlog' else
                          np.log10(pw[:, bm].mean(1))) - trial_off[:, 0]   # (n_ep, n_t)
                    r['ss_band'][bname] = zone_compare(bp[in_zone], bp[~in_zone])
                out['times'] = times[awin]
                out['tf_times'] = tf_times[tf_awin]
            out['res'][ev][ch] = r

    with open(cache_path, 'wb') as f:
        pickle.dump(out, f)
    return out


#%% single-subject figure: columns = channels, rows = TF diff / theta / beta / ERP
def plot_single_subject(key_name, r_subj, ev):
    chs = [c for c in ch_names if 'erp_in' in r_subj['res'][ev].get(c, {})]
    if len(chs) == 0:
        return
    tt, et = r_subj['tf_times'], r_subj['times'] * 1000
    fig, axes = plt.subplots(4, len(chs), figsize=(4.2 * len(chs), 12), squeeze=False,
                             constrained_layout=True,
                             gridspec_kw={'height_ratios': [2, 1, 1, 1.2]})
    diffs = [diff_sign * (r_subj['res'][ev][c]['logpow_in'] - r_subj['res'][ev][c]['logpow_out'])
             for c in chs]
    vmax = np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 99)
    for j, ch in enumerate(chs):
        cr = r_subj['res'][ev][ch]
        ax = axes[0, j]
        im = ax.pcolormesh(tt, tf_freqs, diffs[j], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           shading='nearest')
        if cr['ss_tf']['sig'].any():
            ax.contour(tt, tf_freqs, cr['ss_tf']['sig'].astype(float), levels=[0.5],
                       colors='k', linewidths=0.8)
        add_markers(ax)
        ax.set_title(f"{ch.upper()}  (in={cr['n_in']}, out={cr['n_out']})", fontsize=10)
        if j == 0:
            ax.set_ylabel('Frequency (Hz)')
        for i, (bname, (lo, hi)) in enumerate(bands.items()):
            axb = axes[1 + i, j]
            b = cr['ss_band'][bname]
            for m, se, c, lab in ((b['m_in'], b['se_in'], 'tab:blue', 'In'),
                                  (b['m_out'], b['se_out'], 'tab:red', 'Out')):
                axb.plot(tt, m, color=c, lw=1.5, label=lab)
                axb.fill_between(tt, m - se, m + se, color=c, alpha=0.25)
            if b['sig'].any():
                yl = axb.get_ylim()
                axb.scatter(tt[b['sig']], np.full(b['sig'].sum(), yl[0] + 0.05 * (yl[1] - yl[0])),
                            marker='s', s=4, color='k')
            add_markers(axb)
            axb.grid(alpha=0.3)
            if j == 0:
                axb.set_ylabel(f'{bname} {lo}-{hi} Hz\n' + r'$\log_{10}$ power')
        axe = axes[3, j]
        e = cr['ss_erp']
        for m, se, c, lab in ((e['m_in'], e['se_in'], 'tab:blue', 'In'),
                              (e['m_out'], e['se_out'], 'tab:red', 'Out')):
            axe.plot(et, m * 1e6, color=c, lw=1.2, label=lab)
            axe.fill_between(et, (m - se) * 1e6, (m + se) * 1e6, color=c, alpha=0.25)
        if e['sig'].any():
            yl = axe.get_ylim()
            axe.scatter(et[e['sig']], np.full(e['sig'].sum(), yl[0] + 0.05 * (yl[1] - yl[0])),
                        marker='s', s=4, color='k')
        add_markers(axe, scale=1000)
        axe.axhline(0, color='gray', lw=0.8)
        axe.grid(alpha=0.3)
        axe.set_xlabel(time_label.replace('(s)', '(ms)'), fontsize=8)
        if j == 0:
            axe.set_ylabel(r'ERP ($\mu$V)')
            axes[1, 0].legend(fontsize=7, loc='upper right')
    fig.colorbar(im, ax=axes[0, :].tolist(), label=r'$\log_{10}$ power ' + diff_txt, shrink=0.9)
    fig.suptitle(f'{key_name} — {ev} | {tf_desc} '
                 f'| zone={zone_method}, norm={norm_method}, ref={eeg_ref} '
                 f'| black: FDR p<{alpha} (Welch t across trials) | {marker_txt}', fontsize=10)
    save_or_show_fig(fig, os.path.join(single_dir, f'{key_name}_{ev}_{tf_tag}.png'))


#%% run all subjects
subj_keys = sorted(d for d in os.listdir(data_save_path)
                   if d.startswith('sub-') and os.path.exists(
                       os.path.join(data_save_path, d, f"{d}_preprocessed_dict.pkl.gz")))
subj_res = {}
t0 = time.time()
for key_name in tqdm(subj_keys):
    try:
        subj_res[key_name] = process_subject(key_name)
    except Exception as err:
        print(f"{key_name}: failed ({err!r}), skipping")
        continue
    for ev in zone_events:
        plot_single_subject(key_name, subj_res[key_name], ev)
print(f"Processed {len(subj_res)} subjects in {time.time() - t0:.0f} s")

times = next(r['times'] for r in subj_res.values() if r['times'] is not None)
tf_times = next(r['tf_times'] for r in subj_res.values() if r['tf_times'] is not None)

#%% trial count table
rows = []
for s, r in subj_res.items():
    for ev in zone_events:
        for ch, cr in r['res'][ev].items():
            rows.append(dict(subject=s, event=ev, ch=ch, n_in=cr['n_in'], n_out=cr['n_out'],
                             included='erp_in' in cr))
count_df = pd.DataFrame(rows)
count_df.to_csv(os.path.join(results_dir, "trial_counts.csv"), index=False)
print(count_df.groupby(['event', 'ch'])[['n_in', 'n_out', 'included']]
      .agg({'n_in': 'median', 'n_out': 'median', 'included': 'sum'}))


#%% gather subject-level arrays
def gather(ev, ch):
    """Return dict of subject-stacked arrays. ch='mean' averages available channels per subject."""
    keys = ['erp_in', 'erp_out', 'logpow_in', 'logpow_out']
    out = {k: [] for k in keys}
    for s, r in subj_res.items():
        chs = ch_names if ch == 'mean' else [ch]
        avail = [c for c in chs if 'erp_in' in r['res'][ev].get(c, {})]
        if len(avail) == 0:
            continue
        for k in keys:
            out[k].append(np.mean([r['res'][ev][c][k] for c in avail], axis=0))
    return {k: np.array(v) for k, v in out.items()}


def plot_ersp(d, ev, ch):
    diff = d['logpow_in'] - d['logpow_out']                     # (n_subj, n_freq, n_time)
    n = diff.shape[0]
    t_thresh = stats.t.ppf(1 - alpha / 2, n - 1)
    T_obs, clusters, cl_p, _ = permutation_cluster_1samp_test(
        np.nan_to_num(diff), threshold=t_thresh, n_permutations=n_permutations, tail=0,
        stat_fun=nan_safe_t, out_type='mask', seed=0, verbose=False)
    sig = np.zeros_like(T_obs, dtype=bool)
    for c, p in zip(clusters, cl_p):
        if p < alpha:
            sig |= c

    mean_diff = diff_sign * diff.mean(0)
    vmax = np.nanpercentile(np.abs(mean_diff), 99)

    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    im = ax.pcolormesh(tf_times, tf_freqs, mean_diff, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='nearest')
    if sig.any():
        ax.contour(tf_times, tf_freqs, sig.astype(float), levels=[0.5],
                   colors='k', linewidths=1)
    for lo, hi in bands.values():
        for f in (lo, hi):
            ax.axhline(f, color='orange', lw=0.8, alpha=0.8)
    add_markers(ax)
    fig.colorbar(im, ax=ax, label=r'$\log_{10}$ power  ' + diff_txt + norm_label)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'{ev} — {ch.upper()} — n={n} | {tf_desc} '
                 f'| zone={zone_method}, norm={norm_method}, ref={eeg_ref}'
                 f'\nblack: cluster p<{alpha} ({marker_txt})',
                 fontsize=10)

    # band time courses (Lavie Fig 2A lower panels)
    for i, (bname, (lo, hi)) in enumerate(bands.items()):
        axb = fig.add_subplot(gs[1 + i], sharex=ax)
        bm = (tf_freqs >= lo) & (tf_freqs <= hi)
        bp_in, bp_out = d['logpow_in'][:, bm].mean(1), d['logpow_out'][:, bm].mean(1)
        for y, c, lab in ((bp_in, 'tab:blue', 'In-the-zone'), (bp_out, 'tab:red', 'Out-of-the-zone')):
            m, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(n)
            axb.plot(tf_times, m, color=c, lw=2, label=lab)
            axb.fill_between(tf_times, m - se, m + se, color=c, alpha=0.25)
        _, p = stats.ttest_rel(bp_in, bp_out, axis=0)
        sig_t = fdr(p)
        if sig_t.any():
            yl = axb.get_ylim()
            axb.scatter(tf_times[sig_t], np.full(sig_t.sum(), yl[0] + 0.05 * (yl[1] - yl[0])),
                        marker='s', s=6, color='k', label=f'FDR p<{alpha}')
        add_markers(axb)
        axb.set_ylabel(rf'{bname} {lo}-{hi} Hz' + '\n' + r'$\log_{10}$ power')
        axb.legend(fontsize=8, loc='upper right')
        axb.grid(alpha=0.3)
    axb.set_xlabel(time_label)
    axb.set_xlim(*analysis_window)
    save_or_show_fig(fig, os.path.join(results_dir, f'ERSP_inzone_minus_outzone_{ev}_{ch}_{tf_tag}.png'))


def plot_erp(d, ev, ch):
    e_in, e_out = d['erp_in'] * 1e6, d['erp_out'] * 1e6
    n = e_in.shape[0]
    t_ms = times * 1000
    _, p = stats.ttest_rel(e_in, e_out, axis=0)
    sig_t = fdr(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for y, c, lab in ((e_in, 'tab:blue', 'In-the-zone'), (e_out, 'tab:red', 'Out-of-the-zone')):
        m, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(n)
        ax1.plot(t_ms, m, color=c, lw=2, label=lab)
        ax1.fill_between(t_ms, m - se, m + se, color=c, alpha=0.25)
    dd = e_in - e_out
    m, se = dd.mean(0), dd.std(0, ddof=1) / np.sqrt(n)
    ax2.plot(t_ms, m, color='purple', lw=2, label='In $-$ Out')
    ax2.fill_between(t_ms, m - se, m + se, color='purple', alpha=0.25)
    if sig_t.any():
        yl = ax2.get_ylim()
        ax2.scatter(t_ms[sig_t], np.full(sig_t.sum(), yl[0] + 0.05 * (yl[1] - yl[0])),
                    marker='s', s=6, color='k', label=f'FDR p<{alpha} (paired t)')
    for a in (ax1, ax2):
        a.axhline(0, color='gray', lw=0.8)
        add_markers(a, scale=1000)
        a.set_ylabel(r'Amplitude ($\mu$V)')
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax1.set_title(f'ERP {ev} — {ch.upper()} — n={n} | zone={zone_method} ({marker_txt})', fontsize=10)
    ax2.set_xlabel(time_label.replace('(s)', '(ms)'))
    ax2.set_xlim(analysis_window[0] * 1000, analysis_window[1] * 1000)
    save_or_show_fig(fig, os.path.join(results_dir, f'ERP_inzone_vs_outzone_{ev}_{ch}.png'))


#%% group plots
for ev in zone_events:
    for ch in ch_names + ['mean']:
        d = gather(ev, ch)
        if len(d['erp_in']) < 3:
            print(f"{ev} {ch}: too few subjects, skip")
            continue
        print(f"{ev} {ch}: n_subj = {len(d['erp_in'])}")
        plot_erp(d, ev, ch)
        plot_ersp(d, ev, ch)

#%% log
with open(os.path.join(results_dir, f"run_info_{tf_tag}.txt"), 'w') as f:
    f.write(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"Subjects processed: {list(subj_res)}\n")
    f.write(f"Time-lock: onset + {full_image_delay} s; analysis window {analysis_window} s; "
            f"padding {pad} s\n")
    f.write(f"Preset: {preset}; TF method: {tf_method}; reference: {eeg_ref}; "
            f"trial averaging: {trial_avg}; plotted sign: {'out - in' if diff_sign < 0 else 'in - out'}\n")
    f.write(f"Bands: {bands}\n")
    f.write(f"TF: freqs {tf_freqs[0]}-{tf_freqs[-1]} Hz, n_cycles={tf_n_cycles}, "
            f"time_bandwidth={tf_time_bandwidth}, decim={tf_decim}\n")
    f.write(f"Zone split: {zone_method} " + ("(in = smoothed VTC < per-subject median VTC)" if zone_method == 'vtc_median'
            else "(trial-wise |z RT| within run; in <= 1 SD, out > 1 SD)") + "\n")
    f.write(f"Normalisation: {norm_method}\n")
    f.write(f"Rejection: {epoch_reject_crit} within analysis window\n")
    f.write("Group stats: cluster permutation (TF), paired t + FDR (bands, ERP)\n")
    f.write("Single-subject stats: Welch t across trials + FDR\n")
print("Done. Results in", results_dir)
