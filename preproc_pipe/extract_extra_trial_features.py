#%% Extra single-trial EEG and eye-tracking features (literature survey, 2026-11-06)
# Features from "EEG & Pupil Features of Sustained Attention -- Literature Survey" that are not in
# extract_EEG_trial_features.py. One row per gradCPT trial, keyed by (run, trial_idx) like the EEG feature
# table, so the two tables join row for row.
#
# Scopes (how EEG_extra_features_vs_VTC.py uses each feature):
#   window  : computed in a window CENTRED on the trial onset (WIN_S = 10 s, i.e. onset +- 5 s; about the
#             7 s FWHM of the VTC smoothing). Tested on all trials.
#   rolling : trial-to-trial statistic over a centred run of trials (ROLL_SD_N = 9 for SDs, ITPC_N = 15 for
#             phase locking, PUPIL_EVOKED_N = 13 for the stimulus-locked pupil response). All trials.
#   trial   : single-trial value. Tested on city_correct trials only, so a button press is present on every
#             trial and press / no-press differences cannot drive the result.
#
# EEG (preprocessed .fif, channels fz cz pz oz, same files as extract_EEG_trial_features.py)
#   window, per channel (Welch, 2 s Hann, 50 % overlap, median average):
#     <ch>_ap_exp, <ch>_ap_off     aperiodic exponent / offset: robust log-log line fit 2-40 Hz (the FOOOF
#                                  'robust_ap_fit': fit, keep points below the fit, refit)
#     <ch>_iaf                     individual alpha frequency: peak of the aperiodic-removed spectrum in 7-14 Hz
#                                  (parabolic interpolation; NaN if no local peak >= 0.05 log10 units)
#     <ch>_alpha_flat, <ch>_theta_flat   periodic power: mean aperiodic-removed log10 power, 7-14 / 4-8 Hz
#     <ch>_medfreq                 spectral median frequency, 1-30 Hz
#     <ch>_tbr, <ch>_tar           log10 theta/beta and theta/alpha absolute-power ratios
#     <ch>_theta_win, <ch>_alpha_win   log10 absolute band power in the same window (reference: band power
#                                  is already in the pipeline, here only re-measured in the centred window)
#     coh_<a>_<b>_<band>           magnitude-squared coherence (theta, alpha) for fz-pz and fz-oz
#   trial / rolling, theta (Morlet 4-8 Hz, n_cycles = f/2, on the continuous run):
#     <ch>_theta_trial             log10 stimulus-locked theta power in [onset, onset + 0.8 s)   (trial)
#     <ch>_theta_trial_sd9         rolling SD of <ch>_theta_trial over 9 trials                   (rolling)
#     <ch>_theta_itpc15            bias-corrected theta inter-trial phase coherence, [0, 0.4] s
#                                  after onset, over 15 trials                                    (rolling)
#   trial, early ERP (oz, pz): <ch>_P1 / <ch>_N1 mean amplitude 80-130 / 140-200 ms, baseline (-0.2, 0)
#   trial, CPP (pz, city_correct): pz_cpp_slope (uV/s, linear fit -250 to -50 ms before the press, 10 Hz
#                                  low-pass), pz_cpp_amp (mean -100 to 0 ms before press, minus the
#                                  pre-onset baseline)
#   trial, wavelet (fz, Sensors 2021): db4 multiresolution analysis at 250 Hz, bands delta (A5, 0-4 Hz),
#     theta (D5), alpha (D4), beta (D3), gamma (D2, 31-62 Hz; the data are low-passed at 45 Hz), in
#     [onset, onset + 0.8 s): fz_dwt_<band>_energy (log10 mean square), _apen (approximate entropy, m = 2,
#     r = 0.2 SD), _tv (mean |diff| / SD), _skew
#   Single-trial EEG values are NaN when the channel's peak-to-peak in the trial window exceeds 100 uV.
#
# Eye tracking (Pupil Labs Neon; BIDS physio .tsv in fNIRS time + raw Neon recording for blinks,
# fixations and saccades; recording matched through the physio .json 'SourceNeonRec')
#   Pupil = mean of both eyes, blinks interpolated, quadratic detrend, 30 Hz low-pass, 60 Hz
#   (utils_eyetracking.preprocess_pupil, as in cmp_EEG_split_by_RT_pupil.py).
#     pupil_base     mean pupil in [-0.2, 0) s (reference: already in the pipeline)               (trial)
#     pupil_deriv    mean pupil derivative in [-0.5, 0) s (reference: already in the pipeline)    (trial)
#     pupil_tepr     task-evoked response: mean [0.6, 1.2] s minus pupil_base                     (trial)
#                    (with 0.8 s trials the next trial's response overlaps; read as a local index)
#     pupil_base_sd9, pupil_tepr_sd9   rolling SD over 9 trials (Unsworth & Robison 2017)       (rolling)
#     pupil_evoked13 peak-to-peak of the trial-locked average of [0, 0.8) s epochs (each demeaned)
#                    over 13 trials: how strongly the pupil follows the 1.25 Hz stimulus rhythm   (rolling)
#     pupil_sd_win   SD of the pupil trace in the window (Zhang et al. 2020)                     (window)
#     blink_rate (/min), blink_dur (s), fix_rate (/min), fix_dur (median, s), sacc_rate (/min),
#     sacc_amp (median deg), sacc_peakvel (median), microsacc_rate (saccades < 1 deg, /min)      (window)
#   Trial pupil features are NaN when a blink overlaps [-0.5, 1.2] s.
#   Caveat: Neon gaze accuracy (~1.5-2 deg) limits microsaccade detection; microsacc_rate counts the
#   small saccades Neon's detector reports, not a dedicated microsaccade algorithm.
#
# Not computable with this montage / design: EEG microstates (4 scalp channels), SSVEP (no flicker),
# MMN (no auditory oddball), CNV (no cue-target interval), fMRI dynamic connectivity.
#
# Output: <data_save_path>/<subject>/features/<subject>_extra_trial_features.csv.gz (+ _params.json)
# Usage:  python extract_extra_trial_features.py [--subjects 670 723] [--overwrite]
#%% libraries
import os
import sys
import io
import glob
import json
import argparse
import contextlib
import warnings
import numpy as np
import pandas as pd
import mne
import pywt
from scipy.signal import butter, sosfiltfilt, welch, coherence, decimate
from scipy.stats import skew
from tqdm import tqdm

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
from utils import data_save_path
from extract_EEG_trial_features import CH_NAMES, RUN_KEYS, get_eeg_fif_path, load_events, get_features_path

PROJECT_PATH = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24'
sys.path.insert(0, os.path.join(PROJECT_PATH, 'code', 'eyetracking'))

#%% parameters
WIN_S = 10.0                # centred window length (s)
MIN_COVER_FRAC = 0.5
WELCH_SEG = 2.0
AP_RANGE = (2.0, 40.0)
ALPHA_RANGE = (7.0, 14.0)
THETA_RANGE = (4.0, 8.0)
BANDS = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}
IAF_MIN_PEAK = 0.05         # log10 units above the aperiodic fit
COH_PAIRS = [('fz', 'pz'), ('fz', 'oz')]
TRIAL_LEN = 0.8             # s, single-trial window after onset
REJECT_PTP_UV = 100.0
ROLL_SD_N = 9
ITPC_N = 15
ITPC_WIN = (0.0, 0.4)
THETA_FREQS = np.arange(4.0, 8.01, 1.0)
P1_WIN, N1_WIN = (0.08, 0.13), (0.14, 0.20)
CPP_SLOPE_WIN, CPP_AMP_WIN = (-0.25, -0.05), (-0.10, 0.0)
CPP_LP_HZ = 10.0
DWT_FS = 250.0
DWT_WAVELET, DWT_LEVEL = 'db4', 5
DWT_BANDS = {'gamma': 1, 'beta': 2, 'alpha': 3, 'theta': 4, 'delta': 5}  # index in pywt.mra output (0 = A5)
APEN_M, APEN_R = 2, 0.2
PUPIL_BASE_WIN, PUPIL_DERIV_WIN, PUPIL_TEPR_WIN = (-0.2, 0.0), (-0.5, 0.0), (0.6, 1.2)
PUPIL_EVOKED_N = 13
MICROSACC_DEG = 1.0
PHYSIO_SUFFIXES = ['_latest', '_20260423', '_20260311_correct_idx', '']


def get_extra_features_path(subject):
    return get_features_path(subject).replace('_EEG_trial_features', '_extra_trial_features')


def feature_info():
    """name -> dict(family, scope, ref, label). ref = the feature type is already in the pipeline."""
    info = {}

    def add(name, family, scope, label, ref=False):
        info[name] = dict(family=family, scope=scope, ref=ref, label=label)
    for ch in CH_NAMES:
        add(f'{ch}_ap_exp', 'eeg_spectral', 'window', f'{ch} aperiodic exponent')
        add(f'{ch}_ap_off', 'eeg_spectral', 'window', f'{ch} aperiodic offset')
        add(f'{ch}_iaf', 'eeg_spectral', 'window', f'{ch} individual alpha freq')
        add(f'{ch}_alpha_flat', 'eeg_spectral', 'window', f'{ch} periodic alpha')
        add(f'{ch}_theta_flat', 'eeg_spectral', 'window', f'{ch} periodic theta')
        add(f'{ch}_medfreq', 'eeg_spectral', 'window', f'{ch} median frequency')
        add(f'{ch}_tbr', 'eeg_spectral', 'window', f'{ch} theta/beta ratio')
        add(f'{ch}_tar', 'eeg_spectral', 'window', f'{ch} theta/alpha ratio')
        add(f'{ch}_theta_win', 'eeg_spectral', 'window', f'{ch} theta power (10 s)', ref=True)
        add(f'{ch}_alpha_win', 'eeg_spectral', 'window', f'{ch} alpha power (10 s)', ref=True)
    for a, b in COH_PAIRS:
        for band in ['theta', 'alpha']:
            add(f'coh_{a}_{b}_{band}', 'eeg_spectral', 'window', f'{a}-{b} {band} coherence')
    for ch in CH_NAMES:
        add(f'{ch}_theta_trial', 'eeg_trial', 'trial', f'{ch} stim-locked theta')
        add(f'{ch}_theta_trial_sd9', 'eeg_trial', 'rolling', f'{ch} theta trial-to-trial SD')
        add(f'{ch}_theta_itpc15', 'eeg_trial', 'rolling', f'{ch} theta ITPC')
    for ch in ['oz', 'pz']:
        add(f'{ch}_P1', 'eeg_trial', 'trial', f'{ch} P1 amplitude')
        add(f'{ch}_N1', 'eeg_trial', 'trial', f'{ch} N1 amplitude')
    add('pz_cpp_slope', 'eeg_trial', 'trial', 'pz CPP build-up slope')
    add('pz_cpp_amp', 'eeg_trial', 'trial', 'pz CPP amplitude')
    for band in DWT_BANDS:
        for stat in ['energy', 'apen', 'tv', 'skew']:
            add(f'fz_dwt_{band}_{stat}', 'eeg_dwt', 'trial', f'fz DWT {band} {stat}')
    add('pupil_base', 'eye', 'trial', 'baseline pupil', ref=True)
    add('pupil_deriv', 'eye', 'trial', 'pupil derivative', ref=True)
    add('pupil_tepr', 'eye', 'trial', 'task-evoked pupil (TEPR)')
    add('pupil_base_sd9', 'eye', 'rolling', 'baseline pupil trial SD')
    add('pupil_tepr_sd9', 'eye', 'rolling', 'TEPR trial SD')
    add('pupil_evoked13', 'eye', 'rolling', 'stim-locked pupil amplitude')
    add('pupil_sd_win', 'eye', 'window', 'pupil-size variation')
    add('blink_rate', 'eye', 'window', 'blink rate')
    add('blink_dur', 'eye', 'window', 'blink duration')
    add('fix_rate', 'eye', 'window', 'fixation rate')
    add('fix_dur', 'eye', 'window', 'fixation duration')
    add('sacc_rate', 'eye', 'window', 'saccade rate')
    add('sacc_amp', 'eye', 'window', 'saccade amplitude')
    add('sacc_peakvel', 'eye', 'window', 'saccade peak velocity')
    add('microsacc_rate', 'eye', 'window', 'microsaccade rate (<1 deg)')
    return info


#%% helpers
def win_bounds(onsets, n_samp, sfreq, half=WIN_S / 2):
    s0 = np.clip(np.round((onsets - half) * sfreq).astype(int), 0, n_samp)
    s1 = np.clip(np.round((onsets + half) * sfreq).astype(int), 0, n_samp)
    return s0, s1


def rolling(values, n, func, min_valid):
    """Centred rolling statistic over trials (NaN-aware)."""
    v = pd.Series(values)
    return v.rolling(n, center=True, min_periods=min_valid).apply(lambda x: func(x[np.isfinite(x)]), raw=True).values


def robust_ap_fit(f, logp):
    """FOOOF-style robust aperiodic fit (fixed mode). Returns offset, exponent."""
    lf = np.log10(f)
    b1, b0 = np.polyfit(lf, logp, 1)
    flat = logp - (b0 + b1 * lf)
    flat[flat < 0] = 0
    keep = flat <= np.percentile(flat, 2.5)
    if keep.sum() < 5:
        keep = np.ones_like(keep)
    b1, b0 = np.polyfit(lf[keep], logp[keep], 1)
    return b0, -b1


def approx_entropy(x, m=APEN_M, r_frac=APEN_R):
    x = np.asarray(x, float)
    r = r_frac * x.std()
    if r == 0:
        return np.nan

    def phi(mm):
        X = np.lib.stride_tricks.sliding_window_view(x, mm)
        D = np.abs(X[:, None, :] - X[None, :, :]).max(-1)
        return np.log((D <= r).mean(1)).mean()
    return phi(m) - phi(m + 1)


def clean_trial_mask(data, onsets, sfreq, pre=0.2, post=TRIAL_LEN):
    """(n_trials, n_ch) True when the channel's peak-to-peak in [onset - pre, onset + post] <= threshold."""
    n_ch, n_samp = data.shape
    ok = np.zeros((len(onsets), n_ch), bool)
    for i, on in enumerate(onsets):
        a, b = int(round((on - pre) * sfreq)), int(round((on + post) * sfreq))
        if a < 0 or b > n_samp:
            continue
        seg = data[:, a:b]
        ok[i] = (seg.max(1) - seg.min(1)) <= REJECT_PTP_UV
    return ok


#%% EEG features
def eeg_window_features(data, chs, sfreq, onsets):
    n_ch, n_samp = data.shape
    nper = int(round(WELCH_SEG * sfreq))
    s0, s1 = win_bounds(onsets, n_samp, sfreq)
    out = {k: np.full(len(onsets), np.nan) for k in feature_info() if feature_info()[k]['family'] == 'eeg_spectral'}
    for i in range(len(onsets)):
        if s1[i] - s0[i] < max(MIN_COVER_FRAC * WIN_S * sfreq, 2 * nper):
            continue
        seg = data[:, s0[i]:s1[i]]
        f, P = welch(seg, fs=sfreq, nperseg=nper, noverlap=nper // 2, window='hann', average='median', axis=-1)
        apm = (f >= AP_RANGE[0]) & (f <= AP_RANGE[1])
        for c, ch in enumerate(chs):
            logp = np.log10(P[c, apm])
            off, exp_ = robust_ap_fit(f[apm], logp)
            flat = logp - (off - exp_ * np.log10(f[apm]))
            fa = f[apm]
            am = (fa >= ALPHA_RANGE[0]) & (fa <= ALPHA_RANGE[1])
            tm = (fa >= THETA_RANGE[0]) & (fa < THETA_RANGE[1])
            out[f'{ch}_ap_exp'][i], out[f'{ch}_ap_off'][i] = exp_, off
            out[f'{ch}_alpha_flat'][i] = flat[am].mean()
            out[f'{ch}_theta_flat'][i] = flat[tm].mean()
            fa_a, fl_a = fa[am], flat[am]
            k = fl_a.argmax()
            if 0 < k < len(fl_a) - 1 and fl_a[k] >= IAF_MIN_PEAK:
                y0, y1, y2 = fl_a[k - 1], fl_a[k], fl_a[k + 1]
                den = y0 - 2 * y1 + y2
                shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
                out[f'{ch}_iaf'][i] = fa_a[k] + shift * (fa_a[1] - fa_a[0])
            tot_m = (f >= 1) & (f < 30)
            cum = np.cumsum(P[c, tot_m])
            out[f'{ch}_medfreq'][i] = f[tot_m][np.searchsorted(cum, cum[-1] / 2)]
            bp = {b: P[c, (f >= lo) & (f < hi)].sum() for b, (lo, hi) in BANDS.items()}
            out[f'{ch}_tbr'][i] = np.log10(bp['theta'] / bp['beta'])
            out[f'{ch}_tar'][i] = np.log10(bp['theta'] / bp['alpha'])
            out[f'{ch}_theta_win'][i] = np.log10(bp['theta'] * (f[1] - f[0]))
            out[f'{ch}_alpha_win'][i] = np.log10(bp['alpha'] * (f[1] - f[0]))
        for a, b in COH_PAIRS:
            if a in chs and b in chs:
                fc, C = coherence(seg[chs.index(a)], seg[chs.index(b)], fs=sfreq, nperseg=nper, noverlap=nper // 2)
                out[f'coh_{a}_{b}_theta'][i] = C[(fc >= 4) & (fc < 8)].mean()
                out[f'coh_{a}_{b}_alpha'][i] = C[(fc >= 8) & (fc < 13)].mean()
    return out


def eeg_trial_features(data, chs, sfreq, onsets, rt, is_cc):
    """Stimulus-locked theta (power, rolling SD, ITPC), P1/N1, CPP, wavelet features."""
    n_ch, n_samp = data.shape
    n_tr = len(onsets)
    out = {}
    ok = clean_trial_mask(data, onsets, sfreq)
    on_s = np.round(onsets * sfreq).astype(int)
    L = int(round(TRIAL_LEN * sfreq))

    # --- theta: Morlet on the continuous run ---
    W = mne.time_frequency.tfr_array_morlet(data[None], sfreq, THETA_FREQS, n_cycles=THETA_FREQS / 2,
                                            output='complex', verbose=False)[0]  # (ch, f, t)
    it0, it1 = int(round(ITPC_WIN[0] * sfreq)), int(round(ITPC_WIN[1] * sfreq))
    it_idx = np.arange(it0, it1 + 1, max(1, int(round(0.01 * sfreq))))
    for c, ch in enumerate(chs):
        pw = np.full(n_tr, np.nan)
        ph = np.full((n_tr, len(THETA_FREQS), len(it_idx)), np.nan + 0j)
        for i in range(n_tr):
            if not ok[i, c] or on_s[i] + L > n_samp:
                continue
            seg = W[c, :, on_s[i]:on_s[i] + L]
            pw[i] = np.log10((np.abs(seg) ** 2).mean(1)).mean()
            z = W[c][:, on_s[i] + it_idx]  # (f, t)
            ph[i] = z / np.abs(z)
        out[f'{ch}_theta_trial'] = pw
        out[f'{ch}_theta_trial_sd9'] = rolling(pw, ROLL_SD_N, lambda x: x.std(ddof=1), ROLL_SD_N - 3)
        itpc = np.full(n_tr, np.nan)
        h = ITPC_N // 2
        valid = np.isfinite(ph[:, 0, 0].real)
        for i in range(n_tr):
            sl = slice(max(0, i - h), min(n_tr, i + h + 1))
            v = valid[sl]
            N = v.sum()
            if N < ITPC_N - 5:
                continue
            R2 = np.abs(ph[sl][v].mean(0)) ** 2
            itpc[i] = ((N * R2 - 1) / (N - 1)).mean()  # bias-corrected squared ITC, averaged over f and t
        out[f'{ch}_theta_itpc15'] = itpc

    # --- early ERP (P1 / N1), baseline (-0.2, 0) ---
    b0 = int(round(0.2 * sfreq))
    for ch in ['oz', 'pz']:
        if ch not in chs:
            continue
        c = chs.index(ch)
        for name, (lo, hi) in [('P1', P1_WIN), ('N1', N1_WIN)]:
            v = np.full(n_tr, np.nan)
            a, b = int(round(lo * sfreq)), int(round(hi * sfreq))
            for i in range(n_tr):
                if ok[i, c] and on_s[i] - b0 >= 0 and on_s[i] + b <= n_samp:
                    v[i] = data[c, on_s[i] + a:on_s[i] + b].mean() - data[c, on_s[i] - b0:on_s[i]].mean()
            out[f'{ch}_{name}'] = v

    # --- CPP at pz, city_correct trials, response-locked ---
    slope = np.full(n_tr, np.nan)
    amp = np.full(n_tr, np.nan)
    if 'pz' in chs:
        c = chs.index('pz')
        sos = butter(4, CPP_LP_HZ, btype='low', fs=sfreq, output='sos')
        x = sosfiltfilt(sos, data[c])
        for i in np.where(is_cc & np.isfinite(rt))[0]:
            r = onsets[i] + rt[i]
            a, b = int(round((onsets[i] - 0.2) * sfreq)), int(round((r + 0.2) * sfreq))
            s_a, s_b = int(round((r + CPP_SLOPE_WIN[0]) * sfreq)), int(round((r + CPP_SLOPE_WIN[1]) * sfreq))
            if a < 0 or b > n_samp or min(a, s_a) < 0:
                continue
            raw_seg = data[c, a:b]
            if raw_seg.max() - raw_seg.min() > REJECT_PTP_UV:
                continue
            tt = np.arange(s_a, s_b) / sfreq
            slope[i] = np.polyfit(tt, x[s_a:s_b], 1)[0]
            m_a, m_b = int(round((r + CPP_AMP_WIN[0]) * sfreq)), int(round((r + CPP_AMP_WIN[1]) * sfreq))
            amp[i] = x[m_a:m_b].mean() - x[a:a + int(round(0.2 * sfreq))].mean()
    out['pz_cpp_slope'], out['pz_cpp_amp'] = slope, amp

    # --- wavelet band features at fz ---
    for band in DWT_BANDS:
        for stat in ['energy', 'apen', 'tv', 'skew']:
            out[f'fz_dwt_{band}_{stat}'] = np.full(n_tr, np.nan)
    if 'fz' in chs:
        c = chs.index('fz')
        q = int(round(sfreq / DWT_FS))
        x = decimate(data[c], q, zero_phase=True) if q > 1 else data[c]
        fs2 = sfreq / q
        comps = pywt.mra(x, DWT_WAVELET, level=DWT_LEVEL, transform='dwt')  # [A5, D5, D4, D3, D2, D1]
        L2 = int(round(TRIAL_LEN * fs2))
        for band, k in DWT_BANDS.items():
            sig = comps[k]
            for i in range(n_tr):
                s = int(round(onsets[i] * fs2))
                if not ok[i, c] or s + L2 > len(sig):
                    continue
                seg = sig[s:s + L2]
                sd = seg.std()
                out[f'fz_dwt_{band}_energy'][i] = np.log10(np.mean(seg ** 2))
                out[f'fz_dwt_{band}_apen'][i] = approx_entropy(seg)
                out[f'fz_dwt_{band}_tv'][i] = np.abs(np.diff(seg)).mean() / sd if sd > 0 else np.nan
                out[f'fz_dwt_{band}_skew'][i] = skew(seg)
    return out


#%% eye-tracking features
def find_physio(subject, run_id):
    for suf in PHYSIO_SUFFIXES:
        p = os.path.join(PROJECT_PATH, subject, 'nirs',
                         f'{subject}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio{suf}.tsv')
        if os.path.isfile(p):
            return p
    return None


def open_neon(subject, physio_path):
    js = physio_path.replace('.tsv', '.json')
    if not os.path.isfile(js):
        return None
    rec_name = json.load(open(js)).get('SourceNeonRec')
    d = os.path.join(PROJECT_PATH, 'sourcedata', 'raw', subject, 'eye_tracking', str(rec_name))
    if not rec_name or not os.path.isdir(d):
        return None
    from pupil_labs import neon_recording as nr
    return nr.open(d)


def eye_features(subject, run_id, ev_fnirs):
    """Eye-tracking features for one run; ev_fnirs = the run's fNIRS events.tsv (fNIRS time)."""
    from utils_eyetracking import preprocess_pupil
    n_tr = len(ev_fnirs)
    names = [k for k, v in feature_info().items() if v['family'] == 'eye']
    out = {k: np.full(n_tr, np.nan) for k in names}
    physio = find_physio(subject, run_id)
    if physio is None:
        return out, 'no physio file'
    neon = pd.read_csv(physio, sep='\t')
    try:
        rec = open_neon(subject, physio)
    except Exception as err:
        rec = None
        print(f'{subject} run {run_id}: cannot open Neon recording ({err!r})')
    onsets = ev_fnirs['onset'].values.astype(float)
    # Neon device time (ns) -> fNIRS time (s)
    off = np.median(neon['timestamps'].values - neon['time'].values * 1e-9)
    t_lo, t_hi = neon['timestamps'].min(), neon['timestamps'].max()

    def to_fnirs(ns):
        return np.asarray(ns, float) * 1e-9 + off

    blinks = np.zeros((0, 2))
    if rec is not None:
        try:
            b = rec.blinks.data
            blinks = np.column_stack([to_fnirs(b['start_time']), to_fnirs(b['stop_time'])])
        except Exception:
            pass

    # --- pupil ---
    with contextlib.redirect_stdout(io.StringIO()):
        t_p, pupil = preprocess_pupil(neon, rec=rec, detrend_order=2, is_rm_phasic=False, events_df=ev_fnirs)
    status = 'ok'
    if t_p is None:
        status = 'pupil >30% missing'
    else:
        t_p = np.asarray(t_p)
        fs_p = 1 / np.median(np.diff(t_p))
        dp = np.gradient(pupil, 1 / fs_p)
        in_blink = np.zeros(len(t_p), bool)
        for a, b in blinks:
            in_blink |= (t_p >= a - 0.05) & (t_p <= b + 0.1)

        def mean_in(sig, i, w):
            m = (t_p >= onsets[i] + w[0]) & (t_p < onsets[i] + w[1])
            return sig[m].mean() if m.sum() >= 3 else np.nan
        Lp = int(round(TRIAL_LEN * fs_p))
        epochs = np.full((n_tr, Lp), np.nan)
        for i in range(n_tr):
            m_all = (t_p >= onsets[i] + PUPIL_DERIV_WIN[0]) & (t_p <= onsets[i] + PUPIL_TEPR_WIN[1])
            if m_all.sum() < 10 or in_blink[m_all].any():
                continue
            base = mean_in(pupil, i, PUPIL_BASE_WIN)
            out['pupil_base'][i] = base
            out['pupil_deriv'][i] = mean_in(dp, i, PUPIL_DERIV_WIN)
            out['pupil_tepr'][i] = mean_in(pupil, i, PUPIL_TEPR_WIN) - base
            k = np.searchsorted(t_p, onsets[i])
            if k + Lp <= len(pupil):
                e = pupil[k:k + Lp]
                epochs[i] = e - e.mean()
        out['pupil_base_sd9'] = rolling(out['pupil_base'], ROLL_SD_N, lambda x: x.std(ddof=1), ROLL_SD_N - 3)
        out['pupil_tepr_sd9'] = rolling(out['pupil_tepr'], ROLL_SD_N, lambda x: x.std(ddof=1), ROLL_SD_N - 3)
        h = PUPIL_EVOKED_N // 2
        ok_ep = np.isfinite(epochs[:, 0])
        for i in range(n_tr):
            sl = slice(max(0, i - h), min(n_tr, i + h + 1))
            v = ok_ep[sl]
            if v.sum() >= PUPIL_EVOKED_N - 5:
                avg = epochs[sl][v].mean(0)
                out['pupil_evoked13'][i] = avg.max() - avg.min()
        for i in range(n_tr):
            m = (t_p >= onsets[i] - WIN_S / 2) & (t_p < onsets[i] + WIN_S / 2) & ~in_blink
            if m.sum() >= MIN_COVER_FRAC * WIN_S * fs_p:
                out['pupil_sd_win'][i] = pupil[m].std()

    # --- blinks, fixations, saccades in the centred window ---
    if rec is None:
        return out, status + '; no Neon recording (no blink / fixation / saccade features)'
    try:
        fx = rec.fixations.data
        sc = rec.saccades.data
    except Exception as err:
        return out, status + f'; no fixation / saccade data ({err!r})'
    fix_t = to_fnirs(fx['start_time'])
    fix_d = (fx['stop_time'] - fx['start_time']) * 1e-9
    sac_t = to_fnirs(sc['start_time'])
    sac_amp = np.asarray(sc['amplitude_angle'], float)
    sac_v = np.asarray(sc['max_velocity'], float)
    bl_t, bl_d = blinks[:, 0], blinks[:, 1] - blinks[:, 0]
    for i in range(n_tr):
        a, b = max(onsets[i] - WIN_S / 2, t_lo), min(onsets[i] + WIN_S / 2, t_hi)
        cov = b - a
        if cov < MIN_COVER_FRAC * WIN_S:
            continue
        mins = cov / 60
        mb = (bl_t >= a) & (bl_t < b)
        mf = (fix_t >= a) & (fix_t < b)
        ms = (sac_t >= a) & (sac_t < b)
        out['blink_rate'][i] = mb.sum() / mins
        out['blink_dur'][i] = bl_d[mb].mean() if mb.any() else np.nan
        out['fix_rate'][i] = mf.sum() / mins
        out['fix_dur'][i] = np.median(fix_d[mf]) if mf.any() else np.nan
        out['sacc_rate'][i] = ms.sum() / mins
        out['sacc_amp'][i] = np.median(sac_amp[ms]) if ms.any() else np.nan
        out['sacc_peakvel'][i] = np.median(sac_v[ms]) if ms.any() else np.nan
        out['microsacc_rate'][i] = (ms & (sac_amp < MICROSACC_DEG)).sum() / mins
    return out, status


#%% per subject
def extract_subject(subject):
    tables, log = [], {}
    for run_key in RUN_KEYS:
        run_id = int(run_key[-1])
        fif = get_eeg_fif_path(subject, run_id)
        if not os.path.exists(fif):
            continue
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        ev = load_events(subject, run_id)
        chs = [c for c in CH_NAMES if c in raw.ch_names]
        data = raw.get_data(picks=chs) * 1e6
        onsets = ev['onset'].values.astype(float)
        is_cc = ((ev['trial_type'] == 'city') & (ev['response_code'] == 1)).values
        rt = np.where(is_cc, ev['reaction_time'].values, np.nan)
        df = pd.DataFrame({'subject': subject, 'run': run_id, 'trial_idx': np.arange(len(ev)),
                           'trial_type': ev['trial_type'].values, 'response_code': ev['response_code'].values})
        feats = {}
        feats.update(eeg_window_features(data, chs, sfreq, onsets))
        feats.update(eeg_trial_features(data, chs, sfreq, onsets, rt, is_cc))
        # eye tracking, on the fNIRS events of the same run (same rows as the EEG events)
        ev_f_path = os.path.join(PROJECT_PATH, subject, 'nirs', f'{subject}_task-gradCPT_run-{run_id:02d}_events.tsv')
        if os.path.isfile(ev_f_path):
            ev_f = pd.read_csv(ev_f_path, sep='\t')
            if len(ev_f) != len(ev) or (ev_f['trial_type'].values != ev['trial_type'].values).any():
                log[run_key] = 'fNIRS events do not match EEG events; eye features skipped'
                eye = {k: np.full(len(ev), np.nan) for k, v in feature_info().items() if v['family'] == 'eye'}
            else:
                eye, log[run_key] = eye_features(subject, run_id, ev_f)
        else:
            log[run_key] = 'no fNIRS events file; eye features skipped'
            eye = {k: np.full(len(ev), np.nan) for k, v in feature_info().items() if v['family'] == 'eye'}
        feats.update(eye)
        for k in feature_info():
            df[k] = feats.get(k, np.nan)
        tables.append(df)
    if not tables:
        return None, log
    return pd.concat(tables, ignore_index=True), log


#%% main
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='*')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]
    else:  # the subjects that have the EEG feature table
        subjects = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                          for p in glob.glob(get_features_path('sub-*')))
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, list, dict))}
    params['THETA_FREQS'] = THETA_FREQS.tolist()
    for subject in tqdm(subjects, desc='subjects'):
        out_path = get_extra_features_path(subject)
        if os.path.exists(out_path) and not args.overwrite:
            print(f'{subject}: exists, skipping')
            continue
        try:
            df, log = extract_subject(subject)
        except Exception as err:
            import traceback; traceback.print_exc()
            print(f'{subject}: failed: {err!r}')
            continue
        if df is None:
            print(f'{subject}: no gradCPT runs')
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        with open(out_path.replace('.csv.gz', '_params.json'), 'w') as f:
            json.dump(dict(params=params, eye_log=log), f, indent=1, default=str)
        n_eye = np.isfinite(df['blink_rate']).mean()
        print(f'{subject}: {len(df)} trials, eye coverage {n_eye:.0%}, log {log}')
