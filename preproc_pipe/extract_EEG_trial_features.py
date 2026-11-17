#%% Single-trial EEG and eye-tracking features
# One row per gradCPT trial (all 4 trial types), keyed by (run, trial_idx), where trial_idx is the row of
# the run's events.tsv. EEG and fNIRS events.tsv come from the same gradCPT .mat file and match row for
# row (same trial_type / response_code), so these rows can be joined to the fNIRS events used in
# ../modeling (see get_trial_modulator below).
# (Merged from the former extract_EEG_trial_features.py + extract_extra_trial_features.py, 2026-11-17.)
#
# Spectral settings (same as the two original scripts, so the saved feature tables stay valid):
#   Welch PSD / coherence: 2 s Hann segments (WELCH_SEG), 50 % overlap -> 0.5 Hz bins.
#   Morlet theta: 4-8 Hz in 1 Hz steps, n_cycles = f / 2 (THETA_N_CYCLES).
#   The db4 wavelet (DWT) bands are octave bands set by the decomposition level.
#
# EEG source: <subject>/preprocessed_EEG_and_events/*_task-gradCPT_run-XX_preproc_eeg.fif, the same files
# run_model_cont_EEG_fNIRS.py reads via utils.eeg_preproc_subj_level. Channels fz, cz, pz, oz.
#
# ---------------------------------------------------------------------------------------------------------
# A. ERP (mountain trials only; NaN for city trials), from epochs -0.2 to 1.6 s, baseline (-0.2, 0),
#   linear detrend, 10 Hz zero-phase low-pass -- same epoching as eeg_epoch_subj_level.
#   Windows and estimators follow single_trial_ERP_estimation.ipynb:
#     N2 window 0.42-0.62 s (negative), P3 window 0.25-0.80 s (positive; widened from 0.60-0.95 s so the
#     search starts before the N2 and a late search does not miss the P3)
#     <ch>_<comp>_lat_woody, _amp_woody    Woody (1967) cross-correlation alignment
#     <ch>_<comp>_lat_fal50                 50% fractional-area latency (Kiesel 2008)
#     <ch>_<comp>_meanamp                   mean amplitude in the window
#     <ch>_<comp>_lat_localpeak, _amp_localpeak   Luck (2014) local peak (NaN latency if none)
#     <ch>_<comp>_lat_mlr, _amp_mlr         template + derivative regression (Hu/Mouraux)
#     <ch>_<comp>_area                      polarity-rectified area in the window (uV*s, >= 0)
#     <ch>_<comp>_area_signed               signed integral in the window (uV*s)
#     <ch>_<comp>_edge_woody / _nopeak_localpeak   QC flags
#     <ch>_N2P3_ptp_woody                   N2-to-P3 peak-to-peak amplitude (>= 0): the Woody P3 peak minus
#                                           the trial's minimum between the P3 window start (0.25 s) and the
#                                           Woody P3 peak latency
#     <ch>_N2P3_N2amp_woody, _N2lat_woody   that minimum (the N2 trough) and its latency
#   Notebook recommendation: P3 latency -> woody or fal50; P3 amplitude -> mlr, woody or localpeak
#   (or area/meanamp). Single-trial N2 is not recoverable at this SNR (r <= 0.11 in simulation);
#   use only meanamp/area for N2, and expect attenuated slopes for every ERP feature.
#   Templates (woody, mlr) are built from clean trials of all mountain trials (correct + incorrect)
#   of a subject and channel; every mountain trial is scored, clean or not.
#   <ch>_erp_ok   epoch peak-to-peak <= 100 uV on this channel
#   erp_ok_all    epoch kept by the existing pipeline (all present channels <= 100 uV)
#
# B. Band power (all trials), from the continuous preprocessed EEG (0.1-45 Hz, re-referenced, ICA):
#     <ch>_<band>_<pre|post>       absolute power (uV^2) in [onset - W, onset) / [onset, onset + W), W = 20 s
#     <ch>_<band>_<pre|post>_rel   band power / total power over 1-30 Hz
#   bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30 Hz. PSD by Welch (1 s Hann segments, 50% overlap,
#   median average, so short artifacts do not dominate the 20 s window).
#   Windows are clipped to the recording; bp_<pre|post>_cover_s is the covered length and the power is
#   NaN if it is below half the window. bp_pre_before_task_s = seconds of the pre window that precede
#   the first trial onset of the run (pre-task EEG).
#
# C. Literature-survey features ("EEG & Pupil Features of Sustained Attention -- Literature Survey",
#   2026-11-06); listed with family and scope in feature_info(). Scopes (how EEG_extra_features_vs_VTC.py
#   uses each feature):
#   window  : computed in a window CENTRED on the trial onset (WIN_S = 10 s, i.e. onset +- 5 s; about the
#             7 s FWHM of the VTC smoothing). Tested on all trials.
#   rolling : trial-to-trial statistic over a centred run of trials (ROLL_SD_N = 9 for SDs, ITPC_N = 15 for
#             phase locking, PUPIL_EVOKED_N = 13 for the stimulus-locked pupil response). All trials.
#   trial   : single-trial value. Tested on city_correct trials only, so a button press is present on every
#             trial and press / no-press differences cannot drive the result.
#
#   EEG window, per channel (Welch, 1 s Hann, 50 % overlap, median average):
#     <ch>_ap_exp, <ch>_ap_off     aperiodic exponent / offset: robust log-log line fit 2-40 Hz (the FOOOF
#                                  'robust_ap_fit': fit, keep points below the fit, refit)
#     <ch>_iaf                     individual alpha frequency: peak of the aperiodic-removed spectrum in 7-14 Hz
#                                  (parabolic interpolation; NaN if no local peak >= 0.05 log10 units)
#     <ch>_alpha_flat, <ch>_theta_flat   periodic power: mean aperiodic-removed log10 power, 7-14 / 4-8 Hz
#     <ch>_medfreq                 spectral median frequency, 1-30 Hz
#     <ch>_tbr, <ch>_tar           log10 theta/beta and theta/alpha absolute-power ratios
#     <ch>_theta_win, <ch>_alpha_win   log10 absolute band power in the same window (reference: band power
#                                  is already in section B, here only re-measured in the centred window)
#     coh_<a>_<b>_<band>           magnitude-squared coherence (theta, alpha) for fz-pz and fz-oz
#   EEG trial / rolling, theta (Morlet 4-8 Hz, n_cycles = f/2, on the continuous run):
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
#   Eye tracking (Pupil Labs Neon; BIDS physio .tsv in fNIRS time + raw Neon recording for blinks,
#   fixations and saccades; recording matched through the physio .json 'SourceNeonRec')
#   Pupil = mean of both eyes, blinks interpolated, quadratic detrend, 30 Hz low-pass, 60 Hz
#   (utils_eyetracking.preprocess_pupil, as in cmp_EEG_split_by_RT_pupil.py).
#     pupil_base     mean pupil in [-0.2, 0) s                                                    (trial)
#     pupil_deriv    mean pupil derivative in [-0.5, 0) s                                         (trial)
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
#   Not computable with this montage / design: EEG microstates (4 scalp channels), SSVEP (no flicker),
#   MMN (no auditory oddball), CNV (no cue-target interval), fMRI dynamic connectivity.
# ---------------------------------------------------------------------------------------------------------
#
# Output: <data_save_path>/<subject>/features/<subject>_EEG_trial_features.csv.gz (+ _params.json)
#
# Usage:
#   python extract_EEG_trial_features.py                    # all subjects with a preprocessed gradCPT .fif
#   python extract_EEG_trial_features.py --subjects 723 695 --overwrite
#   python extract_EEG_trial_features.py --bp-win 20
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import data_save_path, get_eeg_preproc_dir, gen_EEG_event_tsv

PROJECT_PATH = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24'
sys.path.insert(0, os.path.join(PROJECT_PATH, 'code', 'eyetracking'))

#%% parameters
CH_NAMES = ['fz', 'cz', 'pz', 'oz']
RUN_KEYS = ['gradcpt1', 'gradcpt2', 'gradcpt3']

# Welch / coherence segment length
WELCH_SEG = 2.0        # s (0.5 Hz bins)

# ERP epoching: same as eeg_epoch_subj_level (stimulus-locked: -0.2 to 1.6 s, linear detrend,
# baseline (None, 0), 100 uV peak-to-peak rejection)
EPOCH_TMIN, EPOCH_TMAX = -0.2, 1.6
EPOCH_DETREND = 1
REJECT_PTP_UV = 100.0
ERP_TRIAL_TYPES = ('mnt',)  # trial_type values that get ERP features

COMPONENTS = {
    'N2': dict(sign=-1, win=(0.42, 0.62)),
    'P3': dict(sign=+1, win=(0.25, 0.80)),
}
LP_HZ = 10.0
LOCAL_PEAK_HW = 0.020  # s
FAL_FRAC = 0.5
WOODY_MAX_LAG = 0.10   # s
WOODY_N_ITER = 10
MLR_PAD = 0.10         # s
MIN_TEMPLATE_TRIALS = 10  # clean trials needed to build a woody / mlr template

# band power
BANDS = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}
TOTAL_BAND = (1.0, 30.0)
BP_WIN = 20.0          # s, window length before and after onset
MIN_COVER_FRAC = 0.5

# literature-survey features
WIN_S = 10.0                # centred window length (s)
AP_RANGE = (2.0, 40.0)
ALPHA_RANGE = (7.0, 14.0)
THETA_RANGE = (4.0, 8.0)
IAF_MIN_PEAK = 0.05         # log10 units above the aperiodic fit
COH_PAIRS = [('fz', 'pz'), ('fz', 'oz')]
TRIAL_LEN = 0.8             # s, single-trial window after onset
ROLL_SD_N = 9
ITPC_N = 15
ITPC_WIN = (0.0, 0.4)
THETA_FREQS = np.arange(4.0, 8.01, 1.0)
THETA_N_CYCLES = THETA_FREQS / 2
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


def get_features_path(subject):
    return os.path.join(data_save_path, subject, 'features', f'{subject}_EEG_trial_features.csv.gz')


#%% ERP estimators (from single_trial_ERP_estimation.ipynb), X: (n_trials, n_times) in uV
def lowpass(X, sfreq, hz=LP_HZ, order=4):
    sos = butter(order, hz, btype='low', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1)


def _win_idx(t, win):
    return np.where((t >= win[0]) & (t <= win[1]))[0]


def window_peak(Y, t, comp):
    idx = _win_idx(t, comp['win'])
    k = (comp['sign'] * Y[:, idx]).argmax(1)
    return dict(amp=Y[np.arange(len(Y)), idx[k]], lat=t[idx[k]],
                edge=(k == 0) | (k == len(idx) - 1))


def est_local_peak(Y, t, sfreq, comp, hw=LOCAL_PEAK_HW):
    """Largest point in the window that exceeds the mean of the +-hw neighbours on both sides.
    No such point: amp = window mean, lat = NaN."""
    s = comp['sign']
    k = max(1, int(round(hw * sfreq)))
    idx = _win_idx(t, comp['win'])
    n = len(Y)
    c = np.concatenate([np.zeros((n, 1)), np.cumsum(Y, 1)], 1)
    left = (c[:, idx] - c[:, idx - k]) / k
    right = (c[:, idx + k + 1] - c[:, idx + 1]) / k
    seg = Y[:, idx]
    score = np.where((s * seg > s * left) & (s * seg > s * right), s * seg, -np.inf)
    kk = score.argmax(1)
    has = np.isfinite(score.max(1))
    rows = np.arange(n)
    return dict(amp=np.where(has, seg[rows, kk], seg.mean(1)),
                lat=np.where(has, t[idx[kk]], np.nan), nopeak=~has)


def est_window_integrals(Y, t, sfreq, comp, frac=FAL_FRAC):
    """Mean amplitude, rectified / signed area and fractional-area latency in the window."""
    idx = _win_idx(t, comp['win'])
    seg = Y[:, idx]
    pos = np.clip(comp['sign'] * seg, 0, None)
    cum = np.cumsum(pos, 1)
    tot = cum[:, -1]
    kk = (cum >= frac * tot[:, None]).argmax(1)
    dt = 1.0 / sfreq
    return dict(meanamp=seg.mean(1),
                area=tot * dt,
                area_signed=np.trapezoid(seg, dx=dt, axis=1),
                lat_fal=np.where(tot > 0, t[idx[kk]], np.nan))


def est_woody(Y, t, sfreq, comp, fit_mask, max_lag=WOODY_MAX_LAG, n_iter=WOODY_N_ITER, return_template=False):
    """Woody alignment. The template is iterated on the fit_mask (clean) trials; every trial gets the
    lag that maximises its correlation with the final template. return_template=True adds the final
    template (on t[window]), the window indices, the per-trial lag (s), the template peak index and the
    per-trial correlation at the chosen lag (for visualization)."""
    n = len(Y)
    idx = _win_idx(t, comp['win'])
    L = int(round(max_lag * sfreq))
    lags = np.arange(-L, L + 1)
    seg = np.stack([Y[:, idx + l] for l in lags], 1)  # (n, n_lag, n_win)
    seg_c = seg - seg.mean(-1, keepdims=True)
    seg_n = seg_c / np.maximum(np.linalg.norm(seg_c, axis=-1, keepdims=True), 1e-12)
    rows = np.arange(n)
    fit_rows = rows[fit_mask]
    lag_i = np.full(n, L)
    tmpl = Y[fit_mask][:, idx].mean(0)
    for _ in range(n_iter):
        tc = tmpl - tmpl.mean()
        new = (seg_n @ (tc / max(np.linalg.norm(tc), 1e-12))).argmax(1)
        tmpl = seg[fit_rows, new[fit_rows]].mean(0)
        converged = np.array_equal(new, lag_i)
        lag_i = new
        if converged:
            break
    kp = (comp['sign'] * tmpl).argmax()
    lag_s = lags[lag_i]
    pk = idx[kp] + lag_s
    out = dict(amp=Y[rows, pk], lat=t[pk], edge=np.abs(lag_s) == L, pk=pk)
    if return_template:
        tc = tmpl - tmpl.mean()
        r = seg_n[rows, lag_i] @ (tc / max(np.linalg.norm(tc), 1e-12))
        out.update(template=tmpl, win_idx=idx, lag_s=lag_s / sfreq, peak_idx=kp, corr=r)
    return out


def est_template_mlr(Y, t, sfreq, comp, fit_mask, pad=MLR_PAD):
    """Regress each trial on [1, T, T', (t - t_peak) T'] (T = clean-trial average) and read the peak
    from the fitted curve. Trials with template weight <= 0 are flagged."""
    win = comp['win']
    fidx = _win_idx(t, (win[0] - pad, win[1] + pad))
    tf = t[fidx]
    T = Y[fit_mask][:, fidx].mean(0)
    dT = np.gradient(T, 1 / sfreq)
    in_w = (tf >= win[0]) & (tf <= win[1])
    tc = tf[in_w][(comp['sign'] * T[in_w]).argmax()]
    Dm = np.column_stack([np.ones_like(T), T, dT, (tf - tc) * dT])
    B, *_ = np.linalg.lstsq(Dm, Y[:, fidx].T, rcond=None)
    full = np.full_like(Y, -np.inf * comp['sign'])
    full[:, fidx] = (Dm @ B).T
    res = window_peak(full, t, comp)
    res['edge'] = res['edge'] | (B[1] <= 0)
    return res


def n2p3_ptp(Y, t, p3_pk, n2_start):
    """N2-to-P3 peak-to-peak: Y at the P3 peak sample p3_pk minus the trial's minimum between n2_start (s)
    and p3_pk (inclusive), so it is >= 0. Returns ptp, the N2 trough amplitude and its latency."""
    i0 = int(np.searchsorted(t, n2_start))
    cols = np.arange(Y.shape[1])
    lo = np.minimum(i0, p3_pk)  # a P3 peak before n2_start gives a one-sample segment (ptp = 0)
    seg = np.where((cols[None, :] >= lo[:, None]) & (cols[None, :] <= p3_pk[:, None]), Y, np.inf)
    k = seg.argmin(1)
    rows = np.arange(len(Y))
    n2 = Y[rows, k]
    return Y[rows, p3_pk] - n2, n2, t[k]


def erp_features(X, t, sfreq, ok):
    """All ERP features of one channel. X: (n_trials, n_times) uV; ok: clean-trial mask."""
    Y = lowpass(X, sfreq)
    out = {}
    use_template = ok.sum() >= MIN_TEMPLATE_TRIALS
    for cname, comp in COMPONENTS.items():
        w = est_window_integrals(Y, t, sfreq, comp)
        lp = est_local_peak(Y, t, sfreq, comp)
        out[f'{cname}_meanamp'] = w['meanamp']
        out[f'{cname}_area'] = w['area']
        out[f'{cname}_area_signed'] = w['area_signed']
        out[f'{cname}_lat_fal50'] = w['lat_fal']
        out[f'{cname}_amp_localpeak'] = lp['amp']
        out[f'{cname}_lat_localpeak'] = lp['lat']
        out[f'{cname}_nopeak_localpeak'] = lp['nopeak']
        nan = np.full(len(X), np.nan)
        if use_template:
            wd = est_woody(Y, t, sfreq, comp, ok)
            ml = est_template_mlr(Y, t, sfreq, comp, ok)
        else:
            wd = dict(amp=nan, lat=nan, edge=np.ones(len(X), bool))
            ml = dict(amp=nan, lat=nan, edge=np.ones(len(X), bool))
        out[f'{cname}_amp_woody'] = wd['amp']
        out[f'{cname}_lat_woody'] = wd['lat']
        out[f'{cname}_edge_woody'] = wd['edge']
        out[f'{cname}_amp_mlr'] = ml['amp']
        out[f'{cname}_lat_mlr'] = ml['lat']
        out[f'{cname}_edge_mlr'] = ml['edge']
        if cname == 'P3':
            if use_template:
                ptp, n2, n2_lat = n2p3_ptp(Y, t, wd['pk'], comp['win'][0])
            else:
                ptp = n2 = n2_lat = nan
            out['N2P3_ptp_woody'] = ptp
            out['N2P3_N2amp_woody'] = n2
            out['N2P3_N2lat_woody'] = n2_lat
    return out


#%% spectra
def spec_nperseg(sfreq):
    """Welch / coherence segment length in samples (WELCH_SEG; nfft = nperseg, no zero padding)."""
    return int(round(WELCH_SEG * sfreq))


def psd_welch(seg, sfreq):
    nper = spec_nperseg(sfreq)
    f, P = welch(seg, fs=sfreq, nperseg=nper, noverlap=nper // 2, nfft=nper, window='hann',
                 average='median', axis=-1)
    return f, P


def coh_welch(x, y, sfreq):
    nper = spec_nperseg(sfreq)
    return coherence(x, y, fs=sfreq, nperseg=nper, noverlap=nper // 2, nfft=nper, window='hann')


#%% band power
def band_power_windows(data, sfreq, onsets_s, win, side):
    """Welch band power in [onset - win, onset) (side='pre') or [onset, onset + win) ('post').
    data: (n_ch, n_samples) in uV. Returns (power dict band -> (n_trials, n_ch), rel dict, cover_s)."""
    n_ch, n_samp = data.shape
    nper = spec_nperseg(sfreq)
    n_tr = len(onsets_s)
    pw = {b: np.full((n_tr, n_ch), np.nan) for b in BANDS}
    rel = {b: np.full((n_tr, n_ch), np.nan) for b in BANDS}
    cover = np.zeros(n_tr)
    for i, on in enumerate(onsets_s):
        a, b = (on - win, on) if side == 'pre' else (on, on + win)
        s0 = max(int(round(a * sfreq)), 0)
        s1 = min(int(round(b * sfreq)), n_samp)
        cover[i] = max(s1 - s0, 0) / sfreq
        if cover[i] < MIN_COVER_FRAC * win or s1 - s0 < nper:
            continue
        f, P = psd_welch(data[:, s0:s1], sfreq)
        df = f[1] - f[0]
        tot = P[:, (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])].sum(-1) * df
        for bname, (lo, hi) in BANDS.items():
            bp = P[:, (f >= lo) & (f < hi)].sum(-1) * df
            pw[bname][i] = bp
            rel[bname][i] = bp / tot
    return pw, rel, cover


#%% EEG / events loading
def load_events(subject, run_id):
    ev_path = os.path.join(get_eeg_preproc_dir(subject), f'{subject}_task-gradCPT_run-{run_id:02d}_events.tsv')
    if not os.path.exists(ev_path):
        gen_EEG_event_tsv(subject.split('-')[-1])
    return pd.read_csv(ev_path, sep='\t')


def get_eeg_fif_path(subject, run_id):
    return os.path.join(get_eeg_preproc_dir(subject), f'{subject}_task-gradCPT_run-{run_id:02d}_preproc_eeg.fif')


def load_eeg_runs(subject):
    """Preprocessed gradCPT EEG per run, read from the same *_preproc_eeg.fif files that
    utils.eeg_preproc_subj_level loads for ../modeling/run_model_cont_EEG_fNIRS.py. (The
    <subject>_preprocessed_dict.pkl.gz copy can differ: eeg_preprocess_pipeline.py rewrites the
    dict on every rerun but keeps an existing .fif.)"""
    eeg_dict = dict()
    for run_key in RUN_KEYS:
        fif = get_eeg_fif_path(subject, int(run_key[-1]))
        if os.path.exists(fif):
            eeg_dict[run_key] = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    return eeg_dict


#%% literature-survey features
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
        add(f'{ch}_theta_win', 'eeg_spectral', 'window', f'{ch} theta power ({WIN_S:g} s)', ref=True)
        add(f'{ch}_alpha_win', 'eeg_spectral', 'window', f'{ch} alpha power ({WIN_S:g} s)', ref=True)
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
    nper = spec_nperseg(sfreq)
    s0, s1 = win_bounds(onsets, n_samp, sfreq)
    info = feature_info()
    out = {k: np.full(len(onsets), np.nan) for k in info if info[k]['family'] == 'eeg_spectral'}
    for i in range(len(onsets)):
        if s1[i] - s0[i] < max(MIN_COVER_FRAC * WIN_S * sfreq, 2 * nper):
            continue
        seg = data[:, s0[i]:s1[i]]
        f, P = psd_welch(seg, sfreq)
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
                fc, C = coh_welch(seg[chs.index(a)], seg[chs.index(b)], sfreq)
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
    W = mne.time_frequency.tfr_array_morlet(data[None], sfreq, THETA_FREQS, n_cycles=THETA_N_CYCLES,
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

def eye_features_for_run(subject, run_id, ev):
    """Eye features on the fNIRS events of the same run (same rows as the EEG events)."""
    nan = {k: np.full(len(ev), np.nan) for k, v in feature_info().items() if v['family'] == 'eye'}
    ev_f_path = os.path.join(PROJECT_PATH, subject, 'nirs', f'{subject}_task-gradCPT_run-{run_id:02d}_events.tsv')
    if not os.path.isfile(ev_f_path):
        return nan, 'no fNIRS events file; eye features skipped'
    ev_f = pd.read_csv(ev_f_path, sep='\t')
    if len(ev_f) != len(ev) or (ev_f['trial_type'].values != ev['trial_type'].values).any():
        return nan, 'fNIRS events do not match EEG events; eye features skipped'
    return eye_features(subject, run_id, ev_f)


def extract_subject(subject, bp_win=BP_WIN):
    """Returns (feature table, eye-tracking log per run)."""
    eeg_dict = load_eeg_runs(subject)

    run_tables, log = [], {}
    erp_store = {}  # ch -> list of (run_df_index, X, ok) to score mountain trials pooled across runs
    for run_key in RUN_KEYS:
        if run_key not in eeg_dict:
            continue
        raw = eeg_dict[run_key]
        run_id = int(run_key[-1])
        sfreq = raw.info['sfreq']
        ev = load_events(subject, run_id)
        df = pd.DataFrame({'subject': subject, 'run': run_id, 'trial_idx': np.arange(len(ev)),
                           'onset_eeg': ev['onset'].values, 'trial_type': ev['trial_type'].values,
                           'response_code': ev['response_code'].values,
                           'reaction_time': ev['reaction_time'].values,
                           'VTC': ev['VTC'].values,
                           'VTC_smoothed': ev['VTC_smoothed'].values if 'VTC_smoothed' in ev else np.nan})
        chs = [c for c in CH_NAMES if c in raw.ch_names]
        data = raw.get_data(picks=chs) * 1e6
        onsets = ev['onset'].values.astype(float)

        # --- band power, all trials ---
        for side in ['pre', 'post']:
            pw, rel, cover = band_power_windows(data, sfreq, onsets, bp_win, side)
            df[f'bp_{side}_cover_s'] = cover
            for bname in BANDS:
                for c_i, ch in enumerate(chs):
                    df[f'{ch}_{bname}_{side}'] = pw[bname][:, c_i]
                    df[f'{ch}_{bname}_{side}_rel'] = rel[bname][:, c_i]
        df['bp_pre_before_task_s'] = np.clip(onsets[0] - (onsets - bp_win), 0, bp_win)

        # --- ERP epochs, mountain trials ---
        erp_rows = np.where(df['trial_type'].isin(ERP_TRIAL_TYPES))[0]
        events = np.column_stack([(onsets[erp_rows] * sfreq).astype(int),
                                  np.zeros(len(erp_rows), int), np.ones(len(erp_rows), int)])
        epochs = mne.Epochs(raw, events, event_id={'stim': 1}, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                            baseline=(None, 0), detrend=EPOCH_DETREND, reject=None, picks=chs,
                            preload=True, verbose=False)
        kept_rows = erp_rows[epochs.selection]  # epochs outside the recording are dropped by mne
        X_all = epochs.get_data(picks=chs) * 1e6  # (n_ep, n_ch, n_times)
        ptp = X_all.max(-1) - X_all.min(-1)
        df['erp_ok_all'] = False
        df.loc[kept_rows, 'erp_ok_all'] = (ptp <= REJECT_PTP_UV).all(1)
        times = epochs.times
        for c_i, ch in enumerate(chs):
            ok = ptp[:, c_i] <= REJECT_PTP_UV
            df[f'{ch}_erp_ok'] = False
            df.loc[kept_rows, f'{ch}_erp_ok'] = ok
            erp_store.setdefault(ch, []).append((len(run_tables), kept_rows, X_all[:, c_i], ok))

        # --- literature-survey features: EEG window / trial, eye tracking ---
        is_cc = ((ev['trial_type'] == 'city') & (ev['response_code'] == 1)).values
        rt = np.where(is_cc, ev['reaction_time'].values, np.nan)
        feats = {}
        feats.update(eeg_window_features(data, chs, sfreq, onsets))
        feats.update(eeg_trial_features(data, chs, sfreq, onsets, rt, is_cc))
        eye, log[run_key] = eye_features_for_run(subject, run_id, ev)
        feats.update(eye)
        df = pd.concat([df, pd.DataFrame({k: feats.get(k, np.nan) for k in feature_info()}, index=df.index)],
                       axis=1)
        run_tables.append(df)

    if not run_tables:
        return None, log

    # score ERP features per channel, templates pooled across runs
    for ch, parts in erp_store.items():
        X = np.concatenate([p[2] for p in parts])
        ok = np.concatenate([p[3] for p in parts])
        feats = erp_features(X, times, sfreq, ok)
        start = 0
        for tbl_i, rows, _, _ in parts:
            sl = slice(start, start + len(rows))
            tbl = run_tables[tbl_i]
            new = {}
            for name, val in feats.items():
                col = np.full(len(tbl), np.nan)  # NaN for city trials; QC flags stored as 0/1
                col[rows] = val[sl]
                new[f'{ch}_{name}'] = col
            run_tables[tbl_i] = pd.concat([tbl, pd.DataFrame(new, index=tbl.index)], axis=1)
            start += len(rows)

    out = pd.concat(run_tables, ignore_index=True)
    # channels removed in every run: add NaN columns so every subject has the same layout
    for ch in CH_NAMES:
        for side in ['pre', 'post']:
            for bname in BANDS:
                for suf in ['', '_rel']:
                    if f'{ch}_{bname}_{side}{suf}' not in out:
                        out[f'{ch}_{bname}_{side}{suf}'] = np.nan
        if f'{ch}_erp_ok' not in out:
            out[f'{ch}_erp_ok'] = False
    # a channel removed in only some runs is NaN (bands, ERP) / False (erp_ok) in those runs
    out[[c for c in out if c.endswith('_erp_ok')]] = out[[c for c in out if c.endswith('_erp_ok')]].fillna(False)
    return out, log


#%% loader for ../modeling
def load_trial_features(subject):
    return pd.read_csv(get_features_path(subject))


def get_trial_modulator(subject, run, ev_df, columns, feat_df=None):
    """Feature rows aligned to the rows of an fNIRS events.tsv DataFrame (or a row subset of it).

    ev_df: the run's events.tsv as read in ../modeling (original row index kept, e.g. after
           ev_df[(ev_df.trial_type == 'mnt') & (ev_df.response_code == 0)]).
    columns: feature column name(s), e.g. 'cz_P3_area' or ['cz_P3_area', 'cz_alpha_pre'].
    Returns a DataFrame indexed like ev_df. Raises if trial_type / response_code disagree."""
    if feat_df is None:
        feat_df = load_trial_features(subject)
    f = feat_df[feat_df['run'] == int(run)].set_index('trial_idx')
    missing = ev_df.index.difference(f.index)
    if len(missing):
        raise ValueError(f'{subject} run {run}: events rows {list(missing[:5])} not in the feature table')
    f = f.loc[ev_df.index]
    bad = (f['trial_type'].values != ev_df['trial_type'].values) | \
          (f['response_code'].values != ev_df['response_code'].values)
    if bad.any():
        raise ValueError(f'{subject} run {run}: {bad.sum()} trials disagree between EEG and fNIRS events')
    cols = [columns] if isinstance(columns, str) else list(columns)
    return f[cols].set_axis(ev_df.index)


#%% main
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', nargs='*', help='subject ids (e.g. 723 695); default: all with a preprocessed gradCPT .fif')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--bp-win', type=float, default=BP_WIN, help='band-power window before/after onset (s)')
    args = ap.parse_args()

    if args.subjects:
        subjects = [s if s.startswith('sub-') else f'sub-{s}' for s in args.subjects]
    else:
        subjects = sorted({os.path.basename(p).split('_')[0] for p in
                           glob.glob(os.path.join(data_save_path, 'sub-*', 'preprocessed_EEG_and_events',
                                                  'sub-*_task-gradCPT_run-*_preproc_eeg.fif'))})

    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, list, dict))}
    params.update(THETA_FREQS=THETA_FREQS.tolist(), THETA_N_CYCLES=THETA_N_CYCLES.tolist(), BP_WIN=args.bp_win)
    for subject in tqdm(subjects, desc='subjects'):
        out_path = get_features_path(subject)
        if os.path.exists(out_path) and not args.overwrite:
            print(f'{subject}: features exist, skipping')
            continue
        try:
            df, log = extract_subject(subject, bp_win=args.bp_win)
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
        n_mnt = (df.trial_type == 'mnt').sum()
        print(f'{subject}: {len(df)} trials ({n_mnt} mnt, {df.erp_ok_all.sum()} clean ERP epochs), '
              f'eye coverage {np.isfinite(df["blink_rate"]).mean():.0%}, {df.shape[1]} columns -> {out_path}')
