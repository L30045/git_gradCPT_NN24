#%% Single-trial EEG features for EEG-modulated HRF designs
# One row per gradCPT trial (all 4 trial types), keyed by (run, trial_idx), where trial_idx is the row of
# the run's events.tsv. EEG and fNIRS events.tsv come from the same gradCPT .mat file and match row for
# row (same trial_type / response_code), so these rows can be joined to the fNIRS events used in
# ../modeling (see get_trial_modulator below).
#
# Features, per channel (fz, cz, pz, oz):
#   ERP (mountain trials only; NaN for city trials), from epochs -0.2 to 1.6 s, baseline (-0.2, 0),
#   linear detrend, 10 Hz zero-phase low-pass -- same epoching as eeg_epoch_subj_level.
#   Windows and estimators follow single_trial_ERP_estimation.ipynb:
#     N2 window 0.42-0.62 s (negative), P3 window 0.60-0.95 s (positive)
#     <ch>_<comp>_lat_woody, _amp_woody    Woody (1967) cross-correlation alignment
#     <ch>_<comp>_lat_fal50                 50% fractional-area latency (Kiesel 2008)
#     <ch>_<comp>_meanamp                   mean amplitude in the window
#     <ch>_<comp>_lat_localpeak, _amp_localpeak   Luck (2014) local peak (NaN latency if none)
#     <ch>_<comp>_lat_mlr, _amp_mlr         template + derivative regression (Hu/Mouraux)
#     <ch>_<comp>_area                      polarity-rectified area in the window (uV*s, >= 0)
#     <ch>_<comp>_area_signed               signed integral in the window (uV*s)
#     <ch>_<comp>_edge_woody / _nopeak_localpeak   QC flags
#   Notebook recommendation: P3 latency -> woody or fal50; P3 amplitude -> mlr, woody or localpeak
#   (or area/meanamp). Single-trial N2 is not recoverable at this SNR (r <= 0.11 in simulation);
#   use only meanamp/area for N2, and expect attenuated slopes for every ERP feature.
#   Templates (woody, mlr) are built from clean trials of all mountain trials (correct + incorrect)
#   of a subject and channel; every mountain trial is scored, clean or not.
#   <ch>_erp_ok   epoch peak-to-peak <= 100 uV on this channel
#   erp_ok_all    epoch kept by the existing pipeline (all present channels <= 100 uV)
#
#   EEG source: <subject>/preprocessed_EEG_and_events/*_task-gradCPT_run-XX_preproc_eeg.fif, the same files
#   run_model_cont_EEG_fNIRS.py reads via utils.eeg_preproc_subj_level.
#
#   Band power (all trials), from the continuous preprocessed EEG (0.1-45 Hz, re-referenced, ICA):
#     <ch>_<band>_<pre|post>       absolute power (uV^2) in [onset - W, onset) / [onset, onset + W)
#     <ch>_<band>_<pre|post>_rel   band power / total power over 1-30 Hz
#   bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30 Hz. PSD by Welch (2 s Hann segments, 50% overlap,
#   median average, so short artifacts do not dominate the 20 s window).
#   Windows are clipped to the recording; bp_<pre|post>_cover_s is the covered length and the power is
#   NaN if it is below half the window. bp_pre_before_task_s = seconds of the pre window that precede
#   the first trial onset of the run (pre-task EEG).
#
# Output: <data_save_path>/<subject>/features/<subject>_EEG_trial_features.csv.gz (+ _params.json)
#
# Usage:
#   python extract_EEG_trial_features.py                    # all subjects with a preprocessed dict
#   python extract_EEG_trial_features.py --subjects 723 695 --overwrite
#   python extract_EEG_trial_features.py --bp-win 20
#%% libraries
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt, welch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import data_save_path, get_eeg_preproc_dir, gen_EEG_event_tsv

#%% parameters
CH_NAMES = ['fz', 'cz', 'pz', 'oz']
RUN_KEYS = ['gradcpt1', 'gradcpt2', 'gradcpt3']

# ERP epoching: same as eeg_epoch_subj_level (stimulus-locked: -0.2 to 1.6 s, linear detrend,
# baseline (None, 0), 100 uV peak-to-peak rejection)
EPOCH_TMIN, EPOCH_TMAX = -0.2, 1.6
EPOCH_DETREND = 1
REJECT_PTP_UV = 100.0
ERP_TRIAL_TYPES = ('mnt',)  # trial_type values that get ERP features

COMPONENTS = {
    'N2': dict(sign=-1, win=(0.42, 0.62)),
    'P3': dict(sign=+1, win=(0.60, 0.95)),
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
WELCH_SEG = 2.0        # s
MIN_COVER_FRAC = 0.5


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


def est_woody(Y, t, sfreq, comp, fit_mask, max_lag=WOODY_MAX_LAG, n_iter=WOODY_N_ITER):
    """Woody alignment. The template is iterated on the fit_mask (clean) trials; every trial gets the
    lag that maximises its correlation with the final template."""
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
    return dict(amp=Y[rows, pk], lat=t[pk], edge=np.abs(lag_s) == L)


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
    return out


#%% band power
def band_power_windows(data, sfreq, onsets_s, win, side):
    """Welch band power in [onset - win, onset) (side='pre') or [onset, onset + win) ('post').
    data: (n_ch, n_samples) in uV. Returns (power dict band -> (n_trials, n_ch), rel dict, cover_s)."""
    n_ch, n_samp = data.shape
    nper = int(round(WELCH_SEG * sfreq))
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
        seg = data[:, s0:s1]
        f, P = welch(seg, fs=sfreq, nperseg=nper, noverlap=nper // 2, window='hann',
                     average='median', axis=-1)
        df = f[1] - f[0]
        tot = P[:, (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])].sum(-1) * df
        for bname, (lo, hi) in BANDS.items():
            bp = P[:, (f >= lo) & (f < hi)].sum(-1) * df
            pw[bname][i] = bp
            rel[bname][i] = bp / tot
    return pw, rel, cover


#%% per subject
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


def extract_subject(subject, bp_win=BP_WIN):
    eeg_dict = load_eeg_runs(subject)

    run_tables = []
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

        # --- band power, all trials ---
        data = raw.get_data(picks=chs) * 1e6
        onsets = ev['onset'].values
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
        run_tables.append(df)

    if not run_tables:
        return None

    # score ERP features per channel, templates pooled across runs
    for ch, parts in erp_store.items():
        X = np.concatenate([p[2] for p in parts])
        ok = np.concatenate([p[3] for p in parts])
        feats = erp_features(X, times, sfreq, ok)
        start = 0
        for tbl_i, rows, _, _ in parts:
            sl = slice(start, start + len(rows))
            tbl = run_tables[tbl_i]
            for name, val in feats.items():
                col = np.full(len(tbl), np.nan)  # NaN for city trials; QC flags stored as 0/1
                col[rows] = val[sl]
                tbl[f'{ch}_{name}'] = col
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
    return out


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

    params = dict(ch_names=CH_NAMES, epoch=(EPOCH_TMIN, EPOCH_TMAX), detrend=EPOCH_DETREND,
                  reject_ptp_uV=REJECT_PTP_UV, erp_trial_types=ERP_TRIAL_TYPES, components=COMPONENTS,
                  lowpass_hz=LP_HZ, local_peak_hw=LOCAL_PEAK_HW, fal_frac=FAL_FRAC,
                  woody_max_lag=WOODY_MAX_LAG, mlr_pad=MLR_PAD, min_template_trials=MIN_TEMPLATE_TRIALS,
                  bands=BANDS, total_band=TOTAL_BAND, bp_win=args.bp_win, welch_seg=WELCH_SEG,
                  min_cover_frac=MIN_COVER_FRAC)
    for subject in tqdm(subjects, desc='subjects'):
        out_path = get_features_path(subject)
        if os.path.exists(out_path) and not args.overwrite:
            print(f'{subject}: features exist, skipping')
            continue
        try:
            df = extract_subject(subject, bp_win=args.bp_win)
        except Exception as err:
            print(f'{subject}: failed: {err!r}')
            continue
        if df is None:
            print(f'{subject}: no gradCPT runs')
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        with open(out_path.replace('.csv.gz', '_params.json'), 'w') as f:
            json.dump(params, f, indent=1)
        n_mnt = (df.trial_type == 'mnt').sum()
        print(f'{subject}: {len(df)} trials ({n_mnt} mnt, {df.erp_ok_all.sum()} clean ERP epochs), '
              f'{df.shape[1]} columns -> {out_path}')
