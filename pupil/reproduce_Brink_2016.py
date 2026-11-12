#%% load library
"""
Reproduce van den Brink, Murphy & Nieuwenhuis (2016) "Pupil Diameter Tracks Lapses
of Attention", PLoS ONE 11(10): e0165274, on our gradCPT data.

Every step follows the paper's Methods/Results. Where our hardware or task forces a
deviation, it is marked "DEVIATION" in the code and listed in the results README.

Pipeline (per subject, per run = "block" in the paper)
------------------------------------------------------
Pupillometry (Methods > Pupillometry)
  1. Pupil diameter of the RIGHT eye (paper: right eye, EyeLink 1000 @ 1 kHz).
  2. Mark samples flagged by the manufacturer's blink detection
     (Neon: rec.blinks; Tobii: samples exported as NaN).
  3. Mark samples of poor quality: the paper iterates 200x at 1 kHz, marking every
     sample whose change from the previous sample exceeds 25 px. Because a marked
     sample is set to 0, each iteration also eats one more sample after every
     marked stretch, i.e. it removes spikes AND pads every gap by 200 ms forward.
     DEVIATION: our units are mm, not EyeLink px, so the threshold is a per-run
     dilation-speed threshold (median + 16 MAD, Kret & Sjak-Shie 2019), the
     individually-tailored analogue of the paper's threshold. The 200 ms forward
     padding is kept.
  4. Interpolate marked samples with shape-preserving piecewise cubic (PCHIP).
  5. Low-pass filter at 6 Hz.
  6. Remove phasic (task-evoked) dilations: per subject, average stimulus-locked
     waveforms for hits, misses, correct rejections, false alarms and the
     response-locked waveform; peak amplitude in 0-5 s re a 200 ms pre-event
     baseline and its latency (tmax). Stick functions (amplitude = peak) are
     convolved with the Hoeks & Levelt (1993) Erlang kernel h = s t^n e^(-n t/tmax),
     n = 10.1, unit height, and regressed out of the unsegmented pupil per run.
Sliding window (Results)
  7. 50-trial window, 15-trial step. Baseline diameter = mean pupil in the window;
     derivative = mean of consecutive-sample differences in the window.
  8. Behavior per window: false alarm rate (FA / mountain trials), proportion of hit
     RTs in the block's slowest quintile, mean hit RT, RTCV = SD(RT in window) /
     block mean RT.
Statistics (Results)
  9. Z-score every series within run. Time-on-task: slope of a straight line,
     averaged over runs per subject, t-test vs 0 (behavior one-tailed > 0, pupil
     two-tailed). Diameter-derivative correlation: Fisher z, averaged, t-test.
 10. Regression per run, behavior_z ~ pupil_z + pupil_z^2 (+ time-on-task),
     coefficients averaged over runs per subject, t-test across subjects.
 11. Window-size robustness: widths 40 s to 4 min in 8 s steps (50-300 trials in
     10-trial steps), coefficients averaged over runs and the 4 behavior measures,
     area under the curve across widths, one-tailed t-test.
 12. Fig 5: z-scored windows pooled across subjects, 30 bins sorted by pupil.
"""
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal, stats
from scipy.interpolate import PchipInterpolator
import matplotlib
import matplotlib.pyplot as plt
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
sys.path.append(os.path.join(git_path, 'modeling'))  # params_setting.py
from params_setting import project_path
from pupil_labs import neon_recording as nr

OUT_DIR = os.path.join(project_path, 'derivatives', 'eye_tracking', 'reproduce_Brink_2016_results')
os.makedirs(OUT_DIR, exist_ok=True)

subject_list = sorted(d for d in os.listdir(project_path) if d.startswith('sub-'))

PARAMS = dict(
    eye='right',            # paper: right eye. 'left' or 'mean' (binocular) for sensitivity
    fs=100.0,               # Hz, common uniform grid for Neon (~94-125 Hz) and Tobii (100 Hz)
    f_lowpass=6.0,          # Hz (paper)
    spike_mad_n=16,         # dilation-speed threshold = median + n*MAD
    pad_after_s=0.2,        # 200 iterations at 1 kHz = 200 ms forward padding (paper)
    max_missing=0.30,       # exclude a run when > 30% of the task is missing
    epoch_pre=0.2,          # s, pre-event baseline (paper)
    epoch_post=5.0,         # s, peak search window (paper)
    hoeks_n=10.1,           # Hoeks & Levelt number of layers (paper)
    soa=0.8,                # s, gradCPT SOA
    win_trials=50,          # 40 s window (paper)
    step_trials=15,         # (paper)
    n_bins=30,              # Fig 5 (paper)
    auc_win_trials=np.arange(50, 301, 10),   # 40 s .. 4 min in 8 s steps (paper)
)

BEH_KEYS   = ['fa_rate', 'slow_q', 'mean_rt', 'rtcv']
BEH_LABELS = {'fa_rate': 'False alarms', 'slow_q': 'Slow quintile',
              'mean_rt': 'Response time', 'rtcv': 'RTCV', 'miss_rate': 'Misses'}
PUP_KEYS   = ['diam', 'deriv']
PUP_LABELS = {'diam': 'Baseline diameter', 'deriv': 'Diameter derivative'}


#%% loading
def find_physio_file(subj, run_id):
    nirs_dir = os.path.join(project_path, subj, 'nirs')
    for v in ['_latest', '_20260423', '']:   # _latest == _20260423 (byte-identical)
        f = os.path.join(nirs_dir, f'{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio{v}.tsv')
        if os.path.isfile(f):
            return f
    return None


_neon_cache = {}
def find_neon_recording(subj, t_ns):
    """Return the Neon recording whose [start, stop] contains t_ns.

    Matching by time, not by run index: several subjects have more Neon recordings
    than gradCPT runs (resting state, restarts), so neon_dirs[run_id-1] can point to
    the wrong recording and its blinks never overlap the run.
    """
    neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
    if not os.path.isdir(neon_dir):
        return None
    if subj not in _neon_cache:
        recs = []
        for d in sorted(os.listdir(neon_dir)):
            if not re.match(r'\d{4}-', d):
                continue
            try:
                recs.append(nr.open(os.path.join(neon_dir, d)))
            except Exception:
                pass
        _neon_cache[subj] = recs
    for rec in _neon_cache[subj]:
        try:
            if rec.start_time <= t_ns <= rec.stop_time:
                return rec
        except Exception:
            continue
    return None


def load_run(subj, run_id, eye='right'):
    """Load raw pupil (+ manufacturer blink marks) and events for one run."""
    physio = find_physio_file(subj, run_id)
    ev_file = os.path.join(project_path, subj, 'nirs', f'{subj}_task-gradCPT_run-{run_id:02d}_events.tsv')
    if physio is None or not os.path.isfile(ev_file):
        return None
    d  = pd.read_csv(physio, sep='\t')
    ev = pd.read_csv(ev_file, sep='\t').sort_values('onset').reset_index(drop=True)
    d  = d.sort_values('timestamps').drop_duplicates('timestamps').reset_index(drop=True)
    t  = d['timestamps'].values.astype(float)
    L  = d['eyeleft_pupilDiameter'].values.astype(float)
    R  = d['eyeright_pupilDiameter'].values.astype(float)
    x  = {'right': R, 'left': L, 'mean': (L + R) / 2}[eye].copy()
    x[~(x > 0)] = np.nan
    device = 'neon' if 'time' in d.columns else 'tobii'
    blink_info = 'nan-coded' if device == 'tobii' else 'none'
    if device == 'neon':
        t_ns = d['time'].values
        rec = find_neon_recording(subj, int(np.median(t_ns)))
        if rec is not None:
            try:
                b = rec.blinks
                starts, stops = np.asarray(b['start_time']), np.asarray(b['stop_time'])
                i0 = np.searchsorted(t_ns, starts, 'left')
                i1 = np.searchsorted(t_ns, stops, 'right')
                for a, z in zip(i0, i1):
                    x[a:z] = np.nan
                blink_info = f'neon ({len(starts)} blinks)'
            except Exception:
                blink_info = 'neon blinks unavailable'
    return dict(t=t, x=x, ev=ev, device=device, blink_info=blink_info,
                physio=os.path.basename(physio))


#%% preprocessing (Methods > Pupillometry)
def mark_artifacts(t, x, n_mad=16, pad_after_s=0.2):
    """Speed-based spike rejection + forward padding of every invalid stretch."""
    bad = np.isnan(x)
    v = np.where(~bad)[0]
    if len(v) < 3:
        return np.ones_like(bad)
    spd_seg = np.abs(np.diff(x[v]) / np.diff(t[v]))
    spd = np.fmax(np.r_[np.nan, spd_seg], np.r_[spd_seg, np.nan])   # max of backward/forward speed
    med = np.nanmedian(spd)
    thr = med + n_mad * np.nanmedian(np.abs(spd - med))
    bad[v[spd > thr]] = True
    # forward padding: each marked stretch eats pad_after_s seconds of the following data
    if pad_after_s > 0 and bad.any():
        ends = np.where(bad & ~np.r_[bad[1:], False])[0]          # last sample of each bad run
        for e in ends:
            j = np.searchsorted(t, t[e] + pad_after_s, 'right')
            bad[e:j] = True
    return bad


def preprocess_run(raw, P):
    """Steps 1-5: artifact marking, PCHIP interpolation onto a uniform grid, 6 Hz low-pass."""
    t, x, ev = raw['t'], raw['x'], raw['ev']
    bad = mark_artifacts(t, x, P['spike_mad_n'], P['pad_after_s'])
    task0, task1 = ev['onset'].iloc[0], ev['onset'].iloc[-1] + P['soa']
    grid = np.arange(task0 - 2.0, task1 + P['epoch_post'] + 1.0, 1 / P['fs'])
    # samples outside the recording are missing (never extrapolated)
    outside = (grid < t[0]) | (grid > t[-1])
    # grid points whose nearest raw sample is marked
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.where(np.abs(t[idx - 1] - grid) < np.abs(t[idx] - grid), idx - 1, idx)
    miss = bad[nearest] | outside
    in_task = (grid >= task0) & (grid < task1)
    frac_missing = miss[in_task].mean()
    frac_outside = outside[in_task].mean()
    good = ~bad
    if good.sum() < 10:
        return None
    y = PchipInterpolator(t[good], x[good], extrapolate=False)(grid)
    # constant fill beyond the first/last valid sample, only so the filter has no NaN
    fv = np.where(~np.isnan(y))[0]
    y[:fv[0]] = y[fv[0]]
    y[fv[-1] + 1:] = y[fv[-1]]
    sos = signal.butter(4, P['f_lowpass'], btype='low', fs=P['fs'], output='sos')
    y = signal.sosfiltfilt(sos, y)
    return dict(grid=grid, y=y, miss=miss, outside=outside, in_task=in_task,
                frac_missing=frac_missing, frac_outside=frac_outside)


def event_table(ev, soa=0.8):
    """Event onsets per type. response_code: 1 hit, -1 miss, 0 CR, -2 FA."""
    city, mnt = ev['trial_type'] == 'city', ev['trial_type'] == 'mnt'
    rc = ev['response_code']
    resp = ev['reaction_time'] > 0
    return {
        'Hit':              ev.loc[city & (rc == 1), 'onset'].values,
        'Miss':             ev.loc[city & (rc == -1), 'onset'].values,
        'Correct rejection': ev.loc[mnt & (rc == 0), 'onset'].values,
        'False alarm':      ev.loc[mnt & (rc == -2), 'onset'].values,
        'Response':         (ev.loc[resp, 'onset'] + ev.loc[resp, 'reaction_time']).values,
    }
EVENT_TYPES = ['Hit', 'Miss', 'Correct rejection', 'False alarm', 'Response']


def epochs(grid, y, miss_outside, onsets, pre, post, fs):
    """Baseline-corrected epochs [-pre, post] around onsets (skips epochs outside the recording)."""
    n_pre, n_post = int(round(pre * fs)), int(round(post * fs))
    out = []
    for o in onsets:
        i = int(round((o - grid[0]) * fs))
        if i - n_pre < 0 or i + n_post >= len(grid) or miss_outside[i - n_pre:i + n_post + 1].any():
            continue
        seg = y[i - n_pre:i + n_post + 1]
        out.append(seg - seg[:n_pre].mean())
    return np.array(out)


def hoeks_kernel(tmax, n, fs):
    tk = np.arange(0, max(3 * tmax, 1.0), 1 / fs)
    h = tk ** n * np.exp(-n * tk / tmax)
    return h / h.max()                  # s scales h to unit height


def phasic_params(runs, P):
    """Per subject: peak amplitude and latency (tmax) of the average waveform per event type."""
    out = {}
    t_ep = np.arange(-int(round(P['epoch_pre'] * P['fs'])), int(round(P['epoch_post'] * P['fs'])) + 1) / P['fs']
    for et in EVENT_TYPES:
        eps = [epochs(r['pp']['grid'], r['pp']['y'], r['pp']['outside'], r['events'][et],
                      P['epoch_pre'], P['epoch_post'], P['fs']) for r in runs]
        eps = [e for e in eps if len(e)]
        if not eps:
            continue
        avg = np.concatenate(eps).mean(0)
        post = t_ep > 0
        k = np.argmax(avg[post])
        out[et] = dict(amp=avg[post][k], tmax=t_ep[post][k], avg=avg, t=t_ep,
                       n=sum(len(e) for e in eps))
    return out


def remove_phasic(r, params, P):
    """Step 6: regress Hoeks-Levelt-convolved stick functions out of the pupil trace."""
    pp = r['pp']
    grid, y = pp['grid'], pp['y']
    cols, names = [], []
    for et in EVENT_TYPES:
        if et not in params or len(r['events'][et]) == 0:
            continue
        stick = np.zeros(len(grid))
        idx = np.round((r['events'][et] - grid[0]) * P['fs']).astype(int)
        idx = idx[(idx >= 0) & (idx < len(grid))]
        np.add.at(stick, idx, params[et]['amp'])
        h = hoeks_kernel(max(params[et]['tmax'], 1 / P['fs']), P['hoeks_n'], P['fs'])
        cols.append(np.convolve(stick, h)[:len(grid)])
        names.append(et)
    X = np.column_stack(cols + [np.ones(len(grid))])
    fit = ~pp['outside']
    beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
    phasic = X[:, :-1] @ beta[:-1]
    resid = y - phasic
    m = fit & pp['in_task']
    r2 = 1 - np.sum((y[m] - X[m] @ beta) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
    return resid, phasic, r2, dict(zip(names, beta[:-1]))


#%% sliding window (Results)
def window_measures(r, win_trials, step_trials, P, pupil_key='resid'):
    """Step 7-8: per-window pupil and behavioral measures for one run."""
    ev, pp = r['ev'], r['pp']
    grid, y = pp['grid'], r[pupil_key]
    city, mnt, rc = ev['trial_type'] == 'city', ev['trial_type'] == 'mnt', ev['response_code']
    hit_rt_block = ev.loc[city & (rc == 1), 'reaction_time'].values
    q80 = np.percentile(hit_rt_block, 80)
    rt_block_mean = hit_rt_block.mean()
    t0 = ev['onset'].iloc[0]
    rows = []
    for s in range(0, len(ev) - win_trials + 1, step_trials):
        w = ev.iloc[s:s + win_trials]
        a, b = w['onset'].iloc[0], w['onset'].iloc[-1] + P['soa']
        m = (grid >= a) & (grid < b)
        if pp['outside'][m].any():
            diam = deriv = np.nan
        else:
            diam, deriv = y[m].mean(), np.diff(y[m]).mean()
        wc, wm, wrc = w['trial_type'] == 'city', w['trial_type'] == 'mnt', w['response_code']
        rts = w.loc[wc & (wrc == 1), 'reaction_time'].values
        n_mnt, n_city = wm.sum(), wc.sum()
        rows.append(dict(
            t_min=((a + b) / 2 - t0) / 60,
            diam=diam, deriv=deriv,
            fa_rate=((wm & (wrc == -2)).sum() / n_mnt) if n_mnt else np.nan,
            slow_q=np.mean(rts >= q80) if len(rts) else np.nan,
            mean_rt=rts.mean() if len(rts) else np.nan,
            rtcv=(rts.std() / rt_block_mean) if len(rts) > 1 else np.nan,
            miss_rate=((wc & (wrc == -1)).sum() / n_city) if n_city else np.nan,
            vtc=w['VTC'].mean() if 'VTC' in w else np.nan,
        ))
    return pd.DataFrame(rows)


def zscore_nan(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / sd


#%% build dataset
def build_dataset(subject_list, P, verbose=True):
    """Preprocess every run, remove phasic responses, and return the run dicts + a QC table."""
    runs_all, qc = [], []
    for subj in subject_list:
        runs = []
        for run_id in range(1, 6):
            raw = load_run(subj, run_id, P['eye'])
            if raw is None:
                continue
            pp = preprocess_run(raw, P)
            row = dict(subj=subj, run=run_id, device=raw['device'], blinks=raw['blink_info'],
                       physio=raw['physio'])
            if pp is None:
                qc.append({**row, 'included': False, 'reason': 'no valid pupil'})
                continue
            row.update(frac_missing=pp['frac_missing'], frac_outside=pp['frac_outside'])
            if pp['frac_missing'] > P['max_missing']:
                qc.append({**row, 'included': False,
                           'reason': f"{pp['frac_missing']:.0%} missing > {P['max_missing']:.0%}"})
                continue
            r = dict(subj=subj, run=run_id, ev=raw['ev'], pp=pp, events=event_table(raw['ev'], P['soa']),
                     device=raw['device'])
            runs.append(r)
            qc.append({**row, 'included': True, 'reason': ''})
        if not runs:
            continue
        params = phasic_params(runs, P)
        for r in runs:
            r['resid'], r['phasic'], r['r2_phasic'], r['beta_phasic'] = remove_phasic(r, params, P)
            r['phasic_params'] = params
            r['filtered'] = r['pp']['y']
            for q in qc:
                if q['subj'] == subj and q['run'] == r['run']:
                    q['r2_phasic'] = r['r2_phasic']
        runs_all.extend(runs)
        if verbose:
            print(f"{subj}: {len(runs)} run(s) | " + ', '.join(
                f"run-{r['run']:02d} miss={r['pp']['frac_missing']:.1%} phasicR2={r['r2_phasic']:.1%}" for r in runs),
                flush=True)
    return runs_all, pd.DataFrame(qc)


def windows_table(runs, win_trials, step_trials, P, pupil_key='resid'):
    """Concatenate window tables and z-score every series within run."""
    dfs = []
    for r in runs:
        w = window_measures(r, win_trials, step_trials, P, pupil_key)
        w.insert(0, 'run', r['run'])
        w.insert(0, 'subj', r['subj'])
        dfs.append(w)
    W = pd.concat(dfs, ignore_index=True)
    for c in PUP_KEYS + BEH_KEYS + ['miss_rate', 'vtc']:
        W[c + '_z'] = W.groupby(['subj', 'run'])[c].transform(zscore_nan)
    return W


#%% statistics
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def ttest(vals, alternative='two-sided'):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return dict(n=len(vals), mean=np.nan, t=np.nan, p=np.nan, d=np.nan)
    t, p = stats.ttest_1samp(vals, 0, alternative=alternative)
    return dict(n=len(vals), mean=vals.mean(), sem=stats.sem(vals), t=t, p=p, d=vals.mean() / vals.std(ddof=1))


def per_subject(W, fn):
    """Apply fn to every run's window table, average the result over runs per subject."""
    res = {}
    for (subj, run), g in W.groupby(['subj', 'run']):
        v = fn(g)
        if v is not None:
            res.setdefault(subj, []).append(v)
    return {s: np.nanmean(np.array(v, float), axis=0) for s, v in res.items()}


def tot_slope(col):
    def fn(g):
        ok = g[[col, 't_min']].dropna()
        if len(ok) < 3:
            return None
        return np.polyfit(ok['t_min'], ok[col], 1)[0]
    return fn


def corr_fisher(g):
    ok = g[['diam_z', 'deriv_z']].dropna()
    if len(ok) < 4:
        return None
    return np.arctanh(np.corrcoef(ok['diam_z'], ok['deriv_z'])[0, 1])


def regress(pcol, bcol, quad=True, tot=False, min_n=6):
    """behavior_z ~ 1 + pupil_z (+ pupil_z^2) (+ time-on-task); returns [lin, quad]."""
    def fn(g):
        ok = g[[pcol, bcol, 't_min']].dropna()
        if len(ok) < min_n:
            return None
        p = ok[pcol].values
        X = [np.ones(len(p)), p]
        if quad:
            X.append(p ** 2)
        if tot:
            X.append(ok['t_min'].values)
        beta, *_ = np.linalg.lstsq(np.column_stack(X), ok[bcol].values, rcond=None)
        return [beta[1], beta[2] if quad else np.nan]
    return fn


def regression_coefs(W, tot):
    """{(pupil, behavior): {subj: [lin, quad]}}; derivative models also carry a quadratic term (paper)."""
    return {(pk, bk): per_subject(W, regress(pk + '_z', bk + '_z', quad=True, tot=tot))
            for pk in PUP_KEYS for bk in BEH_KEYS}


#%% paper values (t(27), N = 28); sign follows the effect direction
PAPER = {
    # time-on-task (Fig 1b one-tailed > 0; Fig 3 two-tailed)
    ('tot', 'fa_rate'):   (1.74, 0.047), ('tot', 'slow_q'): (3.31, 0.001),
    ('tot', 'mean_rt'):   (2.47, 0.010), ('tot', 'rtcv'):   (3.06, 0.003),
    ('tot', 'miss_rate'): (3.31, 0.001),
    ('tot', 'diam'):      (-8.10, 0.001), ('tot', 'deriv'): (4.40, 0.001),
    ('corr', 'diam-deriv'): (-2.48, 0.020),
    # Fig 4 (no time-on-task)
    ('noTOT', 'diam', 'lin', 'fa_rate'): (-2.28, 0.02),  ('noTOT', 'diam', 'lin', 'slow_q'): (-2.60, 0.008),
    ('noTOT', 'diam', 'lin', 'mean_rt'): (-2.38, 0.012), ('noTOT', 'diam', 'lin', 'rtcv'):   (-2.27, 0.016),
    ('noTOT', 'diam', 'quad', 'fa_rate'): (1.99, 0.029),
    ('noTOT', 'deriv', 'lin', 'fa_rate'): (3.71, 0.001), ('noTOT', 'deriv', 'lin', 'slow_q'): (3.10, 0.005),
    ('noTOT', 'deriv', 'lin', 'mean_rt'): (2.10, 0.046), ('noTOT', 'deriv', 'lin', 'rtcv'):   (3.11, 0.005),
    # Fig 6 (time-on-task regressor)
    ('TOT', 'diam', 'quad', 'fa_rate'): (1.99, 0.028), ('TOT', 'diam', 'quad', 'slow_q'): (1.45, 0.08),
    ('TOT', 'diam', 'quad', 'mean_rt'): (2.06, 0.025), ('TOT', 'diam', 'quad', 'rtcv'):   (2.79, 0.005),
    ('TOT', 'deriv', 'lin', 'fa_rate'): (3.09, 0.005), ('TOT', 'deriv', 'lin', 'slow_q'): (2.13, 0.041),
    ('TOT', 'deriv', 'lin', 'mean_rt'): (0.93, 0.360), ('TOT', 'deriv', 'lin', 'rtcv'):   (3.07, 0.005),
    # window-size AUC
    ('AUC-noTOT', 'diam', 'lin'): (-2.62, 0.007), ('AUC-noTOT', 'deriv', 'lin'): (4.05, 0.001),
    ('AUC-noTOT', 'diam', 'quad'): (0.09, 0.47),
    ('AUC-TOT', 'diam', 'lin'): (0.29, 0.39), ('AUC-TOT', 'diam', 'quad'): (3.08, 0.002),
    ('AUC-TOT', 'deriv', 'lin'): (3.08, 0.002),
}
N_PAPER = 28


def summary_row(key, label, res, alternative):
    pt, pp_ = PAPER.get(key, (np.nan, np.nan))
    return dict(test=label, tail=alternative, N=res['n'], mean=res['mean'], t=res['t'], p=res['p'],
                d=res['d'], paper_t=pt, paper_p=pp_, paper_d=pt / np.sqrt(N_PAPER) if np.isfinite(pt) else np.nan,
                same_direction=(np.sign(res['t']) == np.sign(pt)) if np.isfinite(pt) else np.nan,
                sig_ours=res['p'] < 0.05, sig_paper=(pp_ < 0.05) if np.isfinite(pp_) else np.nan)


#%% plotting helpers
def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def plot_coef_bars(coefs, tests, title, fname):
    """Fig 4 / Fig 6 layout: diameter (linear + quadratic) and derivative (linear)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    x = np.arange(len(BEH_KEYS))
    for col, pk in enumerate(PUP_KEYS):
        for row, (ci, ct) in enumerate([(0, 'lin'), (1, 'quad')]):
            ax = axes[row, col]
            if pk == 'deriv' and ct == 'quad':
                ax.set_visible(False)
                continue
            m, e, s = [], [], []
            for bk in BEH_KEYS:
                v = np.array([c[ci] for c in coefs[(pk, bk)].values()])
                m.append(v.mean()); e.append(stats.sem(v)); s.append(stars(tests[(pk, ct, bk)]['p']))
            ax.bar(x, m, 0.6, yerr=e, capsize=4, color='0.6' if ct == 'lin' else 'white',
                   edgecolor='0.4', hatch='' if ct == 'lin' else '///', zorder=3)
            for xi, mi, ei, si in zip(x, m, e, s):
                ax.text(xi, mi + np.sign(mi or 1) * (ei + 0.01), si, ha='center',
                        va='bottom' if mi >= 0 else 'top', fontsize=11)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([BEH_LABELS[b] for b in BEH_KEYS], fontsize=8)
            ax.set_ylabel(f"{'Linear' if ct == 'lin' else 'Quadratic'} β")
            ax.set_title(PUP_LABELS[pk])
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, fname)
    return fig


def plot_fig5(W, n_subj, n_bins, fname):
    """Fig 5: pooled z-scored windows, 30 bins sorted by pupil, with and without time-on-task."""
    colors = {'fa_rate': '#1f3fbf', 'slow_q': '#5b2a99', 'mean_rt': '#9c1f6e', 'rtcv': '#d7191c'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    Wr = W.copy()
    # "without time-on-task": remove the linear time trend within each run from pupil and behavior
    for c in [k + '_z' for k in PUP_KEYS + BEH_KEYS]:
        def detr(g, c=c):
            ok = g[c].notna()
            out = g[c].copy()
            if ok.sum() > 2:
                out[ok] = g.loc[ok, c] - np.polyval(np.polyfit(g.loc[ok, 't_min'], g.loc[ok, c], 1), g.loc[ok, 't_min'])
            return out
        Wr[c] = W.groupby(['subj', 'run'], group_keys=False).apply(detr)
    for col, (D, ttl) in enumerate([(W, 'With time-on-task'), (Wr, 'Without time-on-task')]):
        for row, pk in enumerate(PUP_KEYS):
            ax = axes[row, col]
            for bk in BEH_KEYS:
                ok = D[[pk + '_z', bk + '_z']].dropna()
                o = np.argsort(ok[pk + '_z'].values)
                px = ok[pk + '_z'].values[o]; by = ok[bk + '_z'].values[o]
                bins = np.array_split(np.arange(len(px)), n_bins)
                bx = np.array([px[b].mean() for b in bins]); bb = np.array([by[b].mean() for b in bins])
                ax.scatter(bx, bb, s=12, color=colors[bk], alpha=0.8)
                order = 2 if (pk == 'diam' and col == 1) else 1
                xf = np.linspace(bx.min(), bx.max(), 100)
                ax.plot(xf, np.polyval(np.polyfit(bx, bb, order), xf), color=colors[bk], lw=1.8,
                        label=BEH_LABELS[bk])
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.set_xlabel(f"{PUP_LABELS[pk]} (z)")
            ax.set_ylabel('Behavior (z)')
            ax.set_title(f"{ttl}: {PUP_LABELS[pk]}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Fig 5 reproduction (pooled windows, {n_bins} bins, N={n_subj})')
    fig.tight_layout()
    savefig(fig, fname)
    return fig


def plot_timecourses(W, n_subj, fname):
    """Fig 1b (behavior, raw units) and Fig 3a/b (pupil, z) across time-on-task."""
    keys = BEH_KEYS + ['miss_rate', 'diam_z', 'deriv_z']
    labels = {**BEH_LABELS, 'diam_z': 'Baseline diameter (z)', 'deriv_z': 'Diameter derivative (z)'}
    scale = {'fa_rate': 100, 'slow_q': 100, 'miss_rate': 100}
    units = {'fa_rate': 'Lapse rate (%)', 'slow_q': 'Slowest 1/5th of trials (%)', 'mean_rt': 'RT (s)',
             'rtcv': 'RT SD / block mean RT', 'miss_rate': 'Miss rate (%)', 'diam_z': 'z', 'deriv_z': 'z'}
    W = W.copy()
    W['win'] = W.groupby(['subj', 'run']).cumcount()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, k in zip(axes.flat, keys):
        subj_mean = W.groupby(['subj', 'win'])[[k, 't_min']].mean().reset_index()
        piv = subj_mean.pivot(index='subj', columns='win', values=k) * scale.get(k, 1)
        tm = subj_mean.groupby('win')['t_min'].mean()
        m, e = piv.mean(), piv.sem()
        ax.fill_between(tm, m - e, m + e, color='0.8')
        ax.plot(tm, m, 'k', lw=1.2)
        ax.set_title(labels[k]); ax.set_xlabel('Time (min)'); ax.set_ylabel(units[k])
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(f'Time-on-task (Fig 1b and Fig 3a,b reproduction), N={n_subj}, mean ± SEM across subjects')
    fig.tight_layout()
    savefig(fig, fname)
    return fig


def plot_phasic(runs, fname):
    """Fig 2: trial-averaged empirical (black) vs modeled (red) phasic responses."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4), sharey=True)
    P = PARAMS
    for ax, et in zip(axes, EVENT_TYPES):
        emp, mod = [], []
        for subj in sorted({r['subj'] for r in runs}):
            sr = [r for r in runs if r['subj'] == subj]
            e = [epochs(r['pp']['grid'], r['filtered'], r['pp']['outside'], r['events'][et],
                        P['epoch_pre'], P['epoch_post'], P['fs']) for r in sr]
            mm = [epochs(r['pp']['grid'], r['phasic'], r['pp']['outside'], r['events'][et],
                         P['epoch_pre'], P['epoch_post'], P['fs']) for r in sr]
            e = [x for x in e if len(x)]; mm = [x for x in mm if len(x)]
            if e:
                emp.append(np.concatenate(e).mean(0)); mod.append(np.concatenate(mm).mean(0))
        t_ep = np.arange(emp[0].shape[0]) / P['fs'] - P['epoch_pre']
        for arr, c, lab in [(np.array(emp), 'k', 'Data'), (np.array(mod), 'r', 'Model')]:
            m, s = arr.mean(0), stats.sem(arr, 0)
            ax.fill_between(t_ep, m - s, m + s, color=c, alpha=0.15)
            ax.plot(t_ep, m, c, lw=1.2, label=lab)
        ax.axvline(0, color='0.5', ls='--', lw=0.8)
        ax.set_title(f'{et} (N={len(emp)})'); ax.set_xlabel('Time (s)')
    axes[0].set_ylabel('Pupil (mm, baseline-corrected)')
    axes[-1].legend()
    fig.suptitle('Fig 2 reproduction: phasic pupil responses')
    fig.tight_layout()
    savefig(fig, fname)
    return fig


def plot_fig3c(W, fname):
    ok = W[['diam_z', 'deriv_z']].dropna()
    o = np.argsort(ok['deriv_z'].values)
    dx, py = ok['deriv_z'].values[o], ok['diam_z'].values[o]
    bins = np.array_split(np.arange(len(dx)), 10)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([dx[b].mean() for b in bins], [py[b].mean() for b in bins], 'k-o', ms=3)
    ax.set_xlabel('Derivative (z)'); ax.set_ylabel('Pupil diameter (z)')
    ax.set_title('Fig 3c: diameter as a function of derivative'); ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, fname)
    return fig


#%% full reproduction
def run_reproduction(P, tag=''):
    runs, qc = build_dataset(subject_list, P)
    qc.to_csv(os.path.join(OUT_DIR, f'qc_runs{tag}.csv'), index=False)
    W = windows_table(runs, P['win_trials'], P['step_trials'], P)
    W.to_csv(os.path.join(OUT_DIR, f'windows{tag}.csv'), index=False)
    subjects = sorted(W['subj'].unique())
    n_subj = len(subjects)
    rows = []

    # --- time-on-task (Fig 1b, Fig 3a,b) ---
    for k in BEH_KEYS + ['miss_rate']:
        res = ttest(list(per_subject(W, tot_slope(k + '_z')).values()), 'greater')
        rows.append(summary_row(('tot', k), f'ToT slope: {BEH_LABELS[k]}', res, 'greater'))
    for k in PUP_KEYS:
        res = ttest(list(per_subject(W, tot_slope(k + '_z')).values()), 'two-sided')
        rows.append(summary_row(('tot', k), f'ToT slope: {PUP_LABELS[k]}', res, 'two-sided'))
    # --- diameter vs derivative (Fig 3c) ---
    zc = np.array(list(per_subject(W, corr_fisher).values()))
    res = ttest(zc, 'two-sided')
    rows.append(summary_row(('corr', 'diam-deriv'), 'Corr(diameter, derivative), Fisher z', res, 'two-sided'))
    mean_r = np.tanh(np.nanmean(zc))

    # --- Fig 4 (no ToT) and Fig 6 (ToT) ---
    coef_sets, test_sets = {}, {}
    for tot, tag_m in [(False, 'noTOT'), (True, 'TOT')]:
        coefs = regression_coefs(W, tot)
        tests = {}
        for pk in PUP_KEYS:
            for bk in BEH_KEYS:
                v = np.array(list(coefs[(pk, bk)].values()))
                if pk == 'diam':
                    lin_alt = 'less' if not tot else 'two-sided'
                    tests[(pk, 'lin', bk)] = ttest(v[:, 0], lin_alt)
                    tests[(pk, 'quad', bk)] = ttest(v[:, 1], 'greater')
                    rows.append(summary_row((tag_m, pk, 'lin', bk), f'[{tag_m}] diameter linear: {BEH_LABELS[bk]}',
                                            tests[(pk, 'lin', bk)], lin_alt))
                    rows.append(summary_row((tag_m, pk, 'quad', bk), f'[{tag_m}] diameter quadratic: {BEH_LABELS[bk]}',
                                            tests[(pk, 'quad', bk)], 'greater'))
                else:
                    tests[(pk, 'lin', bk)] = ttest(v[:, 0], 'two-sided')
                    rows.append(summary_row((tag_m, pk, 'lin', bk), f'[{tag_m}] derivative linear: {BEH_LABELS[bk]}',
                                            tests[(pk, 'lin', bk)], 'two-sided'))
        coef_sets[tag_m], test_sets[tag_m] = coefs, tests

    # --- window-size robustness (AUC) ---
    auc_curves = {}
    for tot, tag_m in [(False, 'AUC-noTOT'), (True, 'AUC-TOT')]:
        curves = {('diam', 'lin'): {}, ('diam', 'quad'): {}, ('deriv', 'lin'): {}}
        for wt in P['auc_win_trials']:
            Ww = windows_table(runs, int(wt), P['step_trials'], P)
            co = regression_coefs(Ww, tot)
            for s in subjects:
                for (pk, ct) in curves:
                    ci = 0 if ct == 'lin' else 1
                    vals = [co[(pk, bk)][s][ci] for bk in BEH_KEYS if s in co[(pk, bk)]]
                    curves[(pk, ct)].setdefault(s, []).append(np.nanmean(vals) if vals else np.nan)
        win_min = P['auc_win_trials'] * P['soa'] / 60
        for (pk, ct), d in curves.items():
            auc = [_trapz(v, win_min) for v in d.values() if np.all(np.isfinite(v))]
            alt = {('diam', 'lin'): 'less' if tag_m == 'AUC-noTOT' else 'two-sided',
                   ('diam', 'quad'): 'greater', ('deriv', 'lin'): 'greater'}[(pk, ct)]
            rows.append(summary_row((tag_m, pk, ct), f'[{tag_m}] {PUP_LABELS[pk]} {ct}', ttest(auc, alt), alt))
        auc_curves[tag_m] = curves

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, f'summary_vs_paper{tag}.csv'), index=False)

    # --- figures ---
    if not tag:
        plot_phasic(runs, 'fig2_phasic.png')
        plot_timecourses(W, n_subj, 'fig1b_fig3ab_time_on_task.png')
        plot_fig3c(W, 'fig3c_diam_vs_deriv.png')
        plot_coef_bars(coef_sets['noTOT'], test_sets['noTOT'], f'Fig 4 reproduction (no time-on-task), N={n_subj}',
                       'fig4_coefs_noTOT.png')
        plot_coef_bars(coef_sets['TOT'], test_sets['TOT'], f'Fig 6 reproduction (time-on-task regressor), N={n_subj}',
                       'fig6_coefs_TOT.png')
        plot_fig5(W, n_subj, P['n_bins'], 'fig5_binned.png')
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, tag_m in zip(axes, ['AUC-noTOT', 'AUC-TOT']):
            for (pk, ct), d in auc_curves[tag_m].items():
                arr = np.array([v for v in d.values() if np.all(np.isfinite(v))])
                ax.errorbar(P['auc_win_trials'] * P['soa'], arr.mean(0), stats.sem(arr, 0), capsize=2,
                            label=f'{PUP_LABELS[pk]} {ct}')
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlabel('Window width (s)'); ax.set_ylabel('β (mean over runs and 4 measures)')
            ax.set_title(tag_m); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, 'window_size_auc.png')

    meta = dict(n_subj=n_subj, subjects=subjects, n_runs=W.groupby(['subj', 'run']).ngroups,
                mean_r=mean_r, qc=qc, runs=runs,
                r2_phasic=pd.Series({r['subj']: r['r2_phasic'] for r in runs}).groupby(level=0).mean(),
                frac_interp=pd.Series({(r['subj'], r['run']): r['pp']['frac_missing'] for r in runs}))
    return summary, W, meta


def print_summary(summary, meta):
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 200)
    print(f"\nN subjects = {meta['n_subj']}, runs = {meta['n_runs']}  (paper: N = 28, 3 blocks x 600 trials)")
    fi = meta['frac_interp']
    print(f"Interpolated: {fi.mean():.1%} (SD {fi.std():.1%}, min/max {fi.min():.1%}/{fi.max():.1%})"
          "  (paper: 6.8%, SD 4.6, 0.2/18.7)")
    r2 = meta['r2_phasic']
    print(f"Variance explained by phasic model: {r2.mean():.1%} (SD {r2.std():.1%})  (paper: ~8%, SD 4%)")
    print(f"Mean corr(diameter, derivative) = {meta['mean_r']:.3f}  (paper: -0.12)")
    cols = ['test', 'tail', 'N', 'mean', 't', 'p', 'd', 'paper_t', 'paper_p', 'paper_d', 'same_direction', 'sig_ours', 'sig_paper']
    print(summary[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))


#%% Run the reproduction (paper settings, right eye)
if __name__ == '__main__' or 'get_ipython' in globals():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    summary, W, meta = run_reproduction(PARAMS)
    print_summary(summary, meta)

#%% Sensitivity: binocular average instead of the right eye
if __name__ == '__main__' or 'get_ipython' in globals():
    summary_bino, W_bino, meta_bino = run_reproduction({**PARAMS, 'eye': 'mean'}, tag='_binocular')
    print_summary(summary_bino, meta_bino)
    plt.show()
