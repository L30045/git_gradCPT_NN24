"""Shared pieces of the mnt-correct HRF simulation (simulate_mnt-correct_HRF.py, cmp_HRF_simulation.py).

The HRF is a double gamma  g(t) = Gamma(t; a1) - ratio * Gamma(t; a2)  (unit scale, so the peak and undershoot
delays are about a1 - 1 and a2 - 1 s), scaled to a peak of 1. SPM's canonical HRF is a1 = 6, a2 = 16, ratio = 1/6.
Optionally the HRF is forced back to 0 at t_end: it is multiplied by a cosine taper that falls from 1 at
t_end - taper to 0 at t_end, and is 0 afterwards (t_end = None: no taper).
"""
import numpy as np
from scipy.stats import gamma

CANONICAL = {'a1': 6.0, 'a2': 16.0, 'ratio': 1 / 6}
_tt = np.linspace(0, 40, 40001)


def _raw(tau, a1, a2, ratio):
    return gamma.pdf(tau, a1) - ratio * gamma.pdf(tau, a2)


def end_taper(tau, t_end=None, taper=5.0):
    """1 before t_end - taper, cosine ramp to 0 at t_end, 0 after (all ones if t_end is None)."""
    tau = np.asarray(tau, dtype=float)
    if t_end is None:
        return np.ones_like(tau)
    x = np.clip((tau - (t_end - taper)) / taper, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def double_gamma_hrf(tau, a1=CANONICAL['a1'], a2=CANONICAL['a2'], ratio=CANONICAL['ratio'], t_end=None, taper=5.0):
    """Double gamma (times the end taper if t_end is set) scaled to a peak of 1; 0 for tau <= 0 and tau >= t_end."""
    tau = np.asarray(tau, dtype=float)
    g_max = (_raw(_tt, a1, a2, ratio) * end_taper(_tt, t_end, taper)).max()
    g = _raw(np.clip(tau, 0, None), a1, a2, ratio) * end_taper(tau, t_end, taper)
    return np.where(tau > 0, g, 0.0) / g_max


def canonical_hrf(tau):
    """SPM double gamma (a1 = 6, a2 = 16, ratio = 1/6), scaled to a peak of 1; 0 for tau < 0."""
    return double_gamma_hrf(tau, **CANONICAL)


def hrf_props(a1=CANONICAL['a1'], a2=CANONICAL['a2'], ratio=CANONICAL['ratio'], t_end=None, taper=5.0):
    """(peak latency in s, max rising slope in 1/s, undershoot depth relative to the peak) of the unit-peak HRF."""
    g = double_gamma_hrf(_tt, a1, a2, ratio, t_end, taper)
    return float(_tt[g.argmax()]), float(np.gradient(g, _tt).max()), float(-g.min())


T_PEAK, MAX_SLOPE, _ = hrf_props()   # canonical values (kept for older code)


def true_trial_hrfs(truth, lag_t):
    """Per-trial injected HRFs on the lag axis lag_t (s), locked to the labeled onset.
    truth: rows of the ground-truth csv (amp_uM, time_scale, onset, onset_true and, for simulations made with a
    fitted shape, hrf_a1, hrf_a2, hrf_ratio; without them the canonical shape is used; hrf_t_end / hrf_taper if the
    HRF was tapered to 0). onset_true - onset is the actual start of the HRF relative to the labeled onset
    (sample rounding + jitter)."""
    shift = (truth['onset_true'] - truth['onset']).values
    if {'hrf_a1', 'hrf_a2', 'hrf_ratio'} <= set(truth.columns):
        shapes = truth[['hrf_a1', 'hrf_a2', 'hrf_ratio']].values
    else:
        shapes = np.tile([CANONICAL['a1'], CANONICAL['a2'], CANONICAL['ratio']], (len(truth), 1))
    if 'hrf_t_end' in truth.columns:
        tap = [(None if not np.isfinite(e) else e, w) for e, w in truth[['hrf_t_end', 'hrf_taper']].values]
    else:
        tap = [(None, 5.0)] * len(truth)
    return np.stack([a * double_gamma_hrf(s * (lag_t - d), *p, t_end=e, taper=w)
                     for a, s, d, p, (e, w) in zip(truth['amp_uM'].values, truth['time_scale'].values, shift, shapes,
                                                   tap)])
