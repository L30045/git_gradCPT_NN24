"""Shared pieces of the mnt-correct HRF simulation (simulate_mnt-correct_HRF.py, cmp_HRF_simulation.py)."""
import numpy as np
from scipy.stats import gamma

_tt = np.linspace(0, 32, 32001)
_g_raw = gamma.pdf(_tt, 6) - gamma.pdf(_tt, 16) / 6
_G_MAX = _g_raw.max()


def canonical_hrf(tau):
    """SPM double gamma (peak a1 = 6, undershoot a2 = 16, ratio 1/6), scaled to a peak of 1; 0 for tau < 0."""
    tau = np.asarray(tau, dtype=float)
    return np.where(tau > 0, gamma.pdf(tau, 6) - gamma.pdf(tau, 16) / 6, 0.0) / _G_MAX


T_PEAK = _tt[_g_raw.argmax()]                                # peak latency of g (s)
MAX_SLOPE = np.gradient(_g_raw / _G_MAX, _tt).max()          # max rising slope of g (1/s)


def true_trial_hrfs(truth, lag_t):
    """Per-trial injected HRFs on the lag axis lag_t (s), locked to the labeled onset.
    truth: rows of the ground-truth csv (amp_uM, time_scale, onset, onset_true). onset_true - onset is the
    actual start of the HRF relative to the labeled onset (sample rounding + jitter)."""
    shift = (truth['onset_true'] - truth['onset']).values
    return np.stack([a * canonical_hrf(s * (lag_t - d))
                     for a, s, d in zip(truth['amp_uM'].values, truth['time_scale'].values, shift)])
