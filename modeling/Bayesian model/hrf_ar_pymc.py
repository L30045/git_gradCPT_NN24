"""
Joint Bayesian estimation of a parcel HRF and AR(p) noise in PyMC.

    x = sum_k a_k * (E_k * h) + N b + e,   h = B w,   e ~ AR(p) with Student-t innovations

Setup:
    pip install pymc arviz scipy matplotlib nutpie   # nutpie is optional but much faster

Run:
    python hrf_ar_pymc.py      # simulates data, fits, prints a recovery report, saves hrf_ar_sim.png
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.interpolate import BSpline
from scipy.stats import gamma


# ----------------------------------------------------------------------------
# Design construction
# ----------------------------------------------------------------------------
def lag_basis(n_lag, L, deg=3):
    """Cubic B-spline basis over the HRF window: (n_lag, L)."""
    knots = np.r_[[0] * deg, np.linspace(0, n_lag - 1, L - deg + 1), [n_lag - 1] * deg]
    return BSpline.design_matrix(np.arange(n_lag), knots, deg).toarray()


def event_design(onsets_list, T, B):
    """D[k] = event train for condition k convolved with each basis column: (K, T, L)."""
    D = []
    for ons in onsets_list:
        stick = np.zeros(T)
        stick[np.asarray(ons, dtype=int)] = 1.0
        D.append(np.column_stack([np.convolve(stick, B[:, j])[:T] for j in range(B.shape[1])]))
    return np.stack(D)


def canonical_prior_mean(B, fs):
    """Double-gamma HRF projected onto B, scaled to unit peak.

    Returns w0 and Q, an (L, L-1) orthonormal basis of the directions in w-space
    that leave the projection of h = B w onto the canonical shape h0 = B w0 unchanged.
    Writing w = w0 + Q v then fixes <h, h0> / <h0, h0> = 1 exactly, which identifies
    the scale and sign of the bilinear term a_k * h without a soft constraint."""
    t = np.arange(B.shape[0]) / fs
    h_can = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    w0, *_ = np.linalg.lstsq(B, h_can, rcond=None)
    w0 = w0 / (B @ w0).max()
    g = B.T @ (B @ w0)                       # gradient of <B w, h0> w.r.t. w
    Q = np.linalg.svd(g[None, :])[2][1:].T   # null space of g: (L, L-1)
    return w0, Q


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def pacf_to_ar(r, P):
    """Durbin-Levinson: partial autocorrelations in (-1, 1) -> stationary AR coefficients.
    phi_{k,j} = phi_{k-1,j} - r_k * phi_{k-1,k-j}. Unrolled; P is a Python int."""
    phi = [r[0]]
    for k in range(1, P):
        rk = r[k]
        phi = [phi[j] - rk * phi[k - 1 - j] for j in range(k)] + [rk]
    return pt.stack(phi)


def build_model(x, D, N, B, w0, Q, P, pacf_scale=1.0):
    K, T, L = D.shape
    M = N.shape[1]
    lag_scale = pacf_scale / np.arange(1, P + 1)

    with pm.Model() as model:
        # ---- HRF shape: w = w0 + Q v keeps the projection on the canonical HRF at 1 ----
        tau = pm.HalfNormal("tau", 0.5)
        v = pm.Normal("v", 0.0, 1.0, shape=L - 1)          # deviation from canonical (weak ridge)
        dev = pt.dot(Q, v)
        w = pm.Deterministic("w", w0 + dev)
        # smoothness: 2nd differences of the deviation, scale learned
        pm.Potential("smooth", pm.logp(pm.Normal.dist(0.0, tau), pt.diff(dev, n=2)).sum())
        pm.Deterministic("hrf", pt.dot(B, w))

        # ---- amplitudes, intercept and nuisance ----
        # The intercept is required: with overlapping events the late HRF lags
        # otherwise absorb the baseline shift left by mean-centring x.
        a = pm.Normal("a", 0.0, 1.0, shape=K)
        b0 = pm.Normal("b0", 0.0, 1.0)
        b = pm.Normal("b", 0.0, 5.0, shape=M) if M > 0 else pt.zeros(0)

        # ---- AR noise: unconstrained PACF -> stationary phi ----
        z = pm.Normal("z", 0.0, lag_scale, shape=P)
        phi = pm.Deterministic("phi", pacf_to_ar(pt.tanh(z), P))
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Deterministic("nu", 2.0 + pm.Gamma("nu_minus2", 2.0, 0.1))

        # ---- mean and residual ----
        Dw = pt.tensordot(D, w, axes=[[2], [0]])            # (K, T)
        mu = b0 + pt.dot(a, Dw)
        if M > 0:
            mu = mu + pt.dot(N, b)
        e = x - mu

        # ---- whiten inside the model: innovations for t > P ----
        innov = e[P:]
        for k in range(P):                                  # lag k+1
            innov = innov - phi[k] * e[P - 1 - k: T - 1 - k]

        # The map e[P:] -> innov is unit lower-triangular (Jacobian = 1),
        # so this Potential is the exact conditional likelihood.
        pm.Potential("lik", pm.logp(pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), innov).sum())

    return model


# ----------------------------------------------------------------------------
# Simulated data with a known HRF and AR(2) noise
# ----------------------------------------------------------------------------
def simulate(B, fs, T, seed=0):
    """Plant a known HRF in AR(2) noise. Returns everything needed to fit and to score."""
    rng = np.random.default_rng(seed)
    t = np.arange(B.shape[0]) / fs

    # True HRF: earlier peak and shallower undershoot than the canonical prior mean.
    h = gamma.pdf(t, 5, scale=0.9) - gamma.pdf(t, 14) / 8
    w_true, *_ = np.linalg.lstsq(B, h, rcond=None)          # representable in the basis
    w0, _ = canonical_prior_mean(B, fs)
    h0 = B @ w0
    h_true = B @ w_true
    h_true /= (h_true @ h0) / (h0 @ h0)                      # the model's normalization

    # Two conditions, 60 events each, on a 1 s grid (offset by one sample)
    onsets = [np.sort(rng.choice(np.arange(40, T - 100, 4), 60, replace=False)) + k
              for k in range(2)]
    sticks = [np.bincount(o, minlength=T)[:T].astype(float) for o in onsets]
    a_true = np.array([1.0, 0.5])
    evoked = sum(a * np.convolve(s, h_true)[:T] for a, s in zip(a_true, sticks))

    # AR(2) noise, Gaussian innovations
    phi_true = np.array([0.7, 0.2])
    e = np.zeros(T)
    u = rng.standard_normal(T) * 0.3
    for i in range(2, T):
        e[i] = phi_true @ e[i - 2:i][::-1] + u[i]

    x = evoked + e
    return dict(t=t, x=x, onsets=onsets, h_true=h_true, h0=h0,
                a_true=a_true, phi_true=phi_true, evoked=evoked)


# ----------------------------------------------------------------------------
# Figure: truth vs posterior
# ----------------------------------------------------------------------------
def plot_recovery(idata, sim, x_s, D, P, fs, scale, path="hrf_ar_sim.png"):
    import matplotlib.pyplot as plt

    post = idata.posterior
    t = sim["t"]
    hrf = post["hrf"].stack(s=("chain", "draw")).values.T            # (draws, n_lag)
    phi = post["phi"].stack(s=("chain", "draw")).values.T            # (draws, P)
    a = post["a"].stack(s=("chain", "draw")).values.T                # (draws, K)
    a_true_s = sim["a_true"] / scale                                 # amplitudes in standardized units
    phi_true = np.r_[sim["phi_true"], np.zeros(P - len(sim["phi_true"]))]

    truth_c, est_c, prior_c = "#1f1f1f", "#2a6fdb", "#9a9a9a"
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # (1) HRF
    lo, hi = np.percentile(hrf, [2.5, 97.5], 0)
    cover = np.mean((sim["h_true"] >= lo) & (sim["h_true"] <= hi))
    a0 = ax[0, 0]
    a0.fill_between(t, lo, hi, color=est_c, alpha=0.2, lw=0, label="95% credible band")
    a0.plot(t, hrf.mean(0), color=est_c, lw=2, label="Posterior mean")
    a0.plot(t, sim["h_true"], color=truth_c, lw=1.6, ls="--", label="Truth")
    a0.plot(t, sim["h0"], color=prior_c, lw=1.2, ls=":", label="Canonical (prior mean)")
    a0.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    a0.set(title=f"HRF: band covers truth at {cover:.0%} of lags",
           xlabel="Lag (s)", ylabel="h (projection on canonical = 1)")
    a0.legend(frameon=False, fontsize=9)

    # (2) AR coefficients
    a1 = ax[0, 1]
    lags = np.arange(1, P + 1)
    plo, phi_hi = np.percentile(phi, [2.5, 97.5], 0)
    a1.errorbar(lags, phi.mean(0), yerr=[phi.mean(0) - plo, phi_hi - phi.mean(0)],
                fmt="o", color=est_c, capsize=4, label="Posterior mean, 95% CI")
    a1.scatter(lags, phi_true, marker="_", s=400, color=truth_c, lw=2, label="Truth", zorder=3)
    a1.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    a1.set(title="AR coefficients φ", xlabel="Lag", ylabel="φ", xticks=lags)
    a1.legend(frameon=False, fontsize=9)

    # (3) Condition amplitudes
    a2 = ax[1, 0]
    for k in range(a.shape[1]):
        a2.hist(a[:, k], bins=40, color=f"C{k}", alpha=0.55, label=f"Condition {k + 1} posterior")
        a2.axvline(a_true_s[k], color=f"C{k}", lw=2, ls="--")
    a2.set(title="Amplitudes aₖ (dashed = truth, standardized units)",
           xlabel="aₖ", ylabel="Draws")
    a2.legend(frameon=False, fontsize=9)

    # (4) Data and fit, first 2 minutes
    a3 = ax[1, 1]
    w = post["w"].mean(("chain", "draw")).values
    b0 = post["b0"].mean(("chain", "draw")).item()
    fit = b0 + np.tensordot(D, w, axes=[[2], [0]]).T @ a.mean(0)
    n = int(120 * fs)
    ts = np.arange(n) / fs
    a3.plot(ts, x_s[:n], color="#bbbbbb", lw=0.8, label="Data (standardized)")
    true_fit = (sim["evoked"] - sim["x"].mean()) / scale             # evoked + true offset, standardized
    a3.plot(ts, true_fit[:n], color=truth_c, lw=1.4, ls="--", label="True evoked")
    a3.plot(ts, fit[:n], color=est_c, lw=1.6, label="Posterior mean fit")
    a3.set(title="Data and fitted evoked response (first 2 min)",
           xlabel="Time (s)", ylabel="x")
    a3.legend(frameon=False, fontsize=9, loc="upper right")

    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    fs, T = 4.0, 2400               # 10 min at 4 Hz
    lag_sec, L, P = 20.0, 10, 4     # HRF window, spline basis size, AR order
    USE_NUTPIE = False              # False reproduces the reported numbers; True is faster

    B = lag_basis(int(lag_sec * fs), L)
    sim = simulate(B, fs, T, seed=0)

    x = sim["x"]
    scale = x.std()
    x_s = (x - x.mean()) / scale
    D = event_design(sim["onsets"], T, B)
    N = np.zeros((T, 0))            # no nuisance regressors in the simulation
    w0, Q = canonical_prior_mean(B, fs)

    model = build_model(x_s, D, N, B, w0, Q, P)

    t0 = time.time()
    with model:
        idata = pm.sample(draws=500, tune=500, chains=4, cores=4, target_accept=0.9,
                          nuts_sampler="nutpie" if USE_NUTPIE else "pymc",
                          random_seed=1, progressbar=False)
    print(f"sampling time {time.time() - t0:.0f} s")

    # ---- diagnostics ----
    print("divergences:", int(idata.sample_stats["diverging"].sum()))
    print(az.summary(idata, var_names=["a", "b0", "phi", "sigma", "nu", "tau"], round_to=3))

    # ---- recovery report ----
    hrf = idata.posterior["hrf"].stack(s=("chain", "draw")).values.T
    lo, hi = np.percentile(hrf, [2.5, 97.5], 0)
    a_post = idata.posterior["a"]
    print("\nRecovery vs truth")
    print(f"  HRF 95% band coverage : {np.mean((sim['h_true'] >= lo) & (sim['h_true'] <= hi)):.0%}")
    print(f"  corr(HRF est, truth)  : {np.corrcoef(hrf.mean(0), sim['h_true'])[0, 1]:.4f}")
    print(f"  peak time true / est  : {sim['t'][sim['h_true'].argmax()]:.2f} / "
          f"{sim['t'][hrf.mean(0).argmax()]:.2f} s")
    print(f"  phi true              : {sim['phi_true']}")
    print(f"  phi est               : {idata.posterior['phi'].mean(('chain', 'draw')).values.round(3)}")
    print(f"  a ratio true / est    : {sim['a_true'][1] / sim['a_true'][0]:.3f} / "
          f"{(a_post[..., 1] / a_post[..., 0]).mean().item():.3f}")
    print(f"  a true (standardized) : {(sim['a_true'] / scale).round(3)}")

    plot_recovery(idata, sim, x_s, D, P, fs, scale)
    try:
        idata.to_netcdf("hrf_ar_idata.nc")          # needs netCDF4 or h5netcdf
    except ValueError as err:
        print(f"posterior not saved: {err}")
