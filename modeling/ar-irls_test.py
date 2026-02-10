#%%
from scipy.signal import lfilter
def prewhiten_arp_lfilter(data, rho_coefficients):
    """
    Pre-whiten AR(p) data using lfilter
    
    For AR(p): y(t) = ρ₁×y(t-1) + ρ₂×y(t-2) + ... + ρₚ×y(t-p) + ε(t)
    
    To recover ε(t): ε(t) = y(t) - ρ₁×y(t-1) - ρ₂×y(t-2) - ... - ρₚ×y(t-p)
    
    Parameters:
    -----------
    data : array (n,)
        Time series data
    rho_coefficients : array (p,)
        AR coefficients [ρ₁, ρ₂, ..., ρₚ]
    
    Returns:
    --------
    data_white : array (n,)
        Pre-whitened data
    """
    p = len(rho_coefficients)
    
    # Filter coefficients
    b = np.array([1.0])  # Numerator
    a = np.concatenate([[1.0], -rho_coefficients])  # Denominator: [1, -ρ₁, -ρ₂, ..., -ρₚ]
    
    # Apply filter
    data_white = lfilter(b, a, data)
    
    return data_white

def prewhiten_design_matrix_lfilter(X, rho_coefficients):
    """
    Pre-whiten design matrix using lfilter
    
    Parameters:
    -----------
    X : array (n, p) or (n,)
        Design matrix with n timepoints and p regressors
        Or single regressor (1D array)
    rho_coefficients : array (ar_order,) or float
        AR coefficients
    
    Returns:
    --------
    X_white : array, same shape as X
        Pre-whitened design matrix
    """
    # Handle scalar rho (AR(1))
    if np.isscalar(rho_coefficients):
        rho_coefficients = np.array([rho_coefficients])
    
    # Handle 1D array
    if X.ndim == 1:
        return prewhiten_arp_lfilter(X, rho_coefficients)
    
    # Handle 2D array (multiple regressors)
    n, p = X.shape
    X_white = np.zeros_like(X)
    
    for col in range(p):
        X_white[:, col] = prewhiten_arp_lfilter(X[:, col], rho_coefficients)
    
    return X_white

# ============================================================================
# Complete Regression Example with lfilter
# ============================================================================

np.random.seed(42)
n = 200

# Design matrix: task blocks + intercept
X = np.zeros((n, 2))
X[40:70, 0] = 1   # Task block 1
X[110:140, 0] = 1  # Task block 2
X[:, 1] = 1        # Intercept

true_beta = np.array([2.0, 1.5])
true_rho = np.array([0.65, 0.25])  # AR(2)

# Generate AR(2) errors
errors = np.zeros(n)
errors[0] = np.random.randn() * 0.3
errors[1] = true_rho[0] * errors[0] + np.random.randn() * 0.3
for t in range(2, n):
    errors[t] = (true_rho[0] * errors[t-1] + 
                 true_rho[1] * errors[t-2] + 
                 np.random.randn() * 0.3)

y = X @ true_beta + errors

print("\n" + "="*70)
print("REGRESSION WITH lfilter PRE-WHITENING")
print("="*70)
print(f"True β: {true_beta}")
print(f"True AR(2) coefficients: {true_rho}")

# Step 1: OLS to get initial residuals
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
residuals_ols = y - X @ beta_ols

# Step 2: Estimate AR coefficients from residuals
from statsmodels.tsa.ar_model import AutoReg

# Fit AR(2) to residuals
ar_model = AutoReg(residuals_ols, lags=2, old_names=False)
ar_fitted = ar_model.fit()
rho_estimated = ar_fitted.params[1:]  # Skip intercept

print(f"\nEstimated AR(2) coefficients: {rho_estimated}")

# Step 3: Pre-whiten using lfilter
y_white = prewhiten_arp_lfilter(y, rho_estimated)
X_white = prewhiten_design_matrix_lfilter(X, rho_estimated)

print(f"\nData shapes:")
print(f"  Original: y={y.shape}, X={X.shape}")
print(f"  Whitened: y_white={y_white.shape}, X_white={X_white.shape}")

# Step 4: GLS on whitened data
beta_gls = np.linalg.lstsq(X_white, y_white, rcond=None)[0]
residuals_gls = y_white - X_white @ beta_gls

print(f"\nResults:")
print(f"  True β:      {true_beta}")
print(f"  OLS β:       {beta_ols}")
print(f"  GLS β:       {beta_gls}")
print(f"\n  OLS error:   {np.linalg.norm(beta_ols - true_beta):.4f}")
print(f"  GLS error:   {np.linalg.norm(beta_gls - true_beta):.4f}")

# Check residual autocorrelation
print(f"\nResidual autocorrelation:")
for lag in range(1, 4):
    acf_ols = np.corrcoef(residuals_ols[:-lag], residuals_ols[lag:])[0, 1]
    acf_gls = np.corrcoef(residuals_gls[:-lag], residuals_gls[lag:])[0, 1]
    print(f"  Lag {lag}: OLS={acf_ols:6.3f}, GLS={acf_gls:6.3f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

time = np.arange(n)

# Plot 1: Data and fits
axes[0, 0].plot(time, y, 'b-', alpha=0.5, linewidth=1, label='Data')
axes[0, 0].plot(time, X @ beta_ols, 'r-', linewidth=2, label='OLS fit')
axes[0, 0].plot(time, X @ beta_gls, 'g--', linewidth=2, label='GLS fit')
axes[0, 0].plot(time, X @ true_beta, 'k:', linewidth=2, alpha=0.7, label='True')
axes[0, 0].set_title('Original Data and Model Fits', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Whitened data
axes[0, 1].plot(time, y_white, 'r-', alpha=0.7, linewidth=1)
axes[0, 1].set_title('Pre-whitened Data (lfilter)', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('y*')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: OLS residuals ACF
from scipy import signal
acf_ols_full = signal.correlate(residuals_ols, residuals_ols, mode='full')
acf_ols_full = acf_ols_full[len(acf_ols_full)//2:]
acf_ols_full = acf_ols_full / acf_ols_full[0]

axes[1, 0].stem(np.arange(15), acf_ols_full[:15], basefmt=' ')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[1, 0].axhline(y=1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=-1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
axes[1, 0].set_title('ACF: OLS Residuals (Autocorrelated)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: GLS residuals ACF
acf_gls_full = signal.correlate(residuals_gls, residuals_gls, mode='full')
acf_gls_full = acf_gls_full[len(acf_gls_full)//2:]
acf_gls_full = acf_gls_full / acf_gls_full[0]

axes[1, 1].stem(np.arange(15), acf_gls_full[:15], basefmt=' ', linefmt='g-', markerfmt='go')
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[1, 1].axhline(y=1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=-1.96/np.sqrt(n), color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_title('ACF: GLS Residuals (White)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def ar_irls_lfilter(X, y, ar_order=1, max_iter=20, tune=4.685, 
                    tolerance=1e-6, verbose=True):
    """
    Complete AR-IRLS using scipy.signal.lfilter for pre-whitening
    
    Parameters:
    -----------
    X : array (n, p)
        Design matrix
    y : array (n,) or (n, channels)
        Response data
    ar_order : int
        Order of AR model
    max_iter : int
        Maximum iterations
    tune : float
        Tukey biweight constant
    tolerance : float
        Convergence tolerance
    verbose : bool
        Print iteration details
    
    Returns:
    --------
    results : dict with keys:
        - beta: coefficient estimates
        - rho: AR coefficients
        - weights: robust weights
        - iterations: number of iterations
        - converged: convergence flag
    """
    n = len(y)
    is_multichannel = y.ndim > 1
    n_channels = y.shape[1] if is_multichannel else 1
    
    # Initialize
    w = np.ones(n)
    rho = np.zeros(ar_order)
    
    if verbose:
        print("="*70)
        print(f"AR({ar_order})-IRLS with lfilter")
        print("="*70)
        print(f"Data: {n} timepoints, {n_channels} channels")
        print(f"Design: {X.shape[1]} regressors")
        print(f"Max iterations: {max_iter}, Tolerance: {tolerance}")
        print()
    
    for iteration in range(1, max_iter + 1):
        if verbose:
            print(f"Iteration {iteration}")
            print("-"*40)
        
        rho_old = rho.copy()
        w_old = w.copy()
        
        # STEP 1: Weighted Least Squares
        W = np.diag(w)
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y
        beta = np.linalg.solve(XtWX, XtWy)
        
        # Compute residuals
        residuals = y - X @ beta
        
        # STEP 2: Estimate AR coefficients
        if ar_order == 1:
            if is_multichannel:
                rho = np.array([np.mean([
                    np.corrcoef(residuals[:-1, ch], residuals[1:, ch])[0, 1]
                    for ch in range(n_channels)
                ])])
            else:
                rho = np.array([np.corrcoef(residuals[:-1], residuals[1:])[0, 1]])
        else:
            # Use Yule-Walker for AR(p)
            from statsmodels.tsa.ar_model import AutoReg
            if is_multichannel:
                rho_estimates = []
                for ch in range(n_channels):
                    ar_model = AutoReg(residuals[:, ch], lags=ar_order, old_names=False)
                    ar_fitted = ar_model.fit()
                    rho_estimates.append(ar_fitted.params[1:])
                rho = np.mean(rho_estimates, axis=0)
            else:
                ar_model = AutoReg(residuals, lags=ar_order, old_names=False)
                ar_fitted = ar_model.fit()
                rho = ar_fitted.params[1:]
        
        rho = np.clip(rho, -0.99, 0.99)
        
        if verbose:
            if ar_order == 1:
                print(f"  AR coefficient: ρ = {rho[0]:.4f}")
            else:
                print(f"  AR coefficients: {rho}")
        
        # STEP 3: Pre-whiten using lfilter
        y_white = prewhiten_arp_lfilter(y, rho) if not is_multichannel else \
                  np.column_stack([prewhiten_arp_lfilter(y[:, ch], rho) 
                                  for ch in range(n_channels)])
        X_white = prewhiten_design_matrix_lfilter(X, rho)
        
        # Re-fit on whitened data
        XtW = X_white.T @ W
        XtWX = XtW @ X_white
        XtWy = XtW @ y_white
        beta = np.linalg.solve(XtWX, XtWy)
        
        # Whitened residuals
        residuals_white = y_white - X_white @ beta
        
        # STEP 4: Compute robust weights
        if is_multichannel:
            w_new = np.ones(n)
            for ch in range(n_channels):
                mad = np.median(np.abs(residuals_white[:, ch] - 
                                      np.median(residuals_white[:, ch])))
                if mad < 1e-10:
                    continue
                std_res = residuals_white[:, ch] / (1.4826 * mad)
                mask = np.abs(std_res) <= tune
                u = std_res[mask] / tune
                w_ch = np.zeros(n)
                w_ch[mask] = (1 - u**2)**2
                w_new = np.minimum(w_new, w_ch)  # Take minimum across channels
        else:
            mad = np.median(np.abs(residuals_white - np.median(residuals_white)))
            if mad < 1e-10:
                w_new = np.ones(n)
            else:
                std_res = residuals_white / (1.4826 * mad)
                w_new = np.zeros(n)
                mask = np.abs(std_res) <= tune
                u = std_res[mask] / tune
                w_new[mask] = (1 - u**2)**2
        
        if verbose:
            print(f"  Weights: mean={np.mean(w_new):.4f}, "
                  f"{100*np.mean(w_new < 0.1):.1f}% downweighted")
        
        # STEP 5: Check convergence
        delta_rho = np.max(np.abs(rho - rho_old))
        delta_w = np.max(np.abs(w_new - w_old))
        
        if verbose:
            print(f"  Convergence: Δρ={delta_rho:.6f}, Δw={delta_w:.6f}")
        
        converged = (delta_rho < tolerance) and (delta_w < tolerance)
        
        w = w_new
        
        if converged:
            if verbose:
                print(f"\n✓ Converged after {iteration} iterations!")
            break
        
        if verbose:
            print()
    
    # Final fit
    y_white = prewhiten_arp_lfilter(y, rho) if not is_multichannel else \
              np.column_stack([prewhiten_arp_lfilter(y[:, ch], rho) 
                              for ch in range(n_channels)])
    X_white = prewhiten_design_matrix_lfilter(X, rho)
    W = np.diag(w)
    beta_final = np.linalg.solve(X_white.T @ W @ X_white, X_white.T @ W @ y_white)
    
    residuals_final = y - X @ beta_final
    
    return {
        'beta': beta_final,
        'rho': rho,
        'weights': w,
        'iterations': iteration,
        'converged': converged,
        'residuals': residuals_final
    }

# ============================================================================
# Full Example
# ============================================================================

# Generate data with AR(2) errors
np.random.seed(42)
n = 200

X = np.column_stack([
    np.concatenate([np.zeros(40), np.ones(30), np.zeros(50), 
                   np.ones(30), np.zeros(50)]),
    np.ones(n)
])
true_beta = np.array([2.5, 1.5])
true_rho = np.array([0.6, 0.25])

errors = np.zeros(n)
errors[0] = np.random.randn() * 0.3
errors[1] = true_rho[0] * errors[0] + np.random.randn() * 0.3
for t in range(2, n):
    errors[t] = (true_rho[0] * errors[t-1] + 
                 true_rho[1] * errors[t-2] + 
                 np.random.randn() * 0.3)

# Add some outliers
outliers = [65, 135, 170]
for idx in outliers:
    errors[idx] += np.random.choice([-1, 1]) * 1.5

y = X @ true_beta + errors

print("\n" + "="*70)
print("COMPLETE AR-IRLS EXAMPLE WITH lfilter")
print("="*70)
print(f"True β: {true_beta}")
print(f"True AR(2) coefficients: {true_rho}")
print(f"Added {len(outliers)} outliers at positions: {outliers}")
print()

# Run AR-IRLS with AR(2)
results = ar_irls_lfilter(X, y, ar_order=2, max_iter=15, verbose=True)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"True β:         {true_beta}")
print(f"Estimated β:    {results['beta']}")
print(f"Error:          {np.linalg.norm(results['beta'] - true_beta):.4f}")
print(f"\nTrue AR:        {true_rho}")
print(f"Estimated AR:   {results['rho']}")
print(f"\nConverged:      {results['converged']}")
print(f"Iterations:     {results['iterations']}")
print(f"\nOutlier weights:")
for idx in outliers:
    print(f"  Position {idx}: weight = {results['weights'][idx]:.4f}")

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def check_autocorrelation_visual(residuals, lags=20, title="Residuals"):
    """
    Visual check for autocorrelation using ACF and PACF plots
    
    Parameters:
    -----------
    residuals : array (n,)
        Residuals or whitened data to check
    lags : int
        Number of lags to display
    title : str
        Title for the plots
    """
    n = len(residuals)
    
    # Compute ACF and PACF
    acf_values = acf(residuals, nlags=lags, fft=False)
    pacf_values = pacf(residuals, nlags=lags)
    
    # Confidence intervals (95%)
    conf_int = 1.96 / np.sqrt(n)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ACF
    lags_array = np.arange(lags + 1)
    axes[0].stem(lags_array, acf_values, basefmt=' ')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[0].axhline(y=conf_int, color='r', linestyle='--', linewidth=2, 
                    alpha=0.5, label='95% CI')
    axes[0].axhline(y=-conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[0].set_xlabel('Lag', fontsize=12)
    axes[0].set_ylabel('ACF', fontsize=12)
    axes[0].set_title(f'Autocorrelation Function: {title}', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot PACF
    axes[1].stem(lags_array, pacf_values, basefmt=' ')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1].axhline(y=conf_int, color='r', linestyle='--', linewidth=2, 
                    alpha=0.5, label='95% CI')
    axes[1].axhline(y=-conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].set_xlabel('Lag', fontsize=12)
    axes[1].set_ylabel('PACF', fontsize=12)
    axes[1].set_title(f'Partial Autocorrelation Function: {title}', fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Check how many lags are significant
    significant_lags = np.where(np.abs(acf_values[1:]) > conf_int)[0] + 1
    
    print("="*70)
    print(f"ACF ANALYSIS: {title}")
    print("="*70)
    print(f"Number of observations: {n}")
    print(f"95% confidence interval: ±{conf_int:.4f}")
    print(f"\nSignificant ACF lags (beyond 95% CI):")
    if len(significant_lags) == 0:
        print("  None - No significant autocorrelation detected! ✓")
    else:
        print(f"  Lags: {significant_lags.tolist()}")
        print(f"  Values: {acf_values[significant_lags]}")
    
    return acf_values, pacf_values, significant_lags

# ============================================================================
# Example: Generate and test data
# ============================================================================

np.random.seed(42)
n = 300

print("="*70)
print("CHECKING FOR AUTOCORRELATION - VISUAL METHOD")
print("="*70)

# Case 1: AR(1) data (has autocorrelation)
rho = 0.7
y_ar1 = np.zeros(n)
y_ar1[0] = np.random.randn()
for t in range(1, n):
    y_ar1[t] = rho * y_ar1[t-1] + np.random.randn() * 0.5

print("\n" + "-"*70)
print("CASE 1: Original AR(1) Data (Should have autocorrelation)")
print("-"*70)
acf_ar1, pacf_ar1, sig_lags_ar1 = check_autocorrelation_visual(y_ar1, lags=20, 
                                                                 title="Original AR(1)")
plt.show()

# Case 2: Pre-whitened data (should have no autocorrelation)
from scipy.signal import lfilter
b = [1.0]
a = [1.0, -rho]
y_white = lfilter(b, a, y_ar1)
y_white[0] = y_ar1[0] * np.sqrt(1 - rho**2)

print("\n" + "-"*70)
print("CASE 2: Pre-whitened Data (Should have NO autocorrelation)")
print("-"*70)
acf_white, pacf_white, sig_lags_white = check_autocorrelation_visual(y_white, lags=20,
                                                                       title="Pre-whitened")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print("="*70)
print("GENERATE AR(1) DATA AND TEST AutoReg")
print("="*70)

# ============================================================================
# STEP 1: Generate AR(1) Data
# ============================================================================

np.random.seed(42)

# AR(1) parameters
n = 500  # Number of observations
rho_true = 0.7  # True AR coefficient
sigma = 0.5  # Noise standard deviation
mu = 2.0  # Mean of the process

print(f"\nGenerating AR(1) process:")
print(f"  n = {n} observations")
print(f"  ρ = {rho_true} (AR coefficient)")
print(f"  σ = {sigma} (noise std)")
print(f"  μ = {mu} (mean)")

# Compute the constant term: c = μ(1-ρ)
c = mu * (1 - rho_true)
print(f"  c = {c:.4f} (constant term)")

# Generate white noise
epsilon = np.random.randn(n) * sigma

# Generate AR(1) process: y(t) = c + ρ*y(t-1) + ε(t)
y = np.zeros(n)
y[0] = mu + epsilon[0]  # Initialize near mean

for t in range(1, n):
    y[t] = c + rho_true * y[t-1] + epsilon[t]

print(f"\nGenerated data statistics:")
print(f"  Mean: {y.mean():.4f} (expected: {mu:.4f})")
print(f"  Std:  {y.std():.4f}")
print(f"  Min:  {y.min():.4f}")
print(f"  Max:  {y.max():.4f}")

# ============================================================================
# STEP 2: Visualize the Generated Data
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time series
axes[0, 0].plot(y, 'b-', alpha=0.7, linewidth=1)
axes[0, 0].axhline(y=mu, color='r', linestyle='--', linewidth=2, 
                   label=f'True mean = {mu}')
axes[0, 0].set_title('Generated AR(1) Time Series', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram
axes[0, 1].hist(y, bins=30, alpha=0.7, edgecolor='black', density=True)
axes[0, 1].axvline(x=mu, color='r', linestyle='--', linewidth=2, 
                   label=f'True mean = {mu}')
axes[0, 1].set_title('Distribution', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: ACF
acf_vals = acf(y, nlags=20, fft=False)
lags = np.arange(len(acf_vals))
axes[1, 0].stem(lags, acf_vals, basefmt=' ')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
conf_int = 1.96 / np.sqrt(n)
axes[1, 0].axhline(y=conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].axhline(y=-conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: PACF
pacf_vals = pacf(y, nlags=20)
axes[1, 1].stem(lags, pacf_vals, basefmt=' ')
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
axes[1, 1].axhline(y=conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 1].axhline(y=-conf_int, color='r', linestyle='--', linewidth=2, alpha=0.5)
axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('PACF')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% ============================================================================
# STEP 3: Fit AutoReg Model
# ============================================================================
# AR(1) parameters
n = 500  # Number of observations
rho_true = 0.7  # True AR coefficient
sigma = 0.5  # Noise standard deviation
mu = 2.0  # Mean of the process

# Compute the constant term: c = μ(1-ρ)
c = mu * (1 - rho_true)

# Generate white noise
epsilon = np.random.randn(n) * sigma

# Generate AR(1) process: y(t) = c + ρ*y(t-1) + ε(t)
y = np.zeros(n)
y[0] = mu + epsilon[0]  # Initialize near mean

for t in range(1, n):
    y[t] = c + rho_true * y[t-1] + epsilon[t]
    
# Fit AR(1) model
model = AutoReg(y, lags=1, old_names=False)
results = model.fit()

b = np.array([1.0])  # Numerator
a = np.concatenate([[1.0], -results.params[1:]])  # Denominator: [1, -ρ₁, -ρ₂, ..., -ρₚ]

print(results.params)

yf = lfilter(a,b,y)

plt.figure()
plt.plot(y,label='y')
plt.plot(yf,label='yf')
plt.legend()

