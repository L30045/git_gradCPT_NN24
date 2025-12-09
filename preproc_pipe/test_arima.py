#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_ccf
from scipy.signal import lfilter

# 1. Simulate two time series.
# X is an AR(1) process (autocorrelated), Y is related to X with a 2-lag delay.
np.random.seed(42)
n_samples = 100
# White noise component for X and Y
w_t = np.random.normal(0, 1, n_samples)
eta_t = np.random.normal(0, 1, n_samples)

# Generate X_t = 0.7 * X_{t-1} + W_t (autocorrelated series)
X = [0.0] * n_samples
for i in range(1, n_samples):
    X[i] = 0.7 * X[i-1] + w_t[i]

# Generate Y_t = 0.5 * X_{t-2} + Eta_t (Y lags X by 2, plus white noise)
Y = [0.0] * n_samples
for i in range(2, n_samples):
    Y[i] = 0.5 * X[i-2] + eta_t[i]

X = pd.Series(X)
Y = pd.Series(Y)

# 2. Plot the raw cross-correlation function (CCF)
fig, ax = plt.subplots(figsize=(10, 4))
plot_ccf(X, Y, title="Raw CCF between X and Y", ax=ax, lags=20)
plt.show()

# 3. Pre-whitening Procedure

# A. Fit an appropriate ARIMA model to the INPUT series (X)
# The ACF/PACF of X suggests an AR(1) model. We fit an ARIMA(1, 0, 0)
model_X = ARIMA(X, order=(1, 0, 0))
results_X = model_X.fit()
print(f"Fitted AR(1) model coefficient for X: {results_X.params['ar.L1']}")

# B. Get the residuals from the X model (the pre-whitened X series)
# The residuals (pwx) should be white noise
pwx = results_X.resid
# Remove first observation which is lost due to AR(1) model fitting
pwx = pwx.iloc[1:] 

# C. Apply the SAME filter (derived from X) to the Y series
# The AR(1) model is (1 - phi*B)X_t = W_t where B is the backshift operator.
# The filter coefficients for the AR component are [1, -phi]
phi = results_X.params['ar.L1']
# Use lfilter (linear filter) to apply the filter to Y
# Filter coefficients 'a' for AR part: [1, -phi]
# Filter coefficients 'b' for MA part: [1]
# Note: lfilter uses 'b' for numerator and 'a' for denominator
# Since we are applying the inverse filter (1 - phi*B), we use 'a' for the filter coefficients
a_coeffs = [1, -phi] # AR coefficients for the inverse filter
b_coeffs = [1]
# Apply the filter: new_Y_t = Y_t - phi * Y_{t-1}
filtered_y = lfilter(b_coeffs, a_coeffs, Y)
# Convert back to Series and align indices with pwx
pwy = pd.Series(filtered_y[1:], index=pwx.index)


# 4. Plot the cross-correlation function of the pre-whitened series (pwx and pwy)
fig, ax = plt.subplots(figsize=(10, 4))
# Note: plot_ccf uses adjusted=True by default which handles some edge cases
plot_ccf(pwx, pwy, title="Pre-whitened CCF between X and Y", ax=ax, lags=20)
plt.show()
