#%% load library
import numpy as np
import pickle
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from utils import *
import model
from params_setting import *
import statsmodels.api as sm
from scipy import stats

#%% load data

np.random.seed(42)
n = 100

# True predictors with real effects
X1 = np.random.randn(n)  # Predictor 1
X2 = np.random.randn(n)  # Predictor 2
X3 = np.random.randn(n)  # Predictor 3 (additional)
X4 = np.random.randn(n)  # Predictor 4 (additional)

# Generate y with known relationships
y = 2 + 1.5*X1 + 0.8*X2 + 0.5*X3 + 0.3*X4 + np.random.randn(n)*0.5

# REDUCED MODEL: Only X1 and X2
X_reduced = np.column_stack([X1, X2])
X_reduced = sm.add_constant(X_reduced)  # Add intercept

# FULL MODEL: X1, X2, X3, and X4
X_full = np.column_stack([X1, X2, X3, X4])
X_full = sm.add_constant(X_full)  # Add intercept

# Fit both models
model_reduced = sm.OLS(y, X_reduced).fit()
model_full = sm.OLS(y, X_full).fit()

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Method 1: Using statsmodels built-in compare_f_test
print("\n--- Method 1: Built-in compare_f_test ---")
f_stat, p_value, df_diff = model_full.compare_f_test(model_reduced)
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"df difference: {df_diff}")

# Method 2: Manual calculation (to understand the math)
print("\n--- Method 2: Manual Calculation ---")

# Get residual sum of squares
rss_reduced = model_reduced.ssr
rss_full = model_full.ssr

# Degrees of freedom
n_obs = len(y)
p_reduced = model_reduced.df_model  # number of predictors (excluding intercept)
p_full = model_full.df_model
df_numerator = p_full - p_reduced  # number of additional predictors
df_denominator = n_obs - p_full - 1

# Calculate F-statistic
f_stat_manual = ((rss_reduced - rss_full) / df_numerator) / (rss_full / df_denominator)
p_value_manual = 1 - stats.f.cdf(f_stat_manual, df_numerator, df_denominator)

print(f"RSS (reduced): {rss_reduced:.4f}")
print(f"RSS (full): {rss_full:.4f}")
print(f"Reduction in RSS: {rss_reduced - rss_full:.4f}")
print(f"\nF-statistic: {f_stat_manual:.4f}")
print(f"p-value: {p_value_manual:.6f}")
print(f"df: ({df_numerator}, {df_denominator})")

# Display R-squared comparison
print("\n--- R-squared Comparison ---")
print(f"Reduced model R²: {model_reduced.rsquared:.4f}")
print(f"Full model R²: {model_full.rsquared:.4f}")
print(f"Adjusted R² (reduced): {model_reduced.rsquared_adj:.4f}")
print(f"Adjusted R² (full): {model_full.rsquared_adj:.4f}")

# Display AIC/BIC comparison
print("\n--- Information Criteria ---")
print(f"AIC (reduced): {model_reduced.aic:.2f}")
print(f"AIC (full): {model_full.aic:.2f}")
print(f"BIC (reduced): {model_reduced.bic:.2f}")
print(f"BIC (full): {model_full.bic:.2f}")

# Interpretation
print("\n--- Interpretation ---")
alpha = 0.05
if p_value < alpha:
    print(f"✓ The additional predictors significantly improve the model (p < {alpha})")
    print("  Reject H₀: The additional regressors explain significant additional variance")
else:
    print(f"✗ The additional predictors do NOT significantly improve the model (p ≥ {alpha})")
    print("  Fail to reject H₀: The additional regressors may not be necessary")

# Show individual model summaries
print("\n" + "=" * 60)
print("REDUCED MODEL SUMMARY")
print("=" * 60)
print(model_reduced.summary())

print("\n" + "=" * 60)
print("FULL MODEL SUMMARY")
print("=" * 60)
print(model_full.summary())