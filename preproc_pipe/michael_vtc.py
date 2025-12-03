#%%
import numpy as np
from scipy.signal import filtfilt, windows
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

#%% --- smoothing function ---
L = 20 # trials
W_ori = windows.gaussian(L, std=L/6)  # MATLAB gausswin(L) uses std = (L-1)/6; using L/6 is close
W = W_ori / W_ori.sum()                   # normalize

fig,ax = plt.subplots(2,1)
ax[0].plot(W_ori)
ax[1].plot(W)
print(f"winlen = {len(W_ori)}")

#%% --- get RTs ---
RT = response[:, 4].copy()        # MATLAB columns are 1-indexed, Python is 0-indexed
RT = RT[:len(response)-1]         # cut last null trial

# --- get all non-zero RTs and compute mean/std ---
RT2 = RT[RT > 0]
meanRT = np.mean(RT2)
stdRT = np.std(RT2, ddof=0)       # MATLAB std(...,1) = population std

# --- identify no-RT trials as NaN ---
RT_with_nan = RT.astype(float).copy()
RT_with_nan[RT_with_nan == 0] = np.nan

# --- fill NaNs (MATLAB inpaint_nans) using interpolation ---
# linear interpolation over non-NaN points
x = np.arange(len(RT_with_nan))
good = ~np.isnan(RT_with_nan)
interp_func = interp1d(x[good], RT_with_nan[good], kind='linear', fill_value='extrapolate')
RT_filled = interp_func(x)

# --- z-transform and absolute value ---
zRT = (RT_filled - meanRT) / stdRT
abs_zRT = np.abs(zRT)

# --- ORIGINAL VTC: filtfilt with Gaussian kernel ---
VARIABILITY_TC = filtfilt(W, [1], abs_zRT)   # forward/backward filter
MEDIAN_VAR = np.median(VARIABILITY_TC)
