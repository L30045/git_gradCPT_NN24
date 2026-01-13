
#%% load library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import pickle
import glob
import time
import sys
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices

import cedalion
import cedalion.nirs
from cedalion import units
import cedalion.models.glm as glm
from cedalion.sigproc import quality
from cedalion import units
import cedalion.sigproc.motion_correct as motion
from cedalion.plots import scalp_plot
from scipy.signal import filtfilt, windows
import xarray as xr


#%% helper function
def make_design_matrix(X, winlen=None):
    # flat time series
    X = X.flatten()
    # assign IRF window length
    if not winlen:
        winlen = len(X)
    # create design_matrix
    design_matrix = []
    for t_i in range(winlen):
        shift_S = np.concatenate([np.zeros(t_i), X[:len(X)-t_i]])
        design_matrix.append(shift_S)
    design_matrix = np.stack(design_matrix,axis=1)
    return design_matrix


def estimate_HRF_cov(cov, basis_hrf):

    basis_hrf = basis_hrf.rename({'component':'regressor_c'})
    basis_hrf = basis_hrf.assign_coords(regressor_c=cov.regressor_c.values)

    tmp = xr.dot(cov, basis_hrf, dims='regressor_c')

    tmp = tmp.rename({'regressor_r':'regressor'})
    basis_hrf = basis_hrf.rename({'regressor_c':'regressor'})

    mse_t = xr.dot(basis_hrf, tmp, dims='regressor')

    return mse_t

def estimate_HRF_from_beta(betas, basis_hrf):
        
    basis_hrf = basis_hrf.rename({'component':'regressor'})
    basis_hrf = basis_hrf.assign_coords(regressor=betas.regressor.values)

    hrf_estimate = xr.dot(betas, basis_hrf, dims='regressor')

    hrf_estimates_blcorr = hrf_estimate - hrf_estimate.sel(time = hrf_estimate.time[hrf_estimate.time<0]).mean('time')

    return hrf_estimates_blcorr

def get_drift_regressors(runs, cfg_GLM):
    
    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):
        drift = glm.design_matrix.drift_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)
        
    return drift_regressors

def get_drift_legendre_regressors(runs, cfg_GLM):

    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_legendre_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)

    return drift_regressors

def get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM):
    ss_regressors = []
    i=0
    for run, pruned_chans in zip(runs, pruned_chans_list):

        rec_pruned = prune_mask_ts(run, pruned_chans) # !!! how is this affected when using pruned data
        _, ts_short = cedalion.nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )

        short = glm.design_matrix.average_short_channel_regressor(ts_short)
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def prune_mask_ts(ts, pruned_chans):
    '''
    Function to mask pruned channels with NaN .. essentially repruning channels
    Parameters
    ----------
    ts : data array
        time series from rec[rec_str].
    pruned_chans : list or array
        list or array of channels that were pruned prior.

    Returns
    -------
    ts_masked : data array
        time series that has been "repruned" or masked with data for the pruned channels as NaN.

    '''
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked

def find_zero_crossings(signal, times):
    """
    Find zero crossing indices in a signal.

    Parameters:
    -----------
    signal : array
        The signal data
    times : array
        Time vector corresponding to the signal

    Returns:
    --------
    zero_crossing_times : array
        Times where zero crossings occur
    zero_crossing_indices : array
        Indices where zero crossings occur
    """
    # Find sign changes (zero crossings)
    sign_changes = np.diff(np.sign(signal))
    zero_crossing_indices = np.where(sign_changes != 0)[0]

    # Interpolate to get exact crossing times
    zero_crossing_times = []
    for idx in zero_crossing_indices:
        # Linear interpolation to find exact zero crossing
        t1, t2 = times[idx], times[idx + 1]
        v1, v2 = signal[idx], signal[idx + 1]
        # t_zero = t1 - v1 * (t2 - t1) / (v2 - v1)
        t_zero = t1 + (0 - v1) * (t2 - t1) / (v2 - v1)
        zero_crossing_times.append(t_zero)

    return np.array(zero_crossing_times), zero_crossing_indices

def calculate_component_area(signal, times, start_idx, end_idx):
    """
    Calculate the area under a component using trapezoidal integration.

    Parameters:
    -----------
    signal : array
        The signal data
    times : array
        Time vector corresponding to the signal
    start_idx : int
        Start index of the component
    end_idx : int
        End index of the component

    Returns:
    --------
    area : float
        Area under the curve (can be negative)
    """
    return np.trapz(signal[start_idx:end_idx+1], times[start_idx:end_idx+1])

def extract_n2_p3_features(signal, times, n2_window=(0.4, 0.7), p3_window=(0.6, 1.1)):
    """
    Extract N2 and P3 features from ERP signal.

    Parameters:
    -----------
    signal : array
        The ERP signal
    times : array
        Time vector in seconds
    n2_window : tuple
        Time window (start, end) in seconds to search for N2 peak
    p3_window : tuple
        Time window (start, end) in seconds to search for P3 peak

    Returns:
    --------
    results : dict
        Dictionary containing:
        - zero_crossing_times: all zero crossing time points
        - n2_peak_time: time of N2 peak
        - n2_peak_amp: amplitude of N2 peak
        - n2_area: area of N2 component
        - p3_peak_time: time of P3 peak
        - p3_peak_amp: amplitude of P3 peak
        - p3_area: area of P3 component
    """
    # Find all zero crossings
    zero_crossing_times, zero_crossing_indices = find_zero_crossings(signal, times)

    # Find N2 (negative peak in specified window)
    n2_mask = (times >= n2_window[0]) & (times <= n2_window[1])
    n2_idx = np.argmin(signal[n2_mask])
    n2_global_idx = np.where(n2_mask)[0][n2_idx]
    n2_peak_time = times[n2_global_idx]
    n2_peak_amp = signal[n2_global_idx]

    # Find N2 boundaries (zero crossings around N2 peak)
    n2_start_crossings = zero_crossing_indices[zero_crossing_indices < n2_global_idx]
    n2_end_crossings = zero_crossing_indices[zero_crossing_indices > n2_global_idx]

    if len(n2_start_crossings) > 0 and len(n2_end_crossings) > 0:
        n2_start_idx = n2_start_crossings[-1]
        n2_end_idx = n2_end_crossings[0]
        n2_area = calculate_component_area(signal, times, n2_start_idx, n2_end_idx)
    else:
        n2_area = np.nan
        n2_start_idx = None
        n2_end_idx = None

    # Find P3 (positive peak in specified window)
    p3_mask = (times >= p3_window[0]) & (times <= p3_window[1])
    p3_idx = np.argmax(signal[p3_mask])
    p3_global_idx = np.where(p3_mask)[0][p3_idx]
    p3_peak_time = times[p3_global_idx]
    p3_peak_amp = signal[p3_global_idx]

    # Find P3 boundaries (zero crossings around P3 peak)
    p3_start_crossings = zero_crossing_indices[zero_crossing_indices < p3_global_idx]
    p3_end_crossings = zero_crossing_indices[zero_crossing_indices > p3_global_idx]

    if len(p3_start_crossings) > 0 and len(p3_end_crossings) > 0:
        p3_start_idx = p3_start_crossings[-1]
        p3_end_idx = p3_end_crossings[0]
        p3_area = calculate_component_area(signal, times, p3_start_idx, p3_end_idx)
    else:
        p3_area = np.nan
        p3_start_idx = None
        p3_end_idx = None

    results = {
        'zero_crossing_times': zero_crossing_times,
        'zero_crossing_indices': zero_crossing_indices,
        'n2_peak_time': n2_peak_time,
        'n2_peak_amp': n2_peak_amp,
        'n2_area': n2_area,
        'n2_start_idx': n2_start_idx,
        'n2_end_idx': n2_end_idx,
        'p3_peak_time': p3_peak_time,
        'p3_peak_amp': p3_peak_amp,
        'p3_area': p3_area,
        'p3_start_idx': p3_start_idx,
        'p3_end_idx': p3_end_idx
    }

    return results