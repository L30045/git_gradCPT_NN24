
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
import copy

from tqdm import tqdm
import cedalion
import cedalion.nirs
from cedalion import units
import cedalion.models.glm as glm
from cedalion.sigproc import quality
from cedalion import units
import cedalion.sigproc.motion_correct as motion
from cedalion.plots import scalp_plot
from scipy.signal import filtfilt, windows, lfilter
from statsmodels.tsa.stattools import acf, pacf
import xarray as xr
import cedalion.xrutils as xrutils
import scipy
from statsmodels.stats.stattools import durbin_watson
import cedalion.typing as cdt
from cedalion.models.glm.design_matrix import DesignMatrix
from joblib import Parallel, delayed, parallel_config

from functools import reduce
import operator
import cedalion.sigproc.frequency as frequency

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

def extract_comp_features(signal, times, window_list=[(0.4,0.7),(0.6,1.1)], comp_sign_list=['neg','pos']):
    """
    Extract component features from ERP signal.

    Parameters:
    -----------
    signal : array
        The ERP signal
    times : array
        Time vector in seconds
    window_list : list of tuple
        List of time window (start, end) in seconds to search for component peak
    comp_sign_list : list of str
        Sign of component of interest 
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - zero_crossing_times: all zero crossing time points
        - zero_crossing_indices: all zero crossing time index
        - window_of_intesret_list: list of time window (start, end) in seconds to search for component peak
        - components: componet features
    """
    # Find all zero crossings
    zero_crossing_times, zero_crossing_indices = find_zero_crossings(signal, times)

    result_list = []
    # For each component
    for c_i in range(len(window_list)):
        # Find N2 (negative peak in specified window)
        comp_window = window_list[c_i]
        comp_mask = (times >= comp_window[0]) & (times <= comp_window[1])
        comp_sign = comp_sign_list[c_i]
        if comp_sign=='pos':
            comp_idx = np.argmax(signal[comp_mask])
        else:
            comp_idx = np.argmin(signal[comp_mask])
        comp_global_idx = np.where(comp_mask)[0][comp_idx]
        comp_peak_time = times[comp_global_idx]
        comp_peak_amp = signal[comp_global_idx]

        # Find component boundaries (zero crossings around N2 peak)
        comp_start_crossings = zero_crossing_indices[zero_crossing_indices < comp_global_idx]
        comp_end_crossings = zero_crossing_indices[zero_crossing_indices > comp_global_idx]

        if len(comp_start_crossings) > 0 and len(comp_end_crossings) > 0:
            comp_start_idx = comp_start_crossings[-1]
            comp_end_idx = comp_end_crossings[0]
            comp_area = calculate_component_area(signal, times, comp_start_idx, comp_end_idx)
        else:
            # use the area within the time window
            comp_area = calculate_component_area(signal, times, np.argwhere(comp_mask)[0][0], np.argwhere(comp_mask)[-1][0])
            comp_start_idx = comp_window[0]
            comp_end_idx = comp_window[1]
        # save results
        result = {
            'comp_peak_time': comp_peak_time,
            'comp_peak_amp': comp_peak_amp,
            'comp_area': comp_area,
            'comp_sign': comp_sign,
            'comp_start_idx': comp_start_idx,
            'comp_end_idx': comp_end_idx,
            'window_of_interest': comp_window
        }
        result_list.append(result)

    results = {
        'zero_crossing_times': zero_crossing_times,
        'zero_crossing_indices': zero_crossing_indices,
        'window_of_intesret_list': window_list,
        'components': result_list
    }

    return results

# get event trials
def get_valid_event_idx(ev_name, single_subj_epoch_dict):
    ev_idx_dict = dict()
    for run_key in single_subj_epoch_dict.keys():
        ev_idx_dict[run_key] = dict()
        if len(single_subj_epoch_dict[run_key][ev_name])>0:
            # get preserved trial index
            ev_preserved_idx = np.where([len(log) == 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
            # get rejected trial index
            ev_rejected_idx = np.where([len(log) != 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
            # add dict
            ev_idx_dict[run_key]['preserved'] = ev_preserved_idx
            ev_idx_dict[run_key]['rejected'] = ev_rejected_idx
        else:
            ev_idx_dict[run_key]['preserved'] = []
            ev_idx_dict[run_key]['rejected'] = []
    return ev_idx_dict

# get ERP area
def get_ERP_area(ev_name, single_subj_epoch_dict, is_norm=True):
    # define output dict
    erp_area_dict = dict()
    # define ERP period of interest
    if ev_name.endswith('response'):
        window_list=[(0,0.2)]
        comp_sign_list = ['neg']
    else:
        window_list=[(0.4, 0.7),(0.6, 1.1)]
        comp_sign_list = ['neg', 'pos']
    # for each run
    for run_key in single_subj_epoch_dict.keys():
        erp_area_dict[run_key]=dict()
        if len(single_subj_epoch_dict[run_key][ev_name])>0:
            t_vector = single_subj_epoch_dict[run_key][ev_name].times
            # for each channel, extract ERP area
            ev_eeg = single_subj_epoch_dict[run_key][ev_name].pick(picks='eeg')
            for ch_name in ev_eeg.ch_names:
                ev_ch_eeg = ev_eeg.get_data()[:,ev_eeg.ch_names.index(ch_name),:]            
                # Extract component features
                area_list = []
                for eeg_i in range(len(ev_ch_eeg)):
                    comp_results = extract_comp_features(ev_ch_eeg[eeg_i], t_vector, window_list=window_list, comp_sign_list=comp_sign_list)
                    # summing up all component area
                    comp_area = np.sum([np.abs(x['comp_area']) for x in comp_results['components']])
                    area_list.append(comp_area)
                # rescale area to range 0 to 1. (0 as 0, 1 as max(area))
                area_list = np.array(area_list)
                if is_norm:
                    area_list = area_list/np.max(area_list)
                # store results
                erp_area_dict[run_key][ch_name] = area_list
        else:
            erp_area_dict[run_key] = []
    return erp_area_dict

# add events to design matrix
# add events per run.
def add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=None, select_chs=['cz'], model_mode='stim'):
    """
    select_chs: select channels to add to design matrix
    """
    dm_dict = dict()
    for run_key in run_dict.keys():
        dm_dict[run_key] = dict()
        target_run = run_dict[run_key]['run']
        conc_o = run_dict[run_key]['conc_ts']
        chs_pruned = run_dict[run_key]['chs_pruned']
        ev_df = run_dict[run_key]['ev_df']
        # for each run, get drift and short-separation regressors (if any)
        if cfg_GLM['do_drift']:
            drift_regressors = get_drift_regressors([conc_o], cfg_GLM)
        elif cfg_GLM['do_drift_legendre']:
            drift_regressors = get_drift_legendre_regressors([conc_o], cfg_GLM)
        else:
            drift_regressors = None
        if cfg_GLM['do_short_sep']:
            ss_regressors = get_short_regressors([conc_o], [chs_pruned], cfg_GLM['geo3d'], cfg_GLM)
        else:
            ss_regressors = None
        # for each event, create a dm list
        if not select_event:
            select_event = ev_dict[run_key].keys()
        ev_dms = []
        for ev_name in select_event:
            if ev_name=='mnt_correct':
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt_correct'
            elif ev_name=='mnt_incorrect':
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt_incorrect'
            # check if event exist
            if len(target_ev_df)==0:
                # store in dm_dict
                dm_dict[run_key][ev_name] = []
                continue
            # create stim onset regressors
            stim_ev_df = copy.deepcopy(target_ev_df)
            # rename trial_type
            stim_ev_df.loc[:,'trial_type'] = np.unique(stim_ev_df['trial_type'])[0]+'_stim'
            # build stim-only design matrix
            if model_mode.endswith('reduced'):
                stim_dm = glm.design_matrix.hrf_regressors(
                                                target_run,
                                                stim_ev_df.iloc[ev_dict[run_key][ev_name]['idx']['preserved']],
                                                glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                            )
            else:
                stim_dm = glm.design_matrix.hrf_regressors(
                                                target_run,
                                                stim_ev_df,
                                                glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                            )
            # check if model include EEG-informed regressors
            if not model_mode.lower().startswith('stim'):
                # create design matrix
                dm_list = []
                # for each event, rescale and create a dm
                for ev_i, event_id in enumerate(ev_dict[run_key][ev_name]['idx']['preserved']):
                    dm = glm.design_matrix.hrf_regressors(
                                                target_run,
                                                target_ev_df.iloc[[event_id]],
                                                glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                            )
                    # rescale by Cz area
                    #TODO: allow multiple channels in DM
                    dm.common = dm.common*ev_dict[run_key][ev_name]['area']['cz'][ev_i]
                    # append
                    dm_list.append(dm)
                # Create a new design matrix object with the concatenated common regressors
                dms = dm_list.pop()
                dms_common = dms.common
                # merge all dms along time axis
                while len(dm_list)>0:
                    dms_common += dm_list.pop().common
                # assign merged common back to dms
                dms.common = dms_common
            # build complete model
            if model_mode.lower().startswith('full'):
                # full model
                dms &= reduce(operator.and_, [stim_dm])
            elif model_mode.lower().startswith('stim'):
                # stim-only model
                dms = stim_dm
            # store event DM
            ev_dms.append(dms)
        # combine all event DMs into one big DM
        combined_dm = ev_dms.pop()
        while len(ev_dms)>0:
            ev_dm = ev_dms.pop()
            combined_dm &= reduce(operator.and_, [ev_dm])
            combined_dm.common = combined_dm.common.fillna(0)
        
        # add drift and short-separation regressors if any
        if drift_regressors:
            combined_dm &= reduce(operator.and_, drift_regressors)
            combined_dm.common = combined_dm.common.fillna(0)
        if ss_regressors:
            combined_dm &= reduce(operator.and_, ss_regressors)
            combined_dm.common = combined_dm.common.fillna(0)
        # store in dm_dict
        dm_dict[run_key] = combined_dm
    
    return dm_dict

def create_no_info_dm(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units

    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)

    # merge all DMs
    dms = ss_regressors.pop()
    if len(ss_regressors)>1:
        dms &= reduce(operator.and_, ss_regressors)
    elif len(ss_regressors)>0:
        dms &= reduce(operator.and_, [ss_regressors])
    dms &= reduce(operator.and_, drift_regressors)
    dms.common = dms.common.fillna(0)

    return dms

def create_eeg_dm(run_dict, ev_dict, cfg_GLM, select_event=None, select_chs=['cz']):
    """
    select_chs: select channels to add to design matrix
    """
    dm_dict = dict()
    for run_key in run_dict.keys():
        dm_dict[run_key] = dict()
        target_run = run_dict[run_key]['run']
        conc_o = run_dict[run_key]['conc_ts']
        chs_pruned = run_dict[run_key]['chs_pruned']
        ev_df = run_dict[run_key]['ev_df']
        
        # for each event, create a dm list
        if not select_event:
            select_event = ev_dict[run_key].keys()
        ev_dms = []
        for ev_name in select_event:
            if ev_name=='mnt_correct':
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt-correct-eeg'
            elif ev_name=='mnt_incorrect':
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt-incorrect-eeg'
            # check if event exist
            if len(target_ev_df)==0:
                # store in dm_dict
                dm_dict[run_key][ev_name] = []
                continue
            
            # EEG-informed regressors
            
            # create design matrix
            dm_list = []
            # for each event, rescale and create a dm
            for ev_i, event_id in enumerate(ev_dict[run_key][ev_name]['idx']['preserved']):
                dm = glm.design_matrix.hrf_regressors(
                                            target_run,
                                            target_ev_df.iloc[[event_id]],
                                            glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                        )
                # rescale by Cz area
                #TODO: allow multiple channels in DM
                dm.common = dm.common*ev_dict[run_key][ev_name]['area']['cz'][ev_i]
                # append
                dm_list.append(dm)
            # Create a new design matrix object with the concatenated common regressors
            dms = dm_list.pop()
            dms_common = dms.common
            # merge all dms along time axis
            while len(dm_list)>0:
                dms_common += dm_list.pop().common
            # assign merged common back to dms
            dms.common = dms_common
            
            # store event DM
            ev_dms.append(dms)
        # combine all event DMs into one big DM
        combined_dm = ev_dms.pop()
        while len(ev_dms)>0:
            ev_dm = ev_dms.pop()
            combined_dm &= reduce(operator.and_, [ev_dm])
            combined_dm.common = combined_dm.common.fillna(0)
        
        # store in dm_dict
        dm_dict[run_key] = combined_dm
    
    return dm_dict

def get_GLM_copy_from_pf_DM(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )


    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    return dms

def combine_dm(eeg_dm, reduced_dm):
    eeg_dm &= reduce(operator.and_, [reduced_dm])
    return eeg_dm

# visualize DM
def vis_dm(plt_dm):
    # using xr.DataArray.plot
    f, ax = plt.subplots(1,1,figsize=(12,10))
    # plt_dm.common.sel(chromo="HbO", time=plt_dm.common.time<600).T.plot(vmin=-2,vmax=2)
    plt_dm.common.sel(chromo="HbO").T.plot(vmin=-2,vmax=2)
    #p.xticks(rotation=90)
    plt.show()


# GLM model from pf.GLM()
# Unknown pf.GLM() loaded. Required only 4 inputs (no geo3d).
def GLM_copy_from_pf(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )


    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
    betas = results.sm.params
    cov_params = results.sm.cov_params()

    # 4. estimate HRF and MSE
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

    hrf_mse_list = []
    hrf_estimate_list = []

    for trial_type in trial_type_list:
        
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(run_unit)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(run_unit**2)

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0]).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse

def concatenate_runs_dms(run_dict, dm_dict):
    """
    Concatenate multiple runs along time dimension for joint analysis.
    
    Combines time series and stimulus timing from multiple runs into single
    continuous arrays. Adjusts time coordinates and stimulus onsets to maintain
    temporal continuity. Enables fitting a single GLM across all runs.
    
    """

    CURRENT_OFFSET = 0
    runs_updated = []
    dm_updated = []

    for run_key in run_dict.keys():
        ts = run_dict[run_key]['run']
        dm = dm_dict[run_key]
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.dequantify().pint.quantify('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        dm_new = copy.deepcopy(dm)
        dm_new.common = dm_new.common.assign_coords(time=new_time)

        runs_updated.append(ts_new)
        dm_updated.append(dm_new.common)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    dm_all = copy.deepcopy(dm_dict[run_key])
    dm_all.common = xr.concat(dm_updated, dim="time")
    dm_all.common = dm_all.common.fillna(0)
    
    return Y_all, dm_all, runs_updated

def concatenate_runs(runs, stim):
    """
    Concatenate multiple runs along time dimension for joint analysis.
    
    Combines time series and stimulus timing from multiple runs into single
    continuous arrays. Adjusts time coordinates and stimulus onsets to maintain
    temporal continuity. Enables fitting a single GLM across all runs.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of concentration time series, one per run, with dimensions
        (channel, chromo, time).
    stim : list of pd.DataFrame
        List of stimulus DataFrames, one per run, with columns:
        ['onset', 'duration', 'trial_type']
    
    Returns
    -------
    Y_all : xr.DataArray
        Concatenated time series with dimensions (channel, chromo, time).
        Time coordinates adjusted to be continuous across runs.
    stim_df : pd.DataFrame
        Concatenated stimulus DataFrame with adjusted onset times.
    runs_updated : list of xr.DataArray
        List of runs with updated time coordinates (for design matrix construction).
        
    Notes
    -----
    Time offset for each run is computed as: offset_i = last_time_{i-1} + dt
    All runs are converted to 'molar' units before concatenation.
    Maintains sampling rate continuity between runs.
    """

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for s, ts in zip(stim, runs):
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.dequantify().pint.quantify('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim_shift = s.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated

def calculate_ar_loglikelihood(y, X, beta, ar_coefs, sigma2):
    """
    Calculate log-likelihood for AR-IRLS model.
    
    Parameters:
    -----------
    y : array
        Observed data
    X : array
        Design matrix
    beta : array
        Regression coefficients
    ar_coefs : array
        AR coefficients [φ₁, φ₂, ..., φₚ]
    sigma2 : float
        Error variance
    
    Returns:
    --------
    ll : float
        Log-likelihood
    """
    n = len(y)
    p = len(ar_coefs)
    
    # Compute residuals
    residuals = y - X @ beta
    
    # Apply AR transformation to residuals
    # (This is simplified - actual implementation depends on AR order)
    transformed_resid = np.zeros(n)
    transformed_resid[:p] = residuals[:p]  # First p observations
    
    for t in range(p, n):
        ar_term = sum(ar_coefs[i] * residuals[t-i-1] for i in range(p))
        transformed_resid[t] = residuals[t] - ar_term
    
    # Calculate log-likelihood
    # For AR(p) errors: y ~ N(Xβ, Σ) where Σ depends on AR structure
    
    # Simplified version (assumes transformed residuals ~ N(0, σ²))
    ss_transformed = np.sum(transformed_resid**2)
    
    ll = -n/2 * np.log(2 * np.pi * sigma2) - ss_transformed / (2 * sigma2)
    
    # Add correction for Jacobian of AR transformation
    # (This term depends on AR coefficients)
    ar_poly = np.poly1d([1] + (-ar_coefs).tolist())
    roots = np.roots(ar_poly)
    if np.all(np.abs(roots) > 1):  # Check stationarity
        jacobian_term = n * np.log(1 - np.sum(ar_coefs**2)) / 2
        ll += jacobian_term
    
    return ll


#%% Define my AR-IRLS
def my_ar_irls_GLM(y, x, pmax=30, autoReg=None, M=sm.robust.norms.HuberT()):
    mask = np.isfinite(y.values)

    yorg : pd.Series = pd.Series(y.values[mask].copy())
    xorg : pd.DataFrame = x[mask].reset_index(drop=True)

    y = yorg.copy()
    x = xorg.copy()

    # check if autocorrelation sturcture exist
    if autoReg is not None:
        # use autoReg to prewhiten data and fit reduced model
        # Apply the AR filter to the lhs and rhs of the model
        yf = prewhiten_arp_lfilter(y, autoReg[1:])
        xf = prewhiten_design_matrix_lfilter(x, autoReg[1:])

        rlm_model = sm.RLM(yf, xf, M=M)
        params = rlm_model.fit()

        return params, autoReg

    rlm_model = sm.RLM(y, x, M=M)
    params = rlm_model.fit()

    resid = pd.Series(y - x @ params.params)
    for _ in range(4):  # TODO - check convergence
        y = yorg.copy()
        x = xorg.copy()

        # Update the AR whitening filter
        arcoef = cedalion.math.ar_model.bic_arfit(resid, pmax=pmax)

        # Apply the AR filter to the lhs and rhs of the model
        yf = prewhiten_arp_lfilter(y, arcoef.params[1:])
        xf = prewhiten_design_matrix_lfilter(x, arcoef.params[1:])

        rlm_model = sm.RLM(yf, xf, M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params, arcoef.params

def my_fit(
    ts: cdt.NDTimeSeries,
    design_matrix: DesignMatrix,
    autoReg: None,
    ar_order: int = 30,
    max_jobs: int = -1,
    verbose: bool = False,
):
    # FIXME: unit handling?
    # shoud the design matrix be dimensionless? -> thetas will have units
    ts = ts.pint.dequantify()

    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


    reg_results = xr.DataArray(
        np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
        dims=("channel", dim3_name),
        coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
    )
    autoReg_dict = dict()

    for (
        dim3,
        group_channels,
        group_design_matrix,
    ) in design_matrix.iter_computational_groups(ts):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
            "time", "channel"
        )

        # pass x as a DataFrame to statsmodel to make it aware of regressor names
        x = pd.DataFrame(
            group_design_matrix.values, columns=group_design_matrix.regressor.values
        )

        if(max_jobs==1):
            for chan in tqdm(group_y.channel.values, disable=not verbose):
                if autoReg is not None:
                    result = my_ar_irls_GLM(group_y.loc[:, chan], x, autoReg=autoReg[chan])
                else:    
                    result = my_ar_irls_GLM(group_y.loc[:, chan], x, pmax=ar_order)
                reg_results.loc[chan, dim3] = result[0]
                autoReg_dict[chan] = result[1]
        else:
            args_list=[]
            for chan in group_y.channel.values:
                if autoReg is not None:
                    args_list.append([group_y.loc[:, chan], x, ar_order, autoReg[chan]])
                else:
                    args_list.append([group_y.loc[:, chan], x, ar_order])

            with parallel_config(backend='threading', n_jobs=max_jobs):
                batch_results = tqdm(
                    Parallel(return_as="generator")(
                        delayed(my_ar_irls_GLM)(*args) for args in args_list
                    ),
                    total=len(args_list)
                )

            for chan, result in zip(group_y.channel.values, batch_results):
                reg_results.loc[chan, dim3] = result[0]
                autoReg_dict[chan] = result[1]


    description='AR_IRLS' # FIXME
    reg_results.attrs["description"] = description

    return reg_results, autoReg_dict

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

    NOTE:
    -------
    Mathematically, we should scale the first element to keep the variance
    structure. However, the importance of scaling is small when there are 
    more than 500 samples.

    To keep things simple, I ignore the scaling issue here.
    - Chi 2026-02-10
    """
        
    # Filter coefficients 
    b = np.array([1.0]) 
    a = np.concatenate([[1.0], -rho_coefficients])
    
    # Apply filter (using IIR as FIR, lfilter(a,b,data) instead of lfilter(b,a,data))
    data_white = lfilter(a, b, data)
    
    return data_white

def prewhiten_design_matrix_lfilter(X, rho_coefficients):
    """
    Pre-whiten design matrix using lfilter
    
    Parameters:
    -----------
    X : dataFrame

    rho_coefficients : array (ar_order,) or float
        AR coefficients
    
    Returns:
    --------
    X_white : dataFrame
    """
    # Handle scalar rho (AR(1))
    if np.isscalar(rho_coefficients):
        rho_coefficients = np.array([rho_coefficients])
    
    # Handle 1D array
    if X.ndim == 1:
        return prewhiten_arp_lfilter(X, rho_coefficients)
    
    # Handle 2D array (multiple regressors)
    X_white = pd.DataFrame(np.zeros_like(X),
                            columns=X.columns)
    
    for col in X.columns:
        X_white[col] = prewhiten_arp_lfilter(X[col], rho_coefficients)
    
    return X_white

def vis_fit(y, x, beta):
    resid = y-x@beta
    fig, ax = plt.subplots(1,1)
    ax.plot(y,label='y')
    ax.plot(x@beta,label='x@beta')
    ax.legend()
    plt.show()

def check_if_whiten(yf):
    # Quick Check 1: ACF at lag 1
    print(f"ACF[1] of yf: {acf(yf, nlags=1, fft=False)[1]:.6f}")
    print(f"Should be < {1.96/np.sqrt(len(yf)):.6f}")

    # Quick Check 2: Durbin-Watson
    print(f"DW statistic: {durbin_watson(yf):.4f} (should be between 1.5 and 2.5)")

    # Quick Check 3: Visual
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(acf(y, nlags=20, fft=False), 'o-', label='y')
    plt.plot(acf(yf, nlags=20, fft=False), 'o-', label='yf')
    plt.axhline(1.96/np.sqrt(len(y)), color='r', linestyle='--')
    plt.axhline(-1.96/np.sqrt(len(y)), color='r', linestyle='--')
    plt.legend()
    plt.title('ACF Comparison')
    plt.subplot(122)
    plt.plot(yf[:200])
    plt.title('Whitened Series')
    plt.tight_layout()
    plt.show()

def extract_val_across_channels(f_test_result, chromo='HbO', stat_val='p'):
    val = np.zeros(len(f_test_result.sel(chromo=chromo)))
    for ch_i, ch_result in enumerate(f_test_result.sel(chromo=chromo)):
        tmp_result = ch_result.values.item().summary()
        val[ch_i] = float(tmp_result.split(stat_val)[1].split(',')[0][1:])
    return val


