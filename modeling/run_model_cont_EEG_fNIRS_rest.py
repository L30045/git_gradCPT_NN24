#%% load library
import numpy as np
import pickle
import gzip
import glob
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
import re
import xarray as xr
import cedalion
import cedalion.io
from cedalion import units
from cedalion.sigproc import motion, frequency, quality
import cedalion.nirs
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf
from statsmodels.gam.smooth_basis import BSplines

#%% find subjects with high-quality RS fNIRS and available resting EEG
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')


def _find_rest_eeg_fif(subj_id):
    """Locate the preprocessed resting-state EEG .fif, tolerating naming variants."""
    subj_eeg_dir = os.path.join(_eeg_deriv, f'sub-{subj_id}')
    for fname in [
        f'sub-{subj_id}_task-Rest_run-01_preproc_eeg.fif',
        f'sub-{subj_id}_task_Rest_run-01_preproc_eeg.fif',  # sub-751
        f'sub-{subj_id}_task-Rest_run01_preproc_eeg.fif',   # sub-695
    ]:
        path = os.path.join(subj_eeg_dir, fname)
        if os.path.isfile(path):
            return path
    return None


# subjects with a resting-state fNIRS snirf (task-RS)
_rs_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(project_path, 'sub-*', 'nirs', '*task-RS*nirs.snirf'))
    if re.search(r'sub-(\d+)', f)
}

# subjects with a resting-state preprocessed EEG file
_rest_eeg_subjects = {
    sid for sid in _rs_fnirs_subjects if _find_rest_eeg_fif(sid) is not None
}

subj_id_array = [int(s) for s in sorted(_rs_fnirs_subjects & _rest_eeg_subjects)]

# check if any of subject in subj_id_array is in the excluded_subj (low fNIRS quality)
subj_id_array = [x for x in subj_id_array if f'sub-{x}' not in excluded_subj]

#%% select model type
eeg_reg_type = 'cont_EEG_cz'
is_overwrite = True  # If True, force re-training GLM.
is_save = True  # If True, save DM and GLM results
is_norm = False  # If True, z-score regressors.
is_plot = False  # If True, generate visualization plots
select_chromo = 'HbO'
select_channel = None  # set after loading data if None (first pruned-good channel)
USE_GSR = True
cfg_GLM['do_GSR'] = USE_GSR
len_delay = 15  # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay  # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)

#%% RS fNIRS channel-space preprocessing config (mirrors STEP1_image_recon_per_subj.py, channel-space only)
cfg_prune = {
    'snr_thresh': 5,
    'sd_thresh': [1, 40] * units.mm,
    'amp_thresh': [1e-5, 0.84] * units.V,
}

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)
    os.makedirs(data_save_path, exist_ok=True)

    # check if betas.pkl exist already. If yes, skip this subject.
    betas_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_rest_betas.pkl')
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% LOAD RESTING-STATE fNIRS (channel space; no image-space projection available for RS)
    nirs_dir = os.path.join(project_path, subject, 'nirs')
    rs_snirf_path = os.path.join(nirs_dir, f'{subject}_task-RS_run-01_nirs.snirf')
    if not os.path.isfile(rs_snirf_path):
        print(f"{subject}: RS snirf not found, skipping.")
        continue

    records = cedalion.io.read_snirf(rs_snirf_path)
    rec = records[0]
    rec['amp'] = rec['amp'].pint.dequantify().pint.quantify('V')
    rec['amp'] = quality.repair_amp(rec['amp'], median_len=1)

    geo3d = rec.geo3d
    rec, chs_pruned = pf.prune_channels(
        rec, cfg_prune['amp_thresh'], cfg_prune['sd_thresh'], cfg_prune['snr_thresh']
    )

    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength},
    )

    rec["od_o"] = cedalion.nirs.cw.int2od(rec['amp'])
    rec['od_o'].time.attrs['units'] = units.s
    rec['od_o'] = rec['od_o'].where(~rec['od_o'].isnull(), 1e-18)
    rec['conc_o'] = cedalion.nirs.cw.od2conc(rec['od_o'], geo3d, dpf)

    fnirs_run = rec['conc_o']
    if select_chromo is not None:
        fnirs_run = fnirs_run.sel(chromo=[select_chromo])

    # keep only good channels (chs_pruned == 0.4 marks a channel that passed all QC checks)
    good_channels = chs_pruned.channel.values[chs_pruned.values == 0.4]
    if len(good_channels) == 0:
        print(f"{subject}: no good fNIRS channels after pruning, skipping.")
        continue
    fnirs_run = fnirs_run.sel(channel=good_channels)

    if select_channel is None:
        subj_select_channel = good_channels[0]
    else:
        subj_select_channel = select_channel

    #%% get continuous resting-state EEG
    rest_fif_path = _find_rest_eeg_fif(subj_id)
    if rest_fif_path is None:
        print(f"{subject}: resting EEG not found, skipping.")
        continue

    EEG_raw = mne.io.read_raw_fif(rest_fif_path, preload=True, verbose=False)

    # check if Cz exists (it was never dropped during preprocessing for this run)
    if 'cz' in eeg_reg_type and 'cz' not in [ch.lower() for ch in EEG_raw.ch_names]:
        print(f"{subject}: Cz not present in resting EEG, skipping.")
        continue

    #%% align EEG and fNIRS clocks using the shared digital trigger pulse
    if 'Trigger' not in EEG_raw.ch_names:
        print(f"{subject}: no Trigger channel in resting EEG, skipping.")
        continue
    trig_data = EEG_raw.copy().pick('Trigger').get_data()[0]
    thresh = trig_data.max() / 2
    eeg_crossings = np.where((trig_data[:-1] < thresh) & (trig_data[1:] >= thresh))[0]
    if len(eeg_crossings) == 0:
        print(f"{subject}: no trigger onset found in resting EEG, skipping.")
        continue
    eeg_trigger_t = float(EEG_raw.times[eeg_crossings[0]])

    digital = rec.aux_ts['digital'].values.flatten()
    t_aux = rec.aux_ts['digital'].time.values
    nirs_onsets = np.where(np.diff(digital) > 0)[0]
    if len(nirs_onsets) == 0:
        print(f"{subject}: no trigger onset found in resting fNIRS, skipping.")
        continue
    nirs_trigger_t = float(t_aux[nirs_onsets[0]])

    # nirs_time = eeg_time + t_offset
    t_offset = nirs_trigger_t - eeg_trigger_t

    # shared window: from the trigger onset to the end of whichever recording is shorter
    nirs_t_start = nirs_trigger_t
    nirs_t_stop = fnirs_run.time.values[-1]
    eeg_t_start = nirs_t_start - t_offset
    eeg_t_stop = nirs_t_stop - t_offset

    EEG_raw.crop(tmin=max(eeg_t_start, 0), tmax=min(eeg_t_stop, EEG_raw.times[-1]))

    #%% resample EEG to fNIRS sampling rate
    fnirs_sfreq = 1 / np.diff(fnirs_run.time.values).mean()
    fnirs_run = fnirs_run.sel(time=slice(nirs_t_start, nirs_t_stop))
    n_fnirs_samples = len(fnirs_run.time)

    EEG_resample = EEG_raw.copy()
    EEG_resample.resample(fnirs_sfreq, npad='auto')

    if EEG_resample.n_times > n_fnirs_samples:
        EEG_resample.crop(tmax=EEG_resample.times[n_fnirs_samples - 1])
    elif EEG_resample.n_times < n_fnirs_samples:
        fnirs_run = fnirs_run.isel(time=slice(0, EEG_resample.n_times))
        n_fnirs_samples = EEG_resample.n_times

    # truncate fNIRS so the delay at the beginning of the recording is removed
    fnirs_run = fnirs_run.isel(time=slice(np.round(len_delay * fnirs_sfreq).astype(int), n_fnirs_samples))

    # reset fnirs_run.time to 0
    fnirs_run = fnirs_run.assign_coords(time=fnirs_run.time.values - fnirs_run.time.values[0])
    fnirs_run.time.attrs['units'] = units.s

    all_runs = [fnirs_run]
    eeg_list = [EEG_resample]

    #%% Extract EEG values for DM
    eeg_reg_value_list = [x.get_data(picks='cz').flatten() for x in eeg_list]

    # create EEG regressors
    eeg_regressors = model.get_cont_EEG_regressor(eeg_reg_value_list, fnirs_sfreq, delay=len_delay, z_score=is_norm)

    # concatenate all runs and dms (single RS run, but reuse the same helper for consistency)
    Y_all, dm_all, runs_updated = model.concatenate_runs_dms(all_runs, eeg_regressors)

    #%% Low-rank representation of Delay using BSpline
    all_regressor_names = dm_all.common.regressor.values
    n_regressor = len(all_regressor_names)
    delay_idx = np.arange(n_regressor)
    bspline_basis = BSplines(delay_idx, df=[n_bspline_basis], degree=[bspline_degree],
                              include_intercept=True).basis  # (n_regressor, n_bspline_basis)
    basis_da = xr.DataArray(
        bspline_basis,
        dims=("regressor", "component"),
        coords={"regressor": all_regressor_names,
                "component": [f"bspline{i}" for i in range(n_bspline_basis)]},
    )
    dm_all.common = xr.dot(dm_all.common, basis_da, dims="regressor").rename({"component": "regressor"})

    #%% Combine drift and GSR regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = model.get_drift_regressors(runs_updated, cfg_GLM)
        dm_all &= model.reduce(model.operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = model.get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dm_all &= model.reduce(model.operator.and_, drift_regressors)

    if cfg_GLM['do_GSR']:
        gsr = model.get_global_mean_regressor(runs_updated)
        dm_all &= model.reduce(model.operator.and_, gsr)

    dm_all.common = dm_all.common.fillna(0)

    if is_norm:
        dm_all.common = (dm_all.common - dm_all.common.mean('time')) / dm_all.common.std('time')

    #%% select HbO to fasten training process
    dm_all.common = dm_all.common.sel(chromo=[select_chromo])

    #%% get GLM fitting results for each subject
    print(f"Start cont_EEG GLM fitting ({subject})")
    import cedalion.models.glm as glm
    results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
    betas_all = results.sm.params.copy()
    eeg_reg = [p for p in betas_all.regressor.values if 'bspline' in p]
    betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
    betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])

    #%% visual check fit results and HRF
    if is_plot:
        vis_betas = results.sm.params.copy()
        eeg_reg = [p for p in vis_betas.regressor.values if 'bspline' in p]
        vis_betas = vis_betas.sel(regressor=eeg_reg)
        y_hat = xr.dot(dm_all.common, vis_betas, dims='regressor')

        fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=False)
        y_true = Y_all.sel(channel=subj_select_channel).values.flatten()
        y_hat_vals = y_hat.sel(channel=subj_select_channel).values.flatten()

        axs[0].plot(Y_all.time.values, y_true, label='Y (true)', color='k', linewidth=2)
        axs[0].plot(y_hat.time.values, y_hat_vals, 'b', label='y_hat (EEG)', alpha=0.5)
        axs[0].set_ylabel('HbO concentration')
        axs[0].set_title(f'Channel activity estimation ({subj_select_channel})')
        axs[0].legend()
        axs[0].grid()
        t_betas = np.arange(0, len_delay, 1 / fnirs_sfreq)
        axs[1].plot(t_betas, betas_eeg.sel(channel=subj_select_channel).values.flatten(),
                    label=f'HRF ({subj_select_channel})')
        axs[1].set_title('HRF estimation using resting-state Cz EEG')
        axs[1].legend()
        axs[1].grid()

    #%% save betas for later visualization
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg'] = betas_eeg
        betas_dict['betas_bspline'] = betas_bspline
        betas_dict['basis_da'] = basis_da
        betas_dict['chs_pruned'] = chs_pruned
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)
