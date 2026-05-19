"""
Estimate ERP via GLM deconvolution on continuous EEG using cedalion.

Builds a Gaussian-kernel design matrix on the continuous EEG signal (same
framework as the NIRS GLM in run_model_EEG_inform.py) and fits with AR-IRLS.
The betas reconstructed through estimate_HRF_from_beta give the ERP estimate
and correctly account for overlapping responses from adjacent gradCPT trials.
"""
#%% load library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import sys
import pandas as pd
import xarray as xr
import glob
import time
from tqdm import tqdm
from functools import reduce
import operator

git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from utils import *
from params_setting import *
import model

#%% parameters
# subj_id_array = [695]
subj_id_array = [670, 671, 673, 695, 719, 721, 723, 726, 727, 730, 733, 746, 751, 755]

ch_names = ['fz', 'cz', 'pz', 'oz']

# trial types used to filter the events TSV
select_events = ['mnt-correct-stim']

# regressors actually entered into the GLM design matrix
glm_trial_types = ['mnt-correct-stim', 'mnt-correct-vtc']

tmin = -0.2   # epoch start relative to stimulus onset (s)
tmax =  1.2   # epoch end (s)

cfg_GLM_eeg = {
    'do_drift': False,
    'do_drift_legendre': False,
    'do_short_sep': False,       # no short-separation channels for EEG
    'drift_order': 3,
    'noise_model': 'ar_irls',
    't_delta': 0.02 * units.s,  # spacing between Gaussian basis functions
    't_std':   0.02 * units.s,  # width of each Gaussian
    't_pre':   abs(tmin) * units.s,
    't_post':  tmax * units.s,
}

#%% helper

def mne_raw_to_xr(raw, ch_names_sel):
    """Convert MNE Raw to a cedalion-compatible xr.DataArray.

    Adds a dummy 'chromo' dimension (value 'eeg') so cedalion can locate
    the third dim it expects alongside 'channel' and 'time'.
    """
    available = [c for c in ch_names_sel if c in raw.ch_names]
    data = raw.get_data(picks=available)      # (n_ch, n_times)  units: V
    data = data[:, np.newaxis, :]              # (n_ch, 1, n_times)
    times = raw.times
    da = xr.DataArray(
        data,
        dims=('channel', 'chromo', 'time'),
        coords={
            'channel': available,
            'chromo':  ['eeg'],
            'time':    times,
            'samples': ('time', np.arange(len(times))),
        }
    )
    da.time.attrs['units'] = 'second'
    da = da.pint.quantify('V')
    return da, available

#%% compute GLM-ERP per subject

subj_glm_erp_dict = dict()

for subj_id in tqdm(subj_id_array):
    key_name = f"sub-{subj_id}"
    print(f"\nProcessing {key_name}")

    single_subj_EEG_dict, _ = eeg_preproc_subj_level(subj_id, preproc_params)

    run_list  = []
    stim_list = []

    for run_name in sorted(single_subj_EEG_dict.keys()):
        if 'gradcpt' not in run_name:
            continue
        run_id = int(run_name.split('cpt')[-1])

        EEG = single_subj_EEG_dict[run_name].copy()
        available_chs = [c for c in ch_names if c in EEG.ch_names]
        if not available_chs:
            continue
        EEG.pick(available_chs)

        # build stim DataFrame — mnt-correct-stim only
        event_file = os.path.join(
            data_save_path, key_name,
            f"{key_name}_task-gradCPT_run-{run_id:02d}_events.tsv"
        )
        ev_df = pd.read_csv(event_file, sep='\t').copy()
        ev_df.loc[(ev_df['trial_type'] == 'mnt') & (ev_df['response_code'] == 0),  'trial_type'] = 'mnt-correct-stim'
        ev_df.loc[(ev_df['trial_type'] == 'mnt') & (ev_df['response_code'] != 0),  'trial_type'] = 'mnt-incorrect-stim'
        stim_df = ev_df[ev_df['trial_type'].isin(select_events)].copy()

        # ── Epoch → drop bad → concatenate epochs for GLM ───────────────────
        sfreq      = EEG.info['sfreq']
        onsets_sec = stim_df['onset'].values
        events_arr = np.column_stack([
            (onsets_sec * sfreq).astype(int),
            np.zeros(len(onsets_sec), dtype=int),
            np.ones(len(onsets_sec),  dtype=int),
        ])
        epochs = mne.Epochs(
            EEG, events_arr, event_id=1,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False,
        )
        epochs.drop_bad(reject={'eeg': 150e-6})
        n_kept = len(epochs)
        print(f"  {run_name}: {n_kept}/{len(onsets_sec)} mnt-correct epochs retained")
        if n_kept == 0:
            continue

        # identify which original trials survived rejection
        surviving_samps = epochs.events[:, 0]
        expected_samps  = (onsets_sec * sfreq).astype(int)
        keep_mask       = np.isin(expected_samps, surviving_samps)

        # VTC for surviving epochs, mean-centered per run
        vtc_vals = stim_df['VTC'].values[keep_mask]
        vtc_vals = vtc_vals - vtc_vals.mean()

        # stack epochs into a pseudo-continuous signal: (n_ch, n_ep * n_t)
        epoch_data  = epochs.get_data()              # (n_ep, n_ch, n_t)
        n_ep, _, n_t = epoch_data.shape
        concat_data = epoch_data.transpose(1, 0, 2).reshape(len(available_chs), n_ep * n_t)

        # recompute time axis and onset positions within the concatenated signal
        epoch_dur_sec = n_t / sfreq                  # tmax - tmin
        new_times     = np.arange(n_ep * n_t) / sfreq
        new_onsets    = np.arange(n_ep) * epoch_dur_sec + abs(tmin)

        # canonical regressor (value=1) + VTC parametric modulator (value=VTC)
        new_stim_df = pd.DataFrame({
            'onset':      np.tile(new_onsets, 2),
            'duration':   np.zeros(n_ep * 2),
            'value':      np.concatenate([np.ones(n_ep), vtc_vals]),
            'trial_type': ['mnt-correct-stim'] * n_ep + ['mnt-correct-vtc'] * n_ep,
        })

        eeg_da = xr.DataArray(
            concat_data[:, np.newaxis, :],
            dims=('channel', 'chromo', 'time'),
            coords={
                'channel': available_chs,
                'chromo':  ['eeg'],
                'time':    new_times,
                'samples': ('time', np.arange(n_ep * n_t)),
            }
        )
        eeg_da.time.attrs['units'] = 'second'
        eeg_da = eeg_da.pint.quantify('V')
        # ────────────────────────────────────────────────────────────────────

        run_list.append(eeg_da)
        stim_list.append(new_stim_df)

    if not run_list:
        print(f"No valid runs for {key_name}")
        continue

    # build design matrix using cedalion (HRF + drift; no short-sep for EEG)
    dm_all = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM_eeg, None, None, stim_list)

    # concatenate runs along time (same call as run_model_EEG_inform.py)
    Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)
    run_unit = Y_all.pint.units
    sample_run = run_list[0].copy()
    sample_run = sample_run.assign_coords(samples=('time', np.arange(sample_run.sizes['time'])))
    sample_run.time.attrs['units'] = units.s
# fit GLM with AR-IRLS
    print(f"Fitting GLM for {key_name} ...")
    glm_results, _ = model.my_fit(Y_all, dm_all)

    # reconstruct ERP from betas (estimate_HRF_from_beta includes baseline correction)
    betas     = glm_results.sm.params
    basis_hrf = model.glm.GaussianKernels(
        cfg_GLM_eeg['t_pre'], cfg_GLM_eeg['t_post'],
        cfg_GLM_eeg['t_delta'], cfg_GLM_eeg['t_std']
    )(sample_run)

    subj_result = dict()
    for trial_type in glm_trial_types:
        betas_hrf    = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = model.estimate_HRF_from_beta(betas_hrf, basis_hrf)
        subj_result[trial_type] = hrf_estimate

    subj_glm_erp_dict[key_name] = subj_result

#%% plot GLM-ERP

colors = {'mnt-correct-stim': 'r', 'mnt-correct-vtc': 'b'}
labels = {'mnt-correct-stim': 'Mountain correct', 'mnt-correct-vtc': 'Mountain correct × VTC'}

for ch in ch_names:
    plt.figure(figsize=(10, 6))
    for ev_name in glm_trial_types:
        subj_erps = []
        for subj_result in subj_glm_erp_dict.values():
            if ev_name not in subj_result:
                continue
            try:
                erp = subj_result[ev_name].sel(channel=ch, chromo='eeg').values
                subj_erps.append(erp)
            except KeyError:
                pass
        if not subj_erps:
            continue
        erps      = np.vstack(subj_erps)
        n_subj    = erps.shape[0]
        mean_erp  = np.mean(erps, axis=0)
        sem_erp   = np.std(erps, axis=0) / np.sqrt(n_subj)
        time_vec  = list(subj_glm_erp_dict.values())[-1][ev_name].time.values
        plt.plot(time_vec * 1000, mean_erp,
                 color=colors[ev_name], linewidth=2,
                 label=f'{labels[ev_name]} (n={n_subj})')
        plt.fill_between(time_vec * 1000,
                         mean_erp - 2 * sem_erp,
                         mean_erp + 2 * sem_erp,
                         alpha=0.3, color=colors[ev_name])
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'GLM-ERP  {ch.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
