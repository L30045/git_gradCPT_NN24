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

# regressors used for HRF reconstruction; VTC is added as a single impulse regressor
glm_trial_types = ['mnt-correct-stim']

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

    run_list      = []
    stim_list     = []
    vtc_stim_list = []

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

        # canonical regressor (value=1) for Gaussian kernel basis expansion
        new_stim_df = pd.DataFrame({
            'onset':      new_onsets,
            'duration':   np.zeros(n_ep),
            'value':      np.ones(n_ep),
            'trial_type': ['mnt-correct-stim'] * n_ep,
        })
        # VTC stored separately → added as a single impulse regressor below
        vtc_stim_df = pd.DataFrame({
            'onset':      new_onsets,
            'duration':   np.zeros(n_ep),
            'value':      vtc_vals,
            'trial_type': ['mnt-correct-vtc'] * n_ep,
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
        vtc_stim_list.append(vtc_stim_df)

    if not run_list:
        print(f"No valid runs for {key_name}")
        continue

    # build design matrix using cedalion (HRF + drift; no short-sep for EEG)
    dm_all = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM_eeg, None, None, stim_list)

    # concatenate runs along time (same call as run_model_EEG_inform.py)
    Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)
    run_unit = Y_all.pint.units

    # build single VTC impulse regressor: one sample per onset, amplitude = VTC value
    _, vtc_stim_all, _ = model.concatenate_runs(run_list, vtc_stim_list)
    n_time_total = Y_all.sizes['time']
    vtc_reg = np.zeros(n_time_total)
    for _, row in vtc_stim_all.iterrows():
        samp = int(round(row['onset'] * sfreq))
        if 0 <= samp < n_time_total:
            vtc_reg[samp] = row['value']
    chromo_vals = dm_all.common.coords['chromo'].values
    vtc_da = xr.DataArray(
        np.tile(vtc_reg[:, np.newaxis, np.newaxis], (1, 1, len(chromo_vals))),
        dims=['time', 'regressor', 'chromo'],
        coords={
            'time':      dm_all.common.time,
            'regressor': ['VTC mnt-correct-vtc'],
            'chromo':    chromo_vals,
        },
    )
    dm_all.common = xr.concat([dm_all.common, vtc_da], dim='regressor')

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

colors = {'mnt-correct-stim': 'r'}
labels = {'mnt-correct-stim': 'Mountain correct'}

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

#%% model comparison: GLM with vs without VTC — sub-283
# Fits two models for a single subject and reports:
#   ΔR²     — variance explained gain from adding VTC
#   F-test  — classical added-variable F-test (OLS approximation)
#   t / p   — RLM t-test on the VTC coefficient (from model B)

from scipy.stats import f as f_dist

subj_id_test  = 695
key_name_test = f"sub-{subj_id_test}"
print(f"\nModel comparison for {key_name_test}")

# ── data preparation (mirrors the main loop) ────────────────────────────────
single_subj_EEG_test, _ = eeg_preproc_subj_level(subj_id_test, preproc_params)
run_list_t, stim_list_t, vtc_stim_list_t = [], [], []
sfreq_t = None

for run_name in sorted(single_subj_EEG_test.keys()):
    if 'gradcpt' not in run_name:
        continue
    run_id = int(run_name.split('cpt')[-1])
    EEG = single_subj_EEG_test[run_name].copy()
    avail = [c for c in ch_names if c in EEG.ch_names]
    if not avail:
        continue
    EEG.pick(avail)
    ev_df = pd.read_csv(
        os.path.join(data_save_path, key_name_test,
                     f"{key_name_test}_task-gradCPT_run-{run_id:02d}_events.tsv"),
        sep='\t').copy()
    ev_df.loc[(ev_df['trial_type'] == 'mnt') & (ev_df['response_code'] == 0),  'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type'] == 'mnt') & (ev_df['response_code'] != 0),  'trial_type'] = 'mnt-incorrect-stim'
    stim_df = ev_df[ev_df['trial_type'].isin(select_events)].copy()
    sfreq_t    = EEG.info['sfreq']
    onsets_sec = stim_df['onset'].values
    events_arr = np.column_stack([
        (onsets_sec * sfreq_t).astype(int),
        np.zeros(len(onsets_sec), int), np.ones(len(onsets_sec), int)])
    epochs = mne.Epochs(EEG, events_arr, event_id=1, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    epochs.drop_bad(reject={'eeg': 150e-6})
    if len(epochs) == 0:
        continue
    keep_mask = np.isin((onsets_sec * sfreq_t).astype(int), epochs.events[:, 0])
    vtc_v = stim_df['VTC'].values[keep_mask].copy()
    vtc_v -= vtc_v.mean()
    epoch_data = epochs.get_data()
    n_ep, _, n_t = epoch_data.shape
    concat_data = epoch_data.transpose(1, 0, 2).reshape(len(avail), n_ep * n_t)
    epoch_dur  = n_t / sfreq_t
    new_times  = np.arange(n_ep * n_t) / sfreq_t
    new_onsets = np.arange(n_ep) * epoch_dur + abs(tmin)
    stim_list_t.append(pd.DataFrame({
        'onset': new_onsets, 'duration': np.zeros(n_ep),
        'value': np.ones(n_ep), 'trial_type': ['mnt-correct-stim'] * n_ep}))
    vtc_stim_list_t.append(pd.DataFrame({
        'onset': new_onsets, 'duration': np.zeros(n_ep),
        'value': vtc_v, 'trial_type': ['mnt-correct-vtc'] * n_ep}))
    eeg_da = xr.DataArray(
        concat_data[:, np.newaxis, :],
        dims=('channel', 'chromo', 'time'),
        coords={'channel': avail, 'chromo': ['eeg'], 'time': new_times,
                'samples': ('time', np.arange(n_ep * n_t))})
    eeg_da.time.attrs['units'] = 'second'
    run_list_t.append(eeg_da.pint.quantify('V'))

Y_t, _, _             = model.concatenate_runs(run_list_t, stim_list_t)
_, vtc_stim_all_t, _  = model.concatenate_runs(run_list_t, vtc_stim_list_t)

# ── model A: Gaussian-kernel HRF only (no VTC) ─────────────────────────────
dm_A = model.get_GLM_copy_from_pf_DM(run_list_t, cfg_GLM_eeg, None, None, stim_list_t)
print("Fitting model A (no VTC) ...")
res_A, _ = model.my_fit(Y_t, dm_A)

# ── model B: same + single VTC impulse regressor ────────────────────────────
dm_B = model.get_GLM_copy_from_pf_DM(run_list_t, cfg_GLM_eeg, None, None, stim_list_t)
n_t_tot = Y_t.sizes['time']
vtc_col = np.zeros(n_t_tot)
for _, row in vtc_stim_all_t.iterrows():
    s = int(round(row['onset'] * sfreq_t))
    if 0 <= s < n_t_tot:
        vtc_col[s] = row['value']
chromo_t = dm_B.common.coords['chromo'].values
dm_B.common = xr.concat([
    dm_B.common,
    xr.DataArray(
        np.tile(vtc_col[:, None, None], (1, 1, len(chromo_t))),
        dims=['time', 'regressor', 'chromo'],
        coords={'time': dm_B.common.time,
                'regressor': ['VTC mnt-correct-vtc'],
                'chromo': chromo_t})
], dim='regressor')
print("Fitting model B (with VTC) ...")
res_B, _ = model.my_fit(Y_t, dm_B)

# ── compute metrics per channel ──────────────────────────────────────────────
Y_raw      = Y_t.pint.dequantify().values        # (channel, chromo, time)
channels_t = list(Y_t.channel.values)
x_A = dm_A.common.sel(chromo='eeg').values       # (time, n_reg_A)
x_B = dm_B.common.sel(chromo='eeg').values       # (time, n_reg_B)
n, p_B = x_B.shape

rows = []
for i_ch, ch in enumerate(channels_t):
    y      = Y_raw[i_ch, 0, :]
    ss_tot = np.sum((y - y.mean()) ** 2)

    beta_A = res_A.loc[ch, 'eeg'].item().params.values
    ss_A   = np.sum((y - x_A @ beta_A) ** 2)
    r2_A   = 1.0 - ss_A / ss_tot

    beta_B = res_B.loc[ch, 'eeg'].item().params
    ss_B   = np.sum((y - x_B @ beta_B.values) ** 2)
    r2_B   = 1.0 - ss_B / ss_tot

    # F-test for the single added VTC regressor (OLS approximation)
    F_val = ((ss_A - ss_B) / 1) / (ss_B / (n - p_B))
    p_F   = 1.0 - f_dist.cdf(F_val, 1, n - p_B)

    # RLM t-test directly on the VTC coefficient
    vtc_t = res_B.loc[ch, 'eeg'].item().tvalues['VTC mnt-correct-vtc']
    vtc_p = res_B.loc[ch, 'eeg'].item().pvalues['VTC mnt-correct-vtc']

    rows.append(dict(
        channel=ch,
        R2_noVTC=r2_A, R2_vtc=r2_B, delta_R2=r2_B - r2_A,
        F_stat=F_val, p_F=p_F,
        VTC_t=vtc_t,  VTC_p=vtc_p,
    ))

df_cmp = pd.DataFrame(rows)
print(f"\n{'='*72}")
print(f"GLM model comparison — {key_name_test}")
print(f"{'='*72}")
print(df_cmp.to_string(index=False,
    formatters={
        'R2_noVTC': '{:.4f}'.format, 'R2_vtc':  '{:.4f}'.format,
        'delta_R2': '{:+.5f}'.format,
        'F_stat':   '{:.3f}'.format,  'p_F':    '{:.4f}'.format,
        'VTC_t':    '{:.3f}'.format,  'VTC_p':  '{:.4f}'.format,
    }
))

#%% ── visualize ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x_pos = np.arange(len(channels_t))
w = 0.35

axes[0].bar(x_pos - w/2, df_cmp['R2_noVTC'], w, label='No VTC',   color='steelblue')
axes[0].bar(x_pos + w/2, df_cmp['R2_vtc'],   w, label='With VTC', color='tomato')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([c.upper() for c in channels_t])
axes[0].set_ylabel('R²')
axes[0].set_title('R² by model')
axes[0].legend()

bar_colors = ['green' if v >= 0 else 'red' for v in df_cmp['delta_R2']]
axes[1].bar(x_pos, df_cmp['delta_R2'], color=bar_colors)
# for i, row in df_cmp.iterrows():
#     sig   = '* p<.05' if row['VTC_p'] < 0.05 else 'ns'
#     y_lbl = row['delta_R2'] + (2e-5 if row['delta_R2'] >= 0 else -4e-5)
#     va    = 'bottom' if row['delta_R2'] >= 0 else 'top'
#     axes[1].text(i, y_lbl, f"t={row['VTC_t']:.2f}\n{sig}",
#                  ha='center', va=va, fontsize=9)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([c.upper() for c in channels_t])
axes[1].set_ylabel('ΔR²  (with VTC − no VTC)')
axes[1].set_title('R² improvement  (RLM t-test on VTC β)')
axes[1].axhline(0, color='k', lw=0.8, ls='--')
plt.suptitle(key_name_test, fontweight='bold')
plt.tight_layout()
plt.show()

#%% Band power GLM: does VTC predict city-correct trial band power?
# DV  : log10 band power per trial (theta / alpha / beta × channel)
# IV  : model A: 1 + onset_time (linear drift)
#        model B: 1 + onset_time + VTC  (full)
# Test: added-variable F-test for VTC, H0: β_VTC == 0  (df: 1, n-3)
# Results stored in df_bp_glm

import scipy.signal as sig_proc
from scipy.stats import f as f_dist_bp

bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

bp_glm_rows = []

for subj_id in tqdm(subj_id_array):
    key_name = f"sub-{subj_id}"
    single_subj_EEG_dict, _ = eeg_preproc_subj_level(subj_id, preproc_params)

    epoch_records = []    # one dict per city-correct epoch
    session_offset = 0.0
    sfreq_bp = None

    for run_name in sorted(single_subj_EEG_dict.keys()):
        if 'gradcpt' not in run_name:
            continue
        run_id  = int(run_name.split('cpt')[-1])
        EEG     = single_subj_EEG_dict[run_name].copy()
        avail   = [c for c in ch_names if c in EEG.ch_names]
        if not avail:
            continue
        EEG.pick(avail)
        sfreq_bp = EEG.info['sfreq']
        run_dur  = EEG.n_times / sfreq_bp

        ev_df = pd.read_csv(
            os.path.join(data_save_path, key_name,
                         f"{key_name}_task-gradCPT_run-{run_id:02d}_events.tsv"),
            sep='\t').copy()
        # city correct = correctly withheld (response_code > 0, per BIDS convention)
        stim_cty = ev_df[(ev_df['trial_type'] == 'city') &
                         (ev_df['response_code'] > 0)].copy()

        if len(stim_cty) == 0:
            session_offset += run_dur
            continue

        onsets_sec = stim_cty['onset'].values
        events_arr = np.column_stack([
            (onsets_sec * sfreq_bp).astype(int),
            np.zeros(len(onsets_sec), int),
            np.ones(len(onsets_sec),  int)])
        epochs_cty = mne.Epochs(EEG, events_arr, event_id=1,
                                tmin=tmin, tmax=tmax,
                                baseline=None, preload=True, verbose=False)
        epochs_cty.drop_bad(reject={'eeg': 150e-6})
        if len(epochs_cty) == 0:
            session_offset += run_dur
            continue

        keep_mask  = np.isin((onsets_sec * sfreq_bp).astype(int),
                             epochs_cty.events[:, 0])
        vtc_v      = stim_cty['VTC'].values[keep_mask].copy()
        onset_v    = onsets_sec[keep_mask] + session_offset
        epoch_data = epochs_cty.get_data()           # (n_ep, n_ch, n_t)

        # bandpass filter coefficients computed once per run (same sfreq)
        filt_coefs = {b: sig_proc.butter(4, [flo, fhi], btype='band', fs=sfreq_bp)
                      for b, (flo, fhi) in bands.items()}

        for i_ep in range(epoch_data.shape[0]):
            rec = {'onset_session': float(onset_v[i_ep]),
                   'vtc':           float(vtc_v[i_ep])}
            for i_ch, ch in enumerate(avail):
                for b_name, (bc, ac) in filt_coefs.items():
                    xf = sig_proc.filtfilt(bc, ac, epoch_data[i_ep, i_ch, :])
                    rec[f'{ch}_{b_name}'] = float(np.log10(np.var(xf) + 1e-30))
            for ch in ch_names:      # fill absent channels with NaN
                if ch not in avail:
                    for b_name in bands:
                        rec[f'{ch}_{b_name}'] = np.nan
            epoch_records.append(rec)

        session_offset += run_dur

    if len(epoch_records) < 5:
        print(f"  {key_name}: too few city-correct epochs, skipping")
        continue

    df_ep      = pd.DataFrame(epoch_records)
    onset_arr  = df_ep['onset_session'].values
    vtc_arr    = df_ep['vtc'].values - df_ep['vtc'].mean()
    n_tr       = len(onset_arr)
    onset_norm = (onset_arr - onset_arr.mean()) / (onset_arr.std() + 1e-10)

    X_A = np.column_stack([np.ones(n_tr), onset_norm])            # restricted
    X_B = np.column_stack([np.ones(n_tr), onset_norm, vtc_arr])   # full

    for ch in ch_names:
        for b_name in bands:
            col  = f'{ch}_{b_name}'
            y    = df_ep[col].values
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                continue
            y_v, Xa, Xb = y[mask], X_A[mask], X_B[mask]
            nv = int(mask.sum())

            bA, _, _, _ = np.linalg.lstsq(Xa, y_v, rcond=None)
            bB, _, _, _ = np.linalg.lstsq(Xb, y_v, rcond=None)
            ssA   = float(np.sum((y_v - Xa @ bA) ** 2))
            ssB   = float(np.sum((y_v - Xb @ bB) ** 2))
            ssTot = float(np.sum((y_v - y_v.mean()) ** 2))

            # F-test: H0: adding VTC does not improve fit (1 numerator df)
            F_val = max(ssA - ssB, 0.0) / max(ssB / (nv - 3), 1e-30)
            p_val = float(1.0 - f_dist_bp.cdf(F_val, 1, nv - 3))

            bp_glm_rows.append(dict(
                subject  = key_name,
                channel  = ch,
                band     = b_name,
                n_trials = nv,
                R2_noVTC = 1.0 - ssA / ssTot,
                R2_vtc   = 1.0 - ssB / ssTot,
                delta_R2 = (ssA - ssB) / ssTot,
                F_stat   = F_val,
                p_val    = p_val,
                VTC_beta = float(bB[2]),
            ))

# ── results table ────────────────────────────────────────────────────────────
df_bp_glm = pd.DataFrame(bp_glm_rows)

df_summary = (df_bp_glm
    .groupby(['channel', 'band'])
    .agg(
        n_subj        = ('subject',  'nunique'),
        mean_delta_R2 = ('delta_R2', 'mean'),
        mean_F        = ('F_stat',   'mean'),
        n_sig_p05     = ('p_val',    lambda x: int((x < 0.05).sum())),
        pct_sig_p05   = ('p_val',    lambda x: round(100 * (x < 0.05).mean(), 1)),
    )
    .reset_index()
)

print(f"\nBand power GLM — city-correct epochs")
print(f"  subjects: {df_bp_glm['subject'].nunique()}   "
      f"model: log(band_power) ~ 1 + onset_time [+ VTC]\n")
print(df_summary.to_string(index=False,
    formatters={'mean_delta_R2': '{:+.5f}'.format, 'mean_F': '{:.3f}'.format}))

# ── heatmap visualisation ─────────────────────────────────────────────────────
band_order = ['theta', 'alpha', 'beta']
ch_order   = ch_names

def _pivot(df_sum, metric):
    return (df_sum.pivot(index='channel', columns='band', values=metric)
            .reindex(index=ch_order, columns=band_order).values.astype(float))

dr2_mat  = _pivot(df_summary, 'mean_delta_R2')
psig_mat = _pivot(df_summary, 'pct_sig_p05')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vlim = max(np.nanmax(np.abs(dr2_mat)), 1e-6)
im0  = axes[0].imshow(dr2_mat, aspect='auto', cmap='RdBu_r',
                      vmin=-vlim, vmax=vlim)
axes[0].set_xticks(range(len(band_order))); axes[0].set_xticklabels(band_order)
axes[0].set_yticks(range(len(ch_order)));   axes[0].set_yticklabels([c.upper() for c in ch_order])
axes[0].set_title('Mean ΔR²  (VTC added variance)')
plt.colorbar(im0, ax=axes[0])
for i in range(len(ch_order)):
    for j in range(len(band_order)):
        axes[0].text(j, i, f'{dr2_mat[i,j]:+.4f}',
                     ha='center', va='center', fontsize=7)

im1 = axes[1].imshow(psig_mat, aspect='auto', cmap='Reds', vmin=0, vmax=100)
axes[1].set_xticks(range(len(band_order))); axes[1].set_xticklabels(band_order)
axes[1].set_yticks(range(len(ch_order)));   axes[1].set_yticklabels([c.upper() for c in ch_order])
axes[1].set_title('% subjects  p < 0.05')
plt.colorbar(im1, ax=axes[1], label='%')
for i in range(len(ch_order)):
    for j in range(len(band_order)):
        axes[1].text(j, i, f"{psig_mat[i,j]:.0f}%",
                     ha='center', va='center', fontsize=9)

plt.suptitle('Band power GLM — city-correct | F-test for added VTC regressor',
             fontweight='bold')
plt.tight_layout()
plt.show()
