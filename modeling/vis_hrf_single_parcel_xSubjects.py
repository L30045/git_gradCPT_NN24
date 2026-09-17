#%% Visualize the HRF of a single parcel across subjects, for both the event-based
# HRF GLM (CORRECT_CODE_runGLM_on_image.py-style 3-step fit, refit per subject here)
# and the continuous-EEG GLM (betas saved by run_model_cont_EEG_fNIRS.py, loaded per
# subject as in vis_hrf_cont_EEG_cz.py). One figure per parcel in select_parcels,
# each showing all subjects' HRFs plus the group mean +/- 95% CI.
import os
import gzip
import pickle
import copy
import glob
import re

import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt

import cedalion
import cedalion.io
from cedalion import units
from cedalion.sigproc import frequency
import cedalion.models.glm as glm
from scipy.signal import filtfilt, windows

import sys
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf

from params_setting import *

import warnings
warnings.filterwarnings('ignore')

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
# (matches the mask applied in run_model_cont_EEG_fNIRS.py / check_yhat_AR-IRLS_image_space.py)
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = cedalion.io.load_Adot(Adot_path + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
mask_medial = Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH',
    'Background+FreeSurfer_Defined_Medial_Wall_RH'
])
Adot_brain = Adot_brain.sel(vertex=~mask_medial)

intensity = np.log10(Adot_brain[:, :, 1].sum('channel'))
sensitivity_mask = (intensity > -2).drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values)
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex')
Adot_parcel = Adot_parcel.assign_coords(
    {'is_brain': ('parcel', np.ones(len(Adot_parcel.parcel), dtype=bool))}
)
sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (601 -> 417)

# pick the single most-sensitive parcel (highest total forward-model sensitivity,
# summed over channel/wavelength/vertex) to fit, rather than a fixed name
parcel_total_sensitivity = Adot_parcel.sum(['channel', 'wavelength'])
most_sensitive_parcel = str(parcel_total_sensitivity.parcel.values[np.argmax(parcel_total_sensitivity.values)])

# also grab the next 5 most-sensitive parcels overall (ranks 2-6)
sens_order = np.argsort(parcel_total_sensitivity.values)[::-1]
next_most_sensitive_parcels = [str(parcel_total_sensitivity.parcel.values[i]) for i in sens_order[1:6]]

# also pick the most-sensitive parcel within the DorsAttn network
dorsattn_parcels = [p for p in parcel_total_sensitivity.parcel.values if str(p).startswith('DorsAttn')]
dorsattn_sens = parcel_total_sensitivity.sel(parcel=dorsattn_parcels)
most_sensitive_dorsattn_parcel = str(dorsattn_sens.parcel.values[np.argmax(dorsattn_sens.values)])

# also pick the most-sensitive parcel within the SomMot (somatomotor) network
sommot_parcels = [p for p in parcel_total_sensitivity.parcel.values if str(p).startswith('SomMot')]
sommot_sens = parcel_total_sensitivity.sel(parcel=sommot_parcels)
most_sensitive_sommot_parcel = str(sommot_sens.parcel.values[np.argmax(sommot_sens.values)])

#%% parcel(s) to visualize (matches the choices offered in check_yhat_AR-IRLS_image_space.py)
select_parcels = [
    "SalVentAttnA_FrMed_5_LH",
    most_sensitive_parcel,
    *next_most_sensitive_parcels,
    most_sensitive_dorsattn_parcel,
    most_sensitive_sommot_parcel,
]
select_chromo = 'HbO'
eeg_reg_type = 'cont_EEG_cz_add_15s'
is_hp_fNIRS = False
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
len_delay = 15  # must match run_model_cont_EEG_fNIRS.py's len_delay

for p in select_parcels:
    assert p in sensitive_parcels, f"{p} was excluded by the sensitivity mask"

#%% subject list: subjects with saved continuous-EEG GLM outputs (Fit 2) that also
# have image-space results (Fit 1), excluding subjects already flagged for low fNIRS quality
eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
betas_files = sorted(glob.glob(os.path.join(
    eeg_der_dir, 'sub-*', f'sub-*_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl')))

der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

subjects = []
for f in betas_files:
    m = re.search(r'sub-(\d+)', f)
    subject = f'sub-{m.group(1)}'
    if subject in excluded_subj:
        continue
    image_path = os.path.join(
        der_dir, subject,
        f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl')
    if not os.path.exists(image_path):
        continue
    subjects.append(subject)
print(f'Found {len(subjects)} subjects with both Fit 1 and Fit 2 outputs: {subjects}')


#%% ---- Fit 1 (event-based, 3-step): per-subject data prep + fit, reused from
# check_yhat_AR-IRLS_image_space.py's three_step_fit_event_based() ----
def load_event_based_runs(subject, select_parcel):
    """Load and preprocess this subject's image-space parcel time series for the
    event-based fit: mask low-sensitivity parcels, select select_parcel (+ an
    all-parcel copy for GSR), reorder/crop runs to match the gradCPT event files."""
    with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
        results_chs = pickle.load(f)
    all_stims = results_chs['stims']

    folder = os.path.join(der_dir, subject)
    filepath = folder + f'/{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'
    with open(filepath, 'rb') as f:
        image_results = pickle.load(f)
    all_runs = image_results['parcel_ts']

    L = 20
    W = windows.gaussian(L, std=L / 6) / 2

    if SPLIT_VTC:
        possible_trial_types = ['mnt-correct-in', 'mnt-correct-out', 'mnt-incorrect', 'city-incorrect']
    else:
        possible_trial_types = ['mnt-correct', 'mnt-incorrect']

    stims_pruned_list = []
    all_runs_tmp = []
    for stim, run in zip(all_stims, all_runs):
        mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
        mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
        mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

        if SPLIT_VTC:
            VTC = stim['VTC'].to_numpy()
            VTC = filtfilt(W, sum(W), VTC)
            median = np.median(VTC)
            in_zone = np.where(VTC <= median)[0]
            out_zone = np.where(VTC > median)[0]
            mnt_trials.loc[
                (mnt_trials['trial_type'] == 'mnt-correct') & (mnt_trials.index.isin(in_zone)),
                'trial_type'
            ] = 'mnt-correct-in'
            mnt_trials.loc[
                (mnt_trials['trial_type'] == 'mnt-correct') & (mnt_trials.index.isin(out_zone)),
                'trial_type'
            ] = 'mnt-correct-out'

        if F_MIN > 0:
            run.time.attrs['units'] = units.s
            run_filt = frequency.freq_filter(run, F_MIN * units.Hz, F_MAX * units.Hz)
            all_runs_tmp.append(run_filt)
        else:
            all_runs_tmp.append(run)

        stims_pruned_list.append(mnt_trials)

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs_tmp]

    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel=run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp.copy()

    all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels)) for run in all_runs]

    all_runs_allparcel = [x.sel(chromo=[select_chromo]) for x in all_runs]
    all_runs = [x.sel(parcel=[select_parcel], chromo=[select_chromo]) for x in all_runs]

    nirs_ev_files = sorted(glob.glob(os.path.join(root_dir, subject, 'nirs', f"{subject}_task-gradCPT_run-*_events.tsv")))
    nirs_ev_dfs = {f: pd.read_csv(f, sep='\t') for f in nirs_ev_files}

    run_key_to_run_idx = dict()
    run_key_to_nirs_df = dict()
    for run_num, (nirs_file, nirs_df) in enumerate(nirs_ev_dfs.items(), start=1):
        nirs_onset0 = nirs_df['onset'].values[0]
        for r_i, stim in enumerate(all_stims):
            if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onset0, atol=0.01):
                run_key_to_run_idx[f'gradcpt{run_num}'] = r_i
                run_key_to_nirs_df[f'gradcpt{run_num}'] = nirs_df
                break

    assert len(run_key_to_run_idx) == len(all_runs), "could not match all runs to a gradcpt run key"
    run_keys = [f'gradcpt{i}' for i in range(1, len(all_runs) + 1)]
    reorder_idx = [run_key_to_run_idx[k] for k in run_keys]
    all_runs = [all_runs[i] for i in reorder_idx]
    all_runs_allparcel = [all_runs_allparcel[i] for i in reorder_idx]

    cropped_runs = []
    cropped_runs_allparcel = []
    for run_key, run, run_ap in zip(run_keys, all_runs, all_runs_allparcel):
        nirs_df = run_key_to_nirs_df[run_key]
        nirs_t_start = nirs_df['onset'].values[0]
        nirs_t_stop = nirs_df['onset'].values[-1] + len_delay

        run_c = run.sel(time=slice(max(nirs_t_start, run.time.values[0]),
                                    min(nirs_t_stop, run.time.values[-1])))
        run_c = run_c.assign_coords(time=run_c.time.values - run_c.time.values[0])
        run_c.time.attrs['units'] = units.s
        cropped_runs.append(run_c)

        run_ap_c = run_ap.sel(time=slice(max(nirs_t_start, run_ap.time.values[0]),
                                          min(nirs_t_stop, run_ap.time.values[-1])))
        run_ap_c = run_ap_c.assign_coords(time=run_ap_c.time.values - run_ap_c.time.values[0])
        run_ap_c.time.attrs['units'] = units.s
        cropped_runs_allparcel.append(run_ap_c)

    stims_pruned_list = [stims_pruned_list[i] for i in reorder_idx]

    all_runs = [r.pint.dequantify() for r in cropped_runs]
    all_runs_allparcel = [r.pint.dequantify() for r in cropped_runs_allparcel]

    return all_runs, all_runs_allparcel, stims_pruned_list


def ols_regress_out_per_run(target_runs, regressor_dms):
    resid_runs = []
    fit_runs = []
    for run, dm in zip(target_runs, regressor_dms):
        results = glm.fit(run, dm, noise_model='ols')
        betas = results.sm.params
        fit_vals = xr.dot(dm.common, betas, dims='regressor').transpose(*run.dims)
        resid_runs.append(run - fit_vals)
        fit_runs.append(fit_vals)
    return resid_runs, fit_runs


def three_step_fit_event_based(subject, select_parcel):
    """3-step sequential regression for the event-based fit (Fit 1), for one subject
    and one parcel: OLS drift -> OLS GSR -> AR-IRLS HRF regressors. Returns the
    event-triggered HRF shape per trial type."""
    all_runs, all_runs_allparcel, stims_pruned_list = load_event_based_runs(subject, select_parcel)

    drift_dms = [glm.design_matrix.drift_legendre_regressors(r, cfg_GLM['drift_order']) for r in all_runs]
    runs_resid1, _ = ols_regress_out_per_run(all_runs, drift_dms)
    runs_ap_resid1, _ = ols_regress_out_per_run(all_runs_allparcel, drift_dms)

    gsr_dms = pf.get_global_mean_regressor(runs_ap_resid1)
    runs_resid2, _ = ols_regress_out_per_run(runs_resid1, gsr_dms)

    Y_resid2, stim_df, _ = pf.concatenate_runs(runs_resid2, stims_pruned_list)
    hrf_kernel = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    hrf_dm = glm.design_matrix.hrf_regressors(Y_resid2, stim_df, hrf_kernel)
    hrf_dm.common = hrf_dm.common.fillna(0)

    print(f"Start event-based AR-IRLS fitting on drift+GSR-residualized signal ({subject}, {select_parcel})")
    ar_results = glm.fit(Y_resid2, hrf_dm, noise_model=cfg_GLM['noise_model'])
    betas = ar_results.sm.params

    basis_hrf = hrf_kernel(Y_resid2)
    fs = frequency.sampling_rate(Y_resid2).to('Hz')
    dT = np.round(1 / fs, 3)
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    hrf_estimate_list = []
    for trial_type in stim_df['trial_type'].unique():
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f'HRF {trial_type}'))
        hrf_est = pf.estimate_HRF_from_beta(betas_hrf, basis_hrf)
        hrf_estimate_list.append(hrf_est.expand_dims({'trial_type': [trial_type]}))
    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    reltime = np.linspace(-before_samples * dT, after_samples * dT, len(hrf_estimate.time))
    hrf_estimate = hrf_estimate.assign_coords(time=reltime)
    hrf = hrf_estimate.sel(parcel=select_parcel, chromo=select_chromo)  # dims: time, trial_type

    return hrf


#%% ---- Fit 2 (continuous-EEG, 3-step): load betas saved by run_model_cont_EEG_fNIRS.py ----
def load_continuous_eeg_hrf(subject, select_parcel):
    """Load this subject's saved continuous-EEG delay-response curve for select_parcel
    (betas_eeg, already expanded from bspline components to per-delay-tap resolution
    by run_model_cont_EEG_fNIRS.py) and the matching delay time axis."""
    base = os.path.join(eeg_der_dir, subject, f'{subject}_{eeg_reg_type}_{NOISE_MODEL}_{hp_flag}')
    betas_path = base + '_betas.pkl'
    with open(betas_path, 'rb') as f:
        betas_dict = pickle.load(f)
    betas_eeg = betas_dict['betas_eeg']  # dims: parcel, chromo, component (delay tap)

    hrf_eeg = betas_eeg.sel(parcel=select_parcel, chromo=select_chromo).values
    n_delay_taps = hrf_eeg.shape[-1]
    t_delay = np.arange(n_delay_taps) * (len_delay / n_delay_taps)
    return t_delay, hrf_eeg


#%% ---- run both fits for every subject and every parcel ----
event_hrf_by_parcel = {p: dict() for p in select_parcels}   # parcel -> subject -> xr.DataArray (time, trial_type)
cont_hrf_by_parcel = {p: dict() for p in select_parcels}    # parcel -> subject -> (t_delay, hrf_eeg)

for select_parcel in select_parcels:
    for subject in subjects:
        try:
            hrf_event = three_step_fit_event_based(subject, select_parcel)
            event_hrf_by_parcel[select_parcel][subject] = hrf_event
        except Exception as e:
            print(f"{subject}: event-based fit failed for {select_parcel} ({e}), skipping.")

        try:
            t_delay, hrf_eeg = load_continuous_eeg_hrf(subject, select_parcel)
            cont_hrf_by_parcel[select_parcel][subject] = (t_delay, hrf_eeg)
        except Exception as e:
            print(f"{subject}: continuous-EEG HRF load failed for {select_parcel} ({e}), skipping.")

#%% ---- visualize: for each parcel, one figure for the event-based model (one
# subplot per trial type) and a separate figure for the continuous-EEG model;
# each subject plotted as a thin line, group mean +/- 95% CI overlaid ----
for select_parcel in select_parcels:
    # event-based figure: one subplot per trial type
    subj_hrfs = event_hrf_by_parcel[select_parcel]
    if len(subj_hrfs) > 0:
        trial_types = sorted(set().union(*[set(h.trial_type.values) for h in subj_hrfs.values()]))
        fig_event, axs_event = plt.subplots(len(trial_types), 1, figsize=(7, 5 * len(trial_types)), sharey=True)
        axs_event = np.atleast_1d(axs_event)
        time_vals = next(iter(subj_hrfs.values())).time.values

        for ax, trial_type in zip(axs_event, trial_types):
            subj_curves = [h.sel(trial_type=trial_type).values for h in subj_hrfs.values() if trial_type in h.trial_type.values]
            for curve in subj_curves:
                ax.plot(time_vals, curve, color='b', alpha=0.25, linewidth=1)
            subj_curves = np.stack(subj_curves)
            n_subj = subj_curves.shape[0]
            mean_hrf = subj_curves.mean(axis=0)
            sem_hrf = stats.sem(subj_curves, axis=0)
            ci95 = sem_hrf * stats.t.ppf(0.975, max(n_subj - 1, 1))
            ax.plot(time_vals, mean_hrf, color='b', linewidth=2.5, label=f'mean (n={n_subj})')
            ax.fill_between(time_vals, mean_hrf - ci95, mean_hrf + ci95, color='b', alpha=0.2)
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_xlabel('Time from event onset (s)')
            ax.set_ylabel('HbO concentration')
            ax.set_title(trial_type)
            ax.legend()
            ax.grid()

        fig_event.suptitle(f'{select_parcel} — event-based HRF across subjects')
        plt.tight_layout()
        plt.show()

    # continuous-EEG figure: single subplot, one curve per subject, plus group mean +/- 95% CI
    subj_cont = cont_hrf_by_parcel[select_parcel]
    if len(subj_cont) > 0:
        fig_cont, ax_cont = plt.subplots(1, 1, figsize=(7, 5))
        t_delay = next(iter(subj_cont.values()))[0]
        subj_curves = np.stack([v[1] for v in subj_cont.values()])
        n_subj = subj_curves.shape[0]
        for curve in subj_curves:
            ax_cont.plot(t_delay, curve, color='r', alpha=0.25, linewidth=1)
        mean_hrf = subj_curves.mean(axis=0)
        sem_hrf = stats.sem(subj_curves, axis=0)
        ci95 = sem_hrf * stats.t.ppf(0.975, max(n_subj - 1, 1))
        ax_cont.plot(t_delay, mean_hrf, color='r', linewidth=2.5, label=f'mean (n={n_subj})')
        ax_cont.fill_between(t_delay, mean_hrf - ci95, mean_hrf + ci95, color='r', alpha=0.2)
        ax_cont.axhline(0, color='gray', linewidth=0.8)
        ax_cont.set_xlabel('Delay (s)')
        ax_cont.set_ylabel('Beta (HbO per unit EEG power)')
        ax_cont.set_title(f'{select_parcel} — continuous-EEG delay-response across subjects')
        ax_cont.legend()
        ax_cont.grid()
        plt.tight_layout()
        plt.show()

#%% ---- visualize where each select_parcel is located on the brain surface ----
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices

parcel_plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'parcel_location')
os.makedirs(parcel_plot_dir, exist_ok=True)

for select_parcel in select_parcels:
    highlight_vals = np.where(vertex_parcel == select_parcel, 1.0, 0.0)
    X_highlight = xr.DataArray(
        np.stack([highlight_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )

    parcel_plot_path = os.path.join(parcel_plot_dir, f'{select_parcel}_location')
    image_recon_multi_view(
        X_ts=X_highlight, head=head, cmap='Reds', clim=(0, 1),
        view_type='hbo_brain',
        title_str=select_parcel,
        SAVE=True, filename=parcel_plot_path,
        wdw_size=(1600, 800),
    )
    print(f'Saved parcel location plot to {parcel_plot_path}.png')

# %%
