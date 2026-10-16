#%% load library
# Compare the coworker's Cedalion pipeline (code/cedalion_v26/batch_glm/STEP2_runGLM_on_image.py, with
# processing_func from processing_modules_v26) with my pipeline (cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py) on one
# subject. Both pipelines read the SAME image-space parcel time series (my IR_ts file, flag from params_setting),
# so the difference comes only from preprocessing, design matrix, fitting and plotting.
#   (A) Cedalion pipeline (replicated with the coworker's own functions, imported read-only):
#       per run: 0.01 Hz highpass (cedalion freq_filter), drop the scalp parcel -> concatenate runs ->
#       design matrix = consecutive Gaussian kernels (-2 to 18 s) for mnt-correct, mnt-incorrect and
#       city-incorrect + one GSR regressor per run (mean over all non-scalp parcels) -> Cedalion AR-IRLS ->
#       baseline-corrected HRF (pre-onset mean removed), HRF variance corrected by the image-recon vertex MSE.
#   (B) My pipeline: per run: 0.02 Hz highpass -> truncate to the trial window -> OLS drift -> OLS GSR (mean
#       over the sensitive parcels), design matrix = consecutive Gaussian kernels (-2 to 18 s, 20 kernels) for
#       mnt-correct only, then AR-IRLS, fixAR-IRLS and AR-iRRR as in cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py.
#       For the plots, my HRFs are baseline-corrected the same way as the Cedalion pipeline.
# Plotting follows the coworker's network plots (code/cedalion_v26/FIGURE_PREP): parcels are averaged per
# network (optionally per hemisphere), with variance sum(var) / n_parcel^2, and optionally normalized by the
# max |HRF| of each curve. Nothing in code/cedalion_v26 or processing_modules_v26 is modified.
import numpy as np
import pickle
import gzip
import sys
import os
from functools import reduce
import operator
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *
import cedalion.io
import cedalion.models.glm as glm
from cedalion.models.glm.design_matrix import DesignMatrix
from cedalion.sigproc import frequency
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal
# coworker's processing functions (read-only; do not write __pycache__ into their folder)
sys.dont_write_bytecode = True
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf
sys.dont_write_bytecode = False

#%% parameters
subject = 'sub-723'
select_chromo = 'HbO'
is_overwrite = False # If False, load the saved result instead of refitting
is_save = True
is_plot = False # If True, show plots (they are always saved)
max_jobs = -1

# (A) Cedalion pipeline: settings of STEP2_runGLM_on_image.py with NOISE_MODEL = 'ar_irls', USE_GSR = True
cfg_GLM_cw = {
    'do_drift': False,
    'do_drift_legendre': False,
    'do_short_sep': False,
    'drift_order': 3,
    'do_GSR': True,
    'GSR_weight': None,
    'noise_model': 'ar_irls',
    'HRF_basis': 'cons_gaussians',
    't_delta': 1*units.s,
    't_std': 1*units.s,
    't_pre': 2*units.s,
    't_post': 18*units.s,
}
F_MIN_cw = 0.01 # highpass (Hz), cedalion freq_filter with fmax = 0
F_MAX_cw = 0
cw_trial_type = 'mnt-correct' # trial type compared (the design also has mnt-incorrect and city-incorrect)

# (B) my pipeline: settings of cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py, with the 18 s kernel window
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']
l_cutoff = 0.02 # highpass cutoff for fNIRS (Hz)
len_delay = 15 # trial window ends len_delay after the last event onset (sec)
ar_order = 30
is_scale_Y = True
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685)
target_rank = 2
cand_irrr_lam1 = 10 ** np.arange(-3, 2.01, 0.125)
n_bisect = 20
cfg_GLM['do_GSR'] = True
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
is_baseline_correct_mine = True # subtract the pre-onset (t < 0) mean from my HRFs for the plots

# plotting
networks = { # parcel prefix -> panel title (same 12 networks and order as the coworker's figures)
    'VisCent': 'Visual Central', 'SomMotA': 'Somatomotor A', 'SomMotB': 'Somatomotor B',
    'DorsAttnA': 'Dorsal Attention A', 'DorsAttnB': 'Dorsal Attention B',
    'SalVentAttnA': 'Ventral Attention A', 'SalVentAttnB': 'Ventral Attention B',
    'ContA': 'Control A', 'ContB': 'Control B',
    'DefaultA': 'Default Mode A', 'DefaultB': 'Default Mode B', 'TempPar': 'Temporal Parietal',
}
hemis = ['both', 'LH', 'RH'] # 'both' = networks over both hemispheres; LH/RH as in the coworker's SPLIT_HEMI
pipe_names = ['Cedalion pipe', 'My pipe: AR-IRLS', 'My pipe: fixAR-IRLS', 'My pipe: AR-iRRR']
pipe_colors = {'Cedalion pipe': 'k', 'My pipe: AR-IRLS': 'C0', 'My pipe: fixAR-IRLS': 'C3', 'My pipe: AR-iRRR': 'C2'}
to_uM = 1e6

model_name = 'cmp_Cedalion-pipe_My-pipe'
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, model_name)
save_path = os.path.join(data_save_path, f'{subject}_{model_name}.pkl')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_save_path, exist_ok=True)

#%% sensitive parcels (same rule in both pipelines' plotting: any vertex with log10 sensitivity > -2)
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
sensitive_parcels = Adot_brain_sens.groupby('parcel').sum('vertex').parcel.values

#%% load the image-space data (shared by both pipelines)
preproc_path = os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl')
ir_path = os.path.join(der_dir, subject, f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl')
print(f"image-space input: {ir_path}")

def load_inputs():
    with gzip.open(preproc_path, 'rb') as f:
        all_stims = pickle.load(f)['stims']
    with open(ir_path, 'rb') as f:
        image_results = pickle.load(f)
    return all_stims, image_results['parcel_ts'], image_results['vertex_mse']

#%% (A) Cedalion pipeline
def run_cedalion_pipe():
    """STEP2_runGLM_on_image.py (SPLIT_VTC = False, USE_GSR = True, ar_irls) on my IR_ts file.
    Only select_chromo is fitted: AR-IRLS and the GSR regressor are per chromo, so this does not change
    the select_chromo result. Returns the mnt-correct HRF and its corrected variance (parcel x time, M / M^2)."""
    all_stims, all_runs, vv = load_inputs()

    # image-recon variance per parcel, used to correct the HRF variance
    n_runs = len(vv)
    vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
    vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

    # trial types and highpass, per run
    stims_pruned_list = []
    all_runs_tmp = []
    for stim, run in zip(all_stims, all_runs):
        mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
        mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
        mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'
        city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)].copy()
        city_trials['trial_type'] = 'city-incorrect'
        stims_pruned_list.append(pd.concat([mnt_trials, city_trials], ignore_index=True))

        run = run.sel(chromo=[select_chromo])
        run.time.attrs['units'] = units.s
        all_runs_tmp.append(frequency.freq_filter(run, F_MIN_cw*units.Hz, F_MAX_cw*units.Hz))

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs_tmp]
    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        all_runs_tmp.append(run.sel(parcel=run.parcel != 'scalp'))
    all_runs = all_runs_tmp

    print(f"Start Cedalion pipeline fit ({subject}): {all_runs[0].sizes['parcel']} parcels")
    results, hrf_estimate, hrf_mse, dms = pf.GLM(all_runs, cfg_GLM_cw, None, None, stims_pruned_list)
    print(f"design matrix regressors: {list(dms.common.regressor.values)}")

    # corrected HRF variance: scale the beta covariance by (residual var + image-recon var) / residual var
    Y_all, _, _ = pf.concatenate_runs(all_runs, stims_pruned_list)
    betas = results.sm.params
    cov_params = results.sm.cov_params()
    Y_pred = xr.dot(dms.common, betas, dim='regressor')
    resid = Y_all.pint.dequantify() - Y_pred
    var_resid = resid.var('time')
    weight = (var_resid + vp.sel(chromo=[select_chromo]).pint.dequantify()) / var_resid
    basis_hrf = glm.GaussianKernels(cfg_GLM_cw['t_pre'], cfg_GLM_cw['t_post'],
                                    cfg_GLM_cw['t_delta'], cfg_GLM_cw['t_std'])(Y_all)
    cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {cw_trial_type}"),
                             regressor_c=cov_params.regressor_c.str.startswith(f"HRF {cw_trial_type}"))
    hrf_mse_corr = pf.estimate_HRF_cov(weight * cov_hrf, basis_hrf).assign_coords({'time': hrf_estimate.time})

    hrf = hrf_estimate.sel(trial_type=cw_trial_type, chromo=select_chromo).pint.dequantify()
    return {'hrf': hrf.transpose('parcel', 'time'),
            'hrf_var': hrf_mse_corr.sel(chromo=select_chromo).transpose('parcel', 'time'),
            'hrf_var_uncorr': hrf_mse.sel(trial_type=cw_trial_type, chromo=select_chromo).pint.dequantify().transpose('parcel', 'time'),
            'regressors': dms.common.regressor.values,
            'n_trials': pd.concat(stims_pruned_list)['trial_type'].value_counts().to_dict()}

#%% (B) my pipeline
def run_my_pipe():
    """cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py run_subject with the 18 s kernel window (no plots).
    Returns HRFs (parcel x time, M) and SEs of AR-IRLS, fixAR-IRLS and AR-iRRR (not baseline-corrected)."""
    all_stims, all_runs, _ = load_inputs()
    all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels) & (run.parcel != 'scalp'),
                        chromo=[select_chromo]) for run in all_runs]
    for run in all_runs:
        run.time.attrs['units'] = units.s
    fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

    # match each run key (gradcpt1/2/3) to its fNIRS run via the first event onset
    run_info = dict()
    for run_key in run_keys:
        run_num = f"{run_key[-1]:0>2}"
        nirs_ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                              f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
        nirs_onsets = nirs_ev_df['onset'].values
        run_idx = next(r_i for r_i, stim in enumerate(all_stims)
                       if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onsets[0], atol=0.01))
        run_info[run_key] = (run_idx, nirs_onsets)

    # keep parcels that are finite in every run
    parcels = all_runs[0].parcel.values
    for run_key in run_keys:
        run = all_runs[run_info[run_key][0]].sel(parcel=parcels).pint.dequantify()
        parcels = parcels[np.isfinite(run.values).all(axis=tuple(i for i, d in enumerate(run.dims) if d != 'parcel'))]
    print(f"My pipeline ({subject}): {len(parcels)} parcels")

    gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                        cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])
    gauss_basis = gauss_kernels(all_runs[0]).transpose('time', 'component')
    hrf_t = gauss_basis.time.values

    def preprocess_run(run_key):
        """Highpass -> mnt-correct regressors on the full run -> truncate -> OLS drift -> OLS GSR."""
        run_idx, nirs_onsets = run_info[run_key]
        fnirs_run = all_runs[run_idx].sel(parcel=parcels)

        fnirs_units = fnirs_run.pint.units
        sos = butter(4, l_cutoff, btype='highpass', fs=fnirs_sfreq, output='sos')
        fnirs_run = xr.apply_ufunc(
            sosfiltfilt, sos, fnirs_run.pint.dequantify(),
            input_core_dims=[[], ['time']],
            output_core_dims=[['time']],
            exclude_dims={'time'},
        ).transpose(*fnirs_run.dims).pint.quantify(fnirs_units)
        fnirs_run = fnirs_run.assign_coords({'time': all_runs[run_idx].time})
        fnirs_run.time.attrs['units'] = units.s

        stim = all_stims[run_idx]
        stim = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
        stim['trial_type'] = 'mnt-correct'
        X_stim = glm.design_matrix.hrf_regressors(fnirs_run, stim, gauss_kernels).common.fillna(0)

        t_f = fnirs_run.time.values
        keep = np.where((t_f >= nirs_onsets[0]) & (t_f <= nirs_onsets[-1] + len_delay))[0]
        fnirs_run = fnirs_run.isel(time=keep)
        X_stim = X_stim.isel(time=keep)
        t_rel = t_f[keep] - t_f[keep][0]
        fnirs_run = fnirs_run.assign_coords(time=t_rel)
        fnirs_run.time.attrs['units'] = units.s
        X_stim = X_stim.assign_coords(time=t_rel)
        stim['onset'] -= t_f[keep][0]

        drift_dm = model.get_drift_legendre_regressors([fnirs_run], cfg_GLM)[0]
        drift_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, drift_dm, noise_model='ols').sm.params, drift_dm)
        fnirs_run = (fnirs_run - drift_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)
        gsr_dm = model.get_global_mean_regressor([fnirs_run])[0]
        gsr_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, gsr_dm, noise_model='ols').sm.params, gsr_dm)
        fnirs_run = (fnirs_run - gsr_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)
        return dict(Y=fnirs_run, X=X_stim, stim=stim)

    run_data = {run_key: preprocess_run(run_key) for run_key in run_keys}
    Y_all, _, runs_updated = model.concatenate_runs([run_data[k]['Y'] for k in run_keys],
                                                    [run_data[k]['stim'] for k in run_keys])
    reg_names = run_data[run_keys[0]]['X'].regressor.values
    X_all = xr.concat([run_data[k]['X'].assign_coords(time=run_upd.time.values)
                       for k, run_upd in zip(run_keys, runs_updated)], dim='time')
    X_all = X_all.transpose('time', 'regressor', 'chromo')
    dm_all = DesignMatrix(common=X_all)
    run_len = [run_upd.sizes['time'] for run_upd in runs_updated]
    B = gauss_basis.values

    def to_hrf(C, cov=None):
        hrf = xr.DataArray(C.T @ B.T, dims=('parcel', 'time'), coords={'parcel': parcels, 'time': hrf_t})
        if cov is None:
            return hrf, None
        se = np.sqrt(np.einsum('tk,pkl,tl->pt', B, cov, B))
        return hrf, xr.DataArray(se, dims=('parcel', 'time'), coords={'parcel': parcels, 'time': hrf_t})

    # AR-IRLS
    print(f"Start my AR-IRLS fitting ({subject})")
    glm_results = glm.fit(Y_all, dm_all, noise_model='ar_irls', ar_order=ar_order, max_jobs=max_jobs)
    C_ar = glm_results.sm.params.sel(chromo=select_chromo, parcel=parcels, regressor=reg_names).transpose('regressor', 'parcel').values
    cov_ar = glm_results.sm.cov_params().sel(chromo=select_chromo, parcel=parcels, regressor_r=reg_names,
                                            regressor_c=reg_names).transpose('parcel', 'regressor_r', 'regressor_c').values
    hrf_ar, hrf_se_ar = to_hrf(C_ar, cov_ar)

    # fixAR-IRLS: one spectral whitening kernel on the concatenated runs, then IRLS per parcel
    Y = Y_all.sel(chromo=select_chromo, parcel=parcels).pint.dequantify().transpose('time', 'parcel').values
    X = X_all.sel(chromo=select_chromo).transpose('time', 'regressor').values
    run_icpt = np.repeat(np.eye(len(run_keys)), run_len, axis=0)
    X_ols = np.column_stack([run_icpt, X])
    resid = Y - X_ols @ np.linalg.lstsq(X_ols, Y, rcond=None)[0]
    noise_model = model.NoiseModel(method='spectral').fit(resid)
    Y_white = noise_model.whiten(Y)
    X_white = noise_model.whiten(X)
    Y_scale = np.nanstd(Y_white) if is_scale_Y else 1.0
    Y_white_s = Y_white / Y_scale

    print(f"Start my fixAR-IRLS fitting ({subject})")
    x_df = pd.DataFrame(X_white, columns=reg_names)
    rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
        delayed(model.irls_fit)(pd.Series(Y_white_s[:, p_i]), x_df, M=irls_norm)
        for p_i in tqdm(range(len(parcels)))
    )
    C_fix = np.stack([res.params.values for res in rlm_results], axis=1) * Y_scale
    cov_fix = np.stack([res.cov_params().values for res in rlm_results]) * Y_scale ** 2
    hrf_fix, hrf_se_fix = to_hrf(C_fix, cov_fix)

    # AR-iRRR: smallest lam1 whose fit has rank target_rank
    print(f"Start my AR-iRRR fitting ({subject})")
    n_time, n_parcel = Y_white_s.shape
    X_c = X_white - X_white.mean(0, keepdims=True)
    irrr_weight = [np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / n_time]

    def fit_irrr(lam1):
        C, mu, _, _, _ = irrr_normal(Y_white_s, [X_white], lam1,
                                     {'varyrho': True, 'Tol': 0.01, 'fig': False, 'weight': irrr_weight})
        return {'lam1': lam1, 'C': C, 'rank': matrix_rank(C)}

    irrr_scan = [fit_irrr(lam1) for lam1 in tqdm(cand_irrr_lam1, desc='iRRR lam1 scan')]
    rank_hits = [f for f in irrr_scan if f['rank'] == target_rank]
    if not rank_hits:
        lo = max(f['lam1'] for f in irrr_scan if f['rank'] > target_rank)
        hi = min(f['lam1'] for f in irrr_scan if f['rank'] < target_rank and f['lam1'] > lo)
        for _ in range(n_bisect):
            fit = fit_irrr(np.sqrt(lo * hi))
            if fit['rank'] == target_rank:
                rank_hits = [fit]
                break
            lo, hi = (fit['lam1'], hi) if fit['rank'] > target_rank else (lo, fit['lam1'])
        else:
            raise RuntimeError(f"no irrr_lam1 in [{lo:.4g}, {hi:.4g}] gives rank {target_rank}")
    irrr_fit = min(rank_hits, key=lambda f: f['lam1'])
    print(f"irrr_lam1 = {irrr_fit['lam1']:.4g} (rank {irrr_fit['rank']})")
    hrf_irrr, _ = to_hrf(irrr_fit['C'] * Y_scale)

    return {'My pipe: AR-IRLS': {'hrf': hrf_ar, 'hrf_var': hrf_se_ar ** 2},
            'My pipe: fixAR-IRLS': {'hrf': hrf_fix, 'hrf_var': hrf_se_fix ** 2},
            'My pipe: AR-iRRR': {'hrf': hrf_irrr, 'hrf_var': None, 'irrr_lam1': irrr_fit['lam1']},
            'regressors': reg_names}

#%% run both pipelines (load the saved result unless is_overwrite)
if not is_overwrite and os.path.exists(save_path):
    print(f"loading {save_path}")
    with open(save_path, 'rb') as f:
        res = pickle.load(f)
else:
    res = {'Cedalion pipe': run_cedalion_pipe(), **run_my_pipe()}
    if is_save:
        with open(save_path, 'wb') as f:
            pickle.dump(res, f)
        print(f"saved {save_path}")
print(f"Cedalion pipe trials: {res['Cedalion pipe']['n_trials']}")

#%% HRFs for plotting: baseline-correct my HRFs, put everything on the Cedalion pipeline's time axis
def baseline_correct(hrf):
    """Subtract the pre-onset (t < 0) mean, as pf.estimate_HRF_from_beta does."""
    return hrf - hrf.sel(time=hrf.time[hrf.time < 0]).mean('time')

hrf_t = res['Cedalion pipe']['hrf'].time.values
plot_hrfs = dict() # pipeline -> (hrf, var), parcel x time in M / M^2, on hrf_t
for name in pipe_names:
    hrf, var = res[name]['hrf'], res[name]['hrf_var']
    if name != 'Cedalion pipe':
        if is_baseline_correct_mine:
            hrf = baseline_correct(hrf)
        # same number of samples and spacing; the two pipelines only label the time axis differently
        hrf = hrf.assign_coords(time=hrf_t)
        if var is not None:
            var = var.assign_coords(time=hrf_t)
    plot_hrfs[name] = (hrf, var)

# parcels used in the network averages: the sensitive parcels each pipeline fitted with finite values
for name, (hrf, _) in plot_hrfs.items():
    ok = hrf.parcel.isin(sensitive_parcels) & np.isfinite(hrf).all('time')
    plot_hrfs[name] = (hrf.sel(parcel=ok), None if plot_hrfs[name][1] is None else plot_hrfs[name][1].sel(parcel=ok))
    print(f"{name}: {int(ok.sum())} sensitive parcels in the network averages")

def network_hrf(hrf, var, net, hemi):
    """Network mean HRF and its SE (sum(var) / n^2, as in figprep.to_networks); SE is None without var."""
    sel = [p for p in hrf.parcel.values if p.split('_')[0] == net and (hemi == 'both' or p.split('_')[-1] == hemi)]
    m = hrf.sel(parcel=sel).mean('parcel').values
    se = None if var is None else np.sqrt(var.sel(parcel=sel).sum('parcel').values / len(sel) ** 2)
    return m, se, len(sel)

#%% network plots (3 x 4, coworker's network order), raw and normalized, both hemispheres and split
for hemi in hemis:
    for is_norm in [False, True]:
        fig, axs = plt.subplots(3, 4, figsize=(18, 10.5), sharex=True)
        for ax, (net, net_title) in zip(axs.flat, networks.items()):
            for name in pipe_names:
                m, se, n_par = network_hrf(*plot_hrfs[name], net, hemi)
                norm = np.abs(m).max() if is_norm else 1 / to_uM
                if se is not None:
                    ax.fill_between(hrf_t, (m - se) / norm, (m + se) / norm, color=pipe_colors[name], alpha=0.15, lw=0)
                ax.plot(hrf_t, m / norm, color=pipe_colors[name], lw=2.2 if name == 'Cedalion pipe' else 1.6,
                        label=name)
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.set_title(f'{net_title} ({n_par} parcels)', fontsize=11)
            ax.grid(alpha=0.3)
        axs.flat[0].legend(fontsize=8, frameon=False)
        fig.supxlabel('Time from mnt-correct onset (s)')
        fig.supylabel('Normalized amplitude (each curve / its max |HRF|)' if is_norm else f'{select_chromo} (µM)')
        hemi_str = 'both hemispheres' if hemi == 'both' else f'{hemi} only'
        bc_str = 'baseline-corrected' if is_baseline_correct_mine else 'no baseline correction'
        fig.suptitle(f'{subject}: network mnt-correct HRF ({hemi_str}), ± 1 SE; my pipeline {bc_str}')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(os.path.join(plot_dir, f'{subject}_network_hrf_{hemi}{"_norm" if is_norm else ""}.png'), dpi=150)
        if is_plot:
            plt.show()
        plt.close(fig)

#%% agreement between the Cedalion pipeline and each of my models
ref_hrf = plot_hrfs['Cedalion pipe'][0]
corr_parcel = dict()
for name in pipe_names[1:]:
    hrf = plot_hrfs[name][0]
    common = np.intersect1d(ref_hrf.parcel.values, hrf.parcel.values)
    corr_parcel[name] = pd.Series({p: np.corrcoef(ref_hrf.sel(parcel=p), hrf.sel(parcel=p))[0, 1] for p in common})
corr_parcel = pd.DataFrame(corr_parcel)
corr_net = pd.DataFrame({
    name: {f'{net}': np.corrcoef(network_hrf(*plot_hrfs['Cedalion pipe'], net, 'both')[0],
                                 network_hrf(*plot_hrfs[name], net, 'both')[0])[0, 1] for net in networks}
    for name in pipe_names[1:]})
amp_ratio = pd.DataFrame({
    name: {net: np.abs(network_hrf(*plot_hrfs[name], net, 'both')[0]).max()
                / np.abs(network_hrf(*plot_hrfs['Cedalion pipe'], net, 'both')[0]).max() for net in networks}
    for name in pipe_names[1:]})
print("per-parcel HRF r vs the Cedalion pipeline:")
print(corr_parcel.describe().loc[['count', 'mean', '50%', 'min', 'max']].round(3).to_string())
print("network-mean HRF r vs the Cedalion pipeline (both hemispheres):")
print(corr_net.round(2).to_string())
print("network-mean peak |HRF| ratio, my model / Cedalion pipeline:")
print(amp_ratio.round(2).to_string())

fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [1, 1.3]})
bins = np.linspace(-1, 1, 41)
for name in corr_parcel:
    axs[0].hist(corr_parcel[name].dropna(), bins=bins, histtype='step', lw=2, color=pipe_colors[name],
                label=f'{name} (median {corr_parcel[name].median():.2f})')
axs[0].set_xlabel('per-parcel HRF r vs Cedalion pipe')
axs[0].set_ylabel('# parcels')
axs[0].legend(fontsize=8, frameon=False, loc='upper left')
x = np.arange(len(networks))
for i, name in enumerate(corr_net):
    axs[1].bar(x + (i - 1) * 0.27, corr_net[name].values, width=0.27, color=pipe_colors[name], label=name)
axs[1].set_xticks(x, list(networks), rotation=45, ha='right', fontsize=8)
axs[1].axhline(0, color='k', lw=0.5)
axs[1].set_ylabel('network-mean HRF r vs Cedalion pipe')
axs[1].legend(fontsize=8, frameon=False)
for ax in axs:
    ax.grid(alpha=0.3)
fig.suptitle(f'{subject}: agreement with the Cedalion pipeline (my pipeline baseline-corrected: {is_baseline_correct_mine})')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, f'{subject}_hrf_corr_vs_cedalion_pipe.png'), dpi=150)
if is_plot:
    plt.show()
plt.close(fig)
