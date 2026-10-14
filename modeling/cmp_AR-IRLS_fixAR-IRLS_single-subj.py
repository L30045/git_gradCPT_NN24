#%% load library
# Compare the mnt-correct HRF of Cedalion AR-IRLS and fixAR-IRLS on one subject, trained on all runs.
# Both models share the same Y and the same event-based design matrix (consecutive Gaussian kernels
# locked to the mnt-correct onsets, same as run_model_EEG_inform_parcel_based.py).
# Per run: highpass -> truncate to the trial window -> OLS drift -> OLS GSR. The runs are then
# concatenated and fit by
#   (1) Cedalion AR-IRLS: cedalion.models.glm.fit(noise_model='ar_irls') on the concatenated runs.
#   (2) fixAR-IRLS: one spectral-whitening kernel estimated on the concatenated residuals and applied to
#       the concatenated Y and X as a single series, then IRLS per parcel (no further AR whitening).
# The HRFs are plotted at the most sensitive parcel of select_network and over the whole network.
import numpy as np
import pickle
import gzip
import sys
import os
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
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon, image_recon_multi_view
from matplotlib.colors import LinearSegmentedColormap
from cedalion.vis.blocks import plot_labeled_points
import cedalion.models.glm as glm
from cedalion.models.glm.design_matrix import DesignMatrix
from scipy.signal import butter, sosfiltfilt

#%% parameters
subj_id = 670
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3'] # all runs are used for training
select_chromo = 'HbO'
select_network = 'ContA' # network whose most sensitive parcel and network HRF are plotted
l_cutoff = 0.02 # highpass cutoff for fNIRS (Hz)
len_delay = 15 # trial window ends len_delay after the last event onset (sec)
ar_order = 30 # max AR order of Cedalion AR-IRLS
is_scale_Y = True # If True, scale whitened Y by its std before the IRLS fit (betas are scaled back)
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685) # same robust norm as model.my_ar_irls_GLM
max_jobs = -1 # parallel jobs across parcels
is_save = True
is_plot = False # If True, show plots (they are always saved)
cfg_GLM['do_GSR'] = True
# event-based design: consecutive Gaussian kernels for mnt-correct trials (same as run_model_EEG_inform_parcel_based.py)
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 10*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
model_colors = {'AR-IRLS': 'C0', 'fixAR-IRLS': 'C3'}

subject = f"sub-{subj_id}"
model_name = 'mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS'
data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject, model_name)
os.makedirs(plot_dir, exist_ok=True)

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
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
# total forward-model sensitivity per parcel (summed over channel/wavelength/vertex), as in
# check_yhat_AR-IRLS_image_space.py
parcel_sensitivity = Adot_brain_sens.groupby('parcel').sum('vertex').sum(['channel', 'wavelength']).to_series()
sensitive_parcels = parcel_sensitivity.index.values

#%% load fNIRS parcel time series
der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
with gzip.open(os.path.join(der_dir, subject, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    all_stims = pickle.load(f)['stims']
with open(os.path.join(der_dir, subject, f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'), 'rb') as f:
    all_runs = pickle.load(f)['parcel_ts']

all_runs = [run.sel(parcel=run.parcel.isin(sensitive_parcels) & (run.parcel != 'scalp'),
                    chromo=[select_chromo]) for run in all_runs]
for run in all_runs:
    run.time.attrs['units'] = units.s
fnirs_sfreq = 1 / np.diff(all_runs[0].time.values).mean()

#%% match each run key (gradcpt1/2/3) to its fNIRS run via the first event onset
run_info = dict() # run_key -> (run_idx, event onsets in fNIRS time)
for run_key in run_keys:
    run_num = f"{run_key[-1]:0>2}"
    nirs_ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                          f"{subject}_task-gradCPT_run-{run_num}_events.tsv"), sep='\t')
    nirs_onsets = nirs_ev_df['onset'].values
    run_idx = next(r_i for r_i, stim in enumerate(all_stims)
                   if len(stim) > 0 and np.isclose(stim['onset'].values[0], nirs_onsets[0], atol=0.01))
    run_info[run_key] = (run_idx, nirs_onsets)

# keep parcels that are finite in every run used here
parcels = all_runs[0].parcel.values
for run_key in run_keys:
    run = all_runs[run_info[run_key][0]].sel(parcel=parcels).pint.dequantify()
    parcels = parcels[np.isfinite(run.values).all(axis=tuple(i for i, d in enumerate(run.dims) if d != 'parcel'))]
print(f"{subject}: {len(parcels)} parcels")

# network parcels and the most sensitive one among them
network_parcels = np.array([p for p in parcels if p.split('_')[0] == select_network])
if len(network_parcels) == 0:
    raise ValueError(f"{subject}: no fitted parcel in network {select_network}.")
select_parcel = parcel_sensitivity[network_parcels].idxmax()
print(f"{select_network}: {len(network_parcels)} parcels, most sensitive = {select_parcel}")

#%% consecutive Gaussian kernels (time from onset x component, at the fNIRS rate)
gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])
gauss_basis = gauss_kernels(all_runs[0]).transpose('time', 'component')
hrf_t = gauss_basis.time.values

#%% per-run preprocessing
def preprocess_run(run_key):
    """Highpass the full fNIRS run, build the mnt-correct Gaussian regressors on the full run (so
    trials at the window edges keep their whole kernels), truncate both to the trial window (first
    event onset -> last event onset + len_delay), then OLS-regress out drift and GSR from Y.
    Returns Y (parcel x chromo x time, molar), the DM (time x regressor x chromo) and the stim table."""
    run_idx, nirs_onsets = run_info[run_key]
    fnirs_run = all_runs[run_idx].sel(parcel=parcels)

    # highpass the full run
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

    # event-based regressors: consecutive Gaussian kernels at the mnt-correct onsets (fNIRS time)
    stim = all_stims[run_idx]
    stim = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
    stim['trial_type'] = 'mnt-correct'
    X_stim = glm.design_matrix.hrf_regressors(fnirs_run, stim, gauss_kernels).common.fillna(0)

    # truncate Y and the DM to the trial window, and reset time to start at 0
    t_f = fnirs_run.time.values
    keep = np.where((t_f >= nirs_onsets[0]) & (t_f <= nirs_onsets[-1] + len_delay))[0]
    fnirs_run = fnirs_run.isel(time=keep)
    X_stim = X_stim.isel(time=keep)
    t_rel = t_f[keep] - t_f[keep][0]
    fnirs_run = fnirs_run.assign_coords(time=t_rel)
    fnirs_run.time.attrs['units'] = units.s
    X_stim = X_stim.assign_coords(time=t_rel)
    stim['onset'] -= t_f[keep][0]

    # OLS-regress out the Legendre drift, then GSR (computed from the drift-residualized signal)
    drift_dm = model.get_drift_legendre_regressors([fnirs_run], cfg_GLM)[0]
    drift_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, drift_dm, noise_model='ols').sm.params, drift_dm)
    fnirs_run = (fnirs_run - drift_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)
    gsr_dm = model.get_global_mean_regressor([fnirs_run])[0]
    gsr_fit = glm.predict(fnirs_run, glm.fit(fnirs_run, gsr_dm, noise_model='ols').sm.params, gsr_dm)
    fnirs_run = (fnirs_run - gsr_fit.pint.dequantify().pint.quantify('molar')).transpose(*fnirs_run.dims)

    print(f"{run_key}: trial window {len(keep)} samples ({len(keep) / fnirs_sfreq:.0f} s), "
          f"{len(stim)} mnt-correct trials")
    return dict(Y=fnirs_run, X=X_stim, stim=stim)

run_data = {run_key: preprocess_run(run_key) for run_key in run_keys}

#%% concatenate runs (gradcpt1, 2, 3 order); the DM rows follow the concatenated time axis
Y_all, _, runs_updated = model.concatenate_runs([run_data[k]['Y'] for k in run_keys],
                                                [run_data[k]['stim'] for k in run_keys])
reg_names = run_data[run_keys[0]]['X'].regressor.values
if not all(np.array_equal(run_data[k]['X'].regressor.values, reg_names) for k in run_keys):
    raise ValueError("mnt-correct regressor names differ across runs.")
if gauss_basis.sizes['component'] != len(reg_names):
    raise ValueError(f"cfg_HRF_basis gives {gauss_basis.sizes['component']} kernels but the DM has {len(reg_names)}.")
X_all = xr.concat([run_data[k]['X'].assign_coords(time=run_upd.time.values)
                   for k, run_upd in zip(run_keys, runs_updated)], dim='time')
X_all = X_all.transpose('time', 'regressor', 'chromo')
dm_all = DesignMatrix(common=X_all)
run_len = [run_upd.sizes['time'] for run_upd in runs_updated]

#%% HRF from basis betas: (component x parcel) betas and (parcel x component x component) covariance
B = gauss_basis.values # (hrf time x component)

def to_hrf(C, cov):
    """HRF and its standard error (parcel x hrf time), in M."""
    hrf = xr.DataArray(C.T @ B.T, dims=('parcel', 'time'), coords={'parcel': parcels, 'time': hrf_t})
    se = np.sqrt(np.einsum('tk,pkl,tl->pt', B, cov, B))
    hrf_se = xr.DataArray(se, dims=('parcel', 'time'), coords={'parcel': parcels, 'time': hrf_t})
    return hrf, hrf_se

#%% (1) Cedalion AR-IRLS on the concatenated runs
print(f"Start Cedalion AR-IRLS fitting ({subject})")
glm_results = glm.fit(Y_all, dm_all, noise_model='ar_irls', ar_order=ar_order, max_jobs=max_jobs)
betas_ar = glm_results.sm.params.sel(chromo=select_chromo, parcel=parcels)
cov_ar = glm_results.sm.cov_params().sel(chromo=select_chromo, parcel=parcels)
C_ar = betas_ar.sel(regressor=reg_names).transpose('regressor', 'parcel').values
cov_ar = cov_ar.sel(regressor_r=reg_names, regressor_c=reg_names).transpose('parcel', 'regressor_r', 'regressor_c').values
hrf_ar, hrf_se_ar = to_hrf(C_ar, cov_ar)

#%% (2) fixAR-IRLS: spectral whitening on the concatenated runs, then IRLS per parcel
Y = Y_all.sel(chromo=select_chromo, parcel=parcels).pint.dequantify().transpose('time', 'parcel').values
X = X_all.sel(chromo=select_chromo).transpose('time', 'regressor').values

# the residual spectrum comes from the OLS residuals of Y ~ 1 + X (one intercept per run, shared
# slopes), averaged across parcels. Runs are concatenated first and the kernel is estimated and
# applied on the concatenated series as one run.
run_icpt = np.repeat(np.eye(len(run_keys)), run_len, axis=0)
X_ols = np.column_stack([run_icpt, X])
resid = Y - X_ols @ np.linalg.lstsq(X_ols, Y, rcond=None)[0]
noise_model = model.NoiseModel(method='spectral').fit(resid)
Y_white = noise_model.whiten(Y)
X_white = noise_model.whiten(X)
Y_scale = np.nanstd(Y_white) if is_scale_Y else 1.0
print(f"spectral whitening on {len(Y)} concatenated samples, w_pad = {noise_model.w_pad}, Y_scale = {Y_scale:.4g}")

# the whitened series are zero-mean (DC bin of the kernel is 0), so no intercept is added
print(f"Start fixAR-IRLS fitting ({subject})")
x_df = pd.DataFrame(X_white, columns=reg_names)
rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
    delayed(model.irls_fit)(pd.Series(Y_white[:, p_i] / Y_scale), x_df, M=irls_norm)
    for p_i in tqdm(range(len(parcels)))
)
C_fix = np.stack([res.params.values for res in rlm_results], axis=1) * Y_scale # (regressor x parcel), M
cov_fix = np.stack([res.cov_params().values for res in rlm_results]) * Y_scale ** 2
hrf_fix, hrf_se_fix = to_hrf(C_fix, cov_fix)

hrfs = {'AR-IRLS': (hrf_ar, hrf_se_ar), 'fixAR-IRLS': (hrf_fix, hrf_se_fix)}

#%% agreement of the two HRFs: correlation over HRF time, per parcel
hrf_corr_all = pd.Series({p: np.corrcoef(hrf_ar.sel(parcel=p), hrf_fix.sel(parcel=p))[0, 1] for p in parcels})
hrf_corr = hrf_corr_all[network_parcels]
print(f"all parcels: HRF correlation AR-IRLS vs fixAR-IRLS, median {hrf_corr_all.median():.3f}")
print(f"{select_network}: HRF correlation AR-IRLS vs fixAR-IRLS per parcel, "
      f"median {hrf_corr.median():.3f} (range {hrf_corr.min():.3f} to {hrf_corr.max():.3f}); "
      f"{select_parcel}: {hrf_corr[select_parcel]:.3f}")

#%% superior view of the selected parcel (red) on the brain, with the Cz landmark marked
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices
surf_png = os.path.join(plot_dir, f'surf_parcel_{select_parcel}.png')
vertex_vals = np.where(vertex_parcel == select_parcel, 1.0, np.nan) # NaN -> gray
X_surf = xr.DataArray(
    np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)
p0, _, _ = image_recon(X_surf, head, cmap='Reds', clim=(0, 1), view_type='hbo_brain',
                       view_position='superior', off_screen=True, wdw_size=(800, 800))
plot_labeled_points(p0, head.landmarks.sel(label=['Cz']))
p0.screenshot(surf_png)
p0.close()

#%% HRF correlation between the two models on the brain surface (all fitted parcels; gray = not fitted)
corr_cmap = LinearSegmentedColormap.from_list('blue_white_red', ['blue', 'white', 'red'])
vertex_corr = hrf_corr_all.reindex(vertex_parcel).values # NaN for parcels not fitted -> gray
X_corr = xr.DataArray(
    np.stack([vertex_corr, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)
corr_surf_png = os.path.join(plot_dir, f'{subject}_hrf_corr_surf')
image_recon_multi_view(
    X_ts=X_corr, head=head, cmap=corr_cmap, clim=(-1, 1), view_type='hbo_brain',
    title_str=f'HRF r, median {hrf_corr_all.median():.2f}',
    SAVE=True, filename=corr_surf_png, wdw_size=(1600, 800),
)

# load the saved surface PNG into a matplotlib figure so it shows in the interactive window
# parcel sensitivity (log10 of the total forward-model sensitivity used to pick select_parcel), fitted parcels only
log_sens = np.log10(parcel_sensitivity[parcels])
X_sens = X_corr.copy(data=np.stack([log_sens.reindex(vertex_parcel).values, np.zeros(n_vertex)], axis=-1))
sens_surf_png = os.path.join(plot_dir, f'{subject}_parcel_sensitivity_surf')
image_recon_multi_view(
    X_ts=X_sens, head=head, cmap=corr_cmap, clim=tuple(np.percentile(log_sens, [5, 95])), view_type='hbo_brain',
    title_str='log10 parcel sensitivity', SAVE=True, filename=sens_surf_png, wdw_size=(1600, 800),
)

fig, axs = plt.subplots(2, 1, figsize=(12, 12))
for ax, png, title in zip(axs, [corr_surf_png, sens_surf_png],
                          ['HRF correlation, AR-IRLS vs fixAR-IRLS', 'Parcel sensitivity (forward model)']):
    ax.imshow(plt.imread(png + '.png'))
    ax.axis('off')
    ax.set_title(f'{subject}: {title} (gray = not fitted)')
fig.tight_layout()
if is_plot:
    plt.show()
plt.close(fig)

#%% plot: HRF of both models at the most sensitive parcel of the network (± 1 SE)
to_uM = 1e6
fig, (ax, surf_ax) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.4, 1]})
for name, (hrf, hrf_se) in hrfs.items():
    h = hrf.sel(parcel=select_parcel).values * to_uM
    se = hrf_se.sel(parcel=select_parcel).values * to_uM
    ax.fill_between(hrf_t, h - se, h + se, color=model_colors[name], alpha=0.2, lw=0)
    ax.plot(hrf_t, h, color=model_colors[name], label=name)
ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Time from mnt-correct onset (s)')
ax.set_ylabel(f'{select_chromo} (µM)')
ax.set_title(f'{select_parcel}\n(most sensitive {select_network} parcel; r = {hrf_corr[select_parcel]:.2f})',
             fontsize='medium')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
surf_ax.imshow(plt.imread(surf_png))
surf_ax.set_title('Parcel location (superior view, Cz marked)', fontsize='small')
surf_ax.axis('off')
fig.suptitle(f'{subject}: mnt-correct HRF, mean ± 1 SE, trained on all runs')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, f'{subject}_hrf_{select_parcel}.png'), dpi=150)
if is_plot:
    plt.show()
plt.close(fig)

#%% plot: network HRF of each model (thin: parcels, bold: network mean), and the two network means
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
net_mean = dict()
for ax, (name, (hrf, _)) in zip(axs, hrfs.items()):
    h_net = hrf.sel(parcel=network_parcels).values * to_uM
    ax.plot(hrf_t, h_net.T, color=model_colors[name], lw=0.6, alpha=0.3)
    ax.plot(hrf_t, hrf.sel(parcel=select_parcel).values * to_uM, color='k', lw=1, ls=':',
            label=f'{select_parcel} (most sensitive)')
    net_mean[name] = (h_net.mean(0), h_net.std(0, ddof=1) / np.sqrt(len(h_net)))
    ax.plot(hrf_t, net_mean[name][0], color=model_colors[name], lw=2.5, label=f'{select_network} mean')
    ax.set_title(f'{name}: {len(network_parcels)} {select_network} parcels')
for name, (m, sem) in net_mean.items():
    axs[2].fill_between(hrf_t, m - sem, m + sem, color=model_colors[name], alpha=0.2, lw=0)
    axs[2].plot(hrf_t, m, color=model_colors[name], lw=2, label=name)
axs[2].set_title(f'{select_network} mean ± SEM across parcels')
for ax in axs:
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Time from mnt-correct onset (s)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axs[0].set_ylabel(f'{select_chromo} (µM)')
fig.suptitle(f'{subject}: {select_network} network mnt-correct HRF, trained on all runs '
             f'(median per-parcel r between models = {hrf_corr.median():.2f})')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, f'{subject}_hrf_network_{select_network}.png'), dpi=150)
if is_plot:
    plt.show()
plt.close(fig)

#%% save
results_dict = {
    'subject': subject, 'train_runs': run_keys, 'parcels': parcels, 'regressors': reg_names,
    'select_network': select_network, 'network_parcels': network_parcels, 'select_parcel': select_parcel,
    'hrf_basis': B, 'hrf_t': hrf_t, 'hrf_corr': hrf_corr_all,
    'ar_irls': {'C': C_ar, 'cov': cov_ar, 'hrf': hrf_ar, 'hrf_se': hrf_se_ar, 'ar_order': ar_order},
    'fixar_irls': {'C': C_fix, 'cov': cov_fix, 'hrf': hrf_fix, 'hrf_se': hrf_se_fix, 'Y_scale': Y_scale,
                   'whiten_kernel_acf': noise_model.acf_kernel[:int(np.round(np.sqrt(len(Y))))]},
}
if is_save:
    save_path = os.path.join(data_save_path, f'{subject}_{model_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"saved {save_path}")
