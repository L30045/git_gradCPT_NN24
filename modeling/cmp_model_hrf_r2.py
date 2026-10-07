#%% load library
# Compare how well the EEG-informed HRF models reconstruct Y_partial (fNIRS after the drift and
# GSR OLS stages), using betas from:
#   1. AR-IRLS (run_model_cont_EEG_fNIRS.py, 3-stage_bspline-test)
#   2. per-subject iRRR (run_model_cont_EEG_fNIRS_iRRR.py)
#   3. group iRRR (run_model_cont_EEG_fNIRS_iRRR_group.py)
# All three share the same Y_all / dm_all, so only the betas differ. RMSE and R2 are computed
# per subject per parcel (in-sample) and summarized with bar plots.
# The last section compares the event-based models (stimulus-locked HRF vs trial-masked cont EEG)
# on the time points inside the mnt trials only.
import numpy as np
import pickle
import gzip
import os
import matplotlib.pyplot as plt
import pandas as pd
from params_setting import *

#%% select model type
eeg_reg_type = 'cont_EEG_cz_3-stage'  # must match the iRRR scripts
ar_irls_reg_type = 'cont_EEG_cz_3-stage_bspline-test'  # must match run_model_cont_EEG_fNIRS.py
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 (Hz)
hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
select_chromo = 'HbO'
is_save = True # If True, save metrics table and figures
# If True, score each AR-IRLS model on Y_partial and Y_hat prewhitened with its own AR filter (the
# space AR-IRLS fits in); iRRR models stay unwhitened (the space they fit in).
# Outputs get a '_whitened' suffix so the all-unwhitened results are kept
is_whiten = False
ar_pmax = 30 # max AR order; glm.fit's default ar_order, used by all AR-IRLS fits here
out_tag = '_whitened' if is_whiten else ''
fit_space = 'AR-IRLS models whitened' if is_whiten else 'unwhitened'
ar_irls_models = ['AR-IRLS', 'Event-based', 'mnt AR-IRLS']  # models fit with AR-IRLS

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
plot_dir = os.path.join(eeg_der_dir, 'HRF_surf', 'group', f'{eeg_reg_type}_model_cmp')
os.makedirs(plot_dir, exist_ok=True)
models = ['AR-IRLS', 'iRRR', 'iRRR_group']
model_colors = {'AR-IRLS': '#2a78d6', 'iRRR': '#eb6834', 'iRRR_group': '#1baf7a'}

#%% load group iRRR fit
group_betas_file = os.path.join(eeg_der_dir, 'group', f'group_{eeg_reg_type}_iRRR_{hp_flag}_betas.pkl')
with open(group_betas_file, 'rb') as f:
    group_betas = pickle.load(f)['betas']
with open(group_betas_file.replace('_betas.pkl', '_stats.pkl'), 'rb') as f:
    group_stats = pickle.load(f)
group_mu = group_stats['intercept']  # (parcel, 1)

#%% compute RMSE / R2 per subject per parcel
def fit_metrics(Y, Y_hat):
    """RMSE and R2 per column (parcel) of time x parcel arrays, ignoring NaNs."""
    res = Y - Y_hat
    rmse = np.sqrt(np.nanmean(res**2, axis=0))
    ss_res = np.nansum(res**2, axis=0)
    ss_tot = np.nansum((Y - np.nanmean(Y, axis=0))**2, axis=0)
    return rmse, 1 - ss_res / ss_tot

import scipy.signal
import cedalion.math.ar_model

def get_ar_filters(resid, pmax=ar_pmax):
    """Per-parcel AR whitening filter [1, -a_1, ..., -a_p] for time x parcel residuals.
    AR-IRLS (cedalion.math.ar_irls) does not save its filter, so re-estimate it the same way:
    a BIC-selected AR model (order <= pmax) fit to the residuals of the final betas."""
    filters = []
    for r in resid.T:
        ar_model = cedalion.math.ar_model.bic_arfit(pd.Series(r[np.isfinite(r)]), pmax=pmax)
        filters.append(np.hstack([1, -ar_model.params[1:]]))
    return filters

def apply_ar_filters(arr, filters):
    """Filter each column of a time x parcel array with its AR filter over the finite samples.
    The first p samples (filter not yet initialized) are set to NaN, as AR-IRLS drops them."""
    out = np.full(arr.shape, np.nan)
    for j, wf in enumerate(filters):
        finite = np.where(np.isfinite(arr[:, j]))[0]
        out[finite, j] = scipy.signal.lfilter(wf, 1, arr[finite, j])
        out[finite[:len(wf) - 1], j] = np.nan
    return out

def to_fit_space(Y, Y_hat, model_name):
    """(Y, Y_hat) in the space the model was fit in: for AR-IRLS models (if is_whiten), both are
    filtered with the per-parcel AR filter re-estimated from that model's own residuals."""
    if not (is_whiten and model_name in ar_irls_models):
        return Y, Y_hat
    ar_filters = get_ar_filters(Y - Y_hat)
    return apply_ar_filters(Y, ar_filters), apply_ar_filters(Y_hat, ar_filters)

rows = []
subjects = []
y_unit = None
for subject in group_stats['subjects']:
    data_dir = os.path.join(eeg_der_dir, subject)
    ar_irls_prefix = os.path.join(data_dir, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    irrr_prefix = os.path.join(data_dir, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}')
    Y_all_path = get_Y_all_path(ar_irls_prefix)
    req_files = [Y_all_path, get_dm_all_path(ar_irls_prefix),
                 get_betas_path(ar_irls_prefix), get_betas_path(irrr_prefix), get_stats_path(irrr_prefix)]
    if not all(os.path.exists(f) for f in req_files):
        print(f"{subject}: missing AR-IRLS or iRRR results, skipping.")
        continue
    print(f"Processing {subject}")

    with gzip.open(Y_all_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(get_dm_all_path(ar_irls_prefix), 'rb') as f:
        dm_all = pickle.load(f)
    with open(get_betas_path(ar_irls_prefix), 'rb') as f:
        ar_betas = pickle.load(f)['betas']
    with open(get_betas_path(irrr_prefix), 'rb') as f:
        irrr_betas = pickle.load(f)['betas']
    with open(get_stats_path(irrr_prefix), 'rb') as f:
        irrr_mu = pickle.load(f)['intercept']  # (parcel, 1)

    # Y_all was built after the drift and GSR OLS stages, so it is already
    # Y_partial = Y_raw - Y_hat_drift - Y_hat_gsr
    Y_da = Y_all.sel(chromo=select_chromo)
    if y_unit is None:
        y_unit = f'{Y_da.pint.units:~P}' if Y_da.pint.units is not None else ''
    Y_da = Y_da.pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    parcels, regressors = Y_da.parcel.values, X_da.regressor.values
    Y_partial, X_np = Y_da.values, X_da.values

    def get_B(betas, regressors=regressors):
        return betas.sel(chromo=select_chromo, parcel=parcels, regressor=regressors) \
                    .transpose('regressor', 'parcel').values

    # AR-IRLS: no intercept in the design (Y_partial is already drift-residualized)
    # iRRR: intercept mu fit on the raw X
    # group iRRR: fit on per-subject demeaned X and Y, so add back the subject mean of Y
    # (the implicit subject-specific intercept of the group model)
    irrr_mu_s = pd.Series(irrr_mu.ravel(), index=irrr_betas.parcel.values)[parcels].values
    group_mu_s = pd.Series(group_mu.ravel(), index=group_betas.parcel.values)[parcels].values
    Y_hat_eeg = {
        'AR-IRLS': X_np @ get_B(ar_betas),
        'iRRR': X_np @ get_B(irrr_betas) + irrr_mu_s,
        'iRRR_group': (X_np - X_np.mean(0, keepdims=True)) @ get_B(group_betas) + group_mu_s
                      + np.nanmean(Y_partial, axis=0, keepdims=True),
    }
    for model_name, Y_hat in Y_hat_eeg.items():
        rmse, r2 = fit_metrics(*to_fit_space(Y_partial, Y_hat, model_name))
        rows.append(pd.DataFrame({'subject': subject, 'model': model_name, 'parcel': parcels,
                                  'rmse': rmse, 'r2': r2}))
    subjects.append(subject)

metrics_df = pd.concat(rows, ignore_index=True)
if y_unit == 'M':  # report RMSE in uM for readability
    metrics_df['rmse'] *= 1e6
    y_unit = 'µM'

#%% summarize: mean over parcels per subject, then mean +/- SEM across subjects
subj_df = metrics_df.groupby(['subject', 'model'])[['rmse', 'r2']].mean().reset_index()
summary_df = subj_df.groupby('model')[['rmse', 'r2']].agg(['mean', 'sem']).reindex(models)
print(subj_df.pivot(index='subject', columns='model', values='r2')[models].to_string(float_format='%.4f'))
print(summary_df.to_string(float_format='%.4g'))
if is_save:
    metrics_df.to_csv(os.path.join(plot_dir, f'model_cmp_rmse_r2_per_parcel{out_tag}.csv'), index=False)

#%% bar plots: per-subject mean over parcels, plus the cross-subject mean +/- SEM
metric_info = {'r2': 'R$^2$', 'rmse': f'RMSE ({y_unit})' if y_unit else 'RMSE'}
x_labels = subjects + [f'Mean\n(n={len(subjects)})']
x = np.arange(len(x_labels))
bar_w = 0.8 / len(models)
fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.1 * len(x_labels)), 7), sharex=True)
for ax, (metric, ylabel) in zip(axes, metric_info.items()):
    for m_i, model_name in enumerate(models):
        subj_vals = subj_df[subj_df.model == model_name].set_index('subject')[metric][subjects].values
        vals = np.append(subj_vals, summary_df.loc[model_name, (metric, 'mean')])
        err = np.append(np.full(len(subjects), np.nan), summary_df.loc[model_name, (metric, 'sem')])
        ax.bar(x + (m_i - (len(models) - 1) / 2) * bar_w, vals, bar_w, yerr=err,
               color=model_colors[model_name], edgecolor='white', linewidth=2,
               error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': '#555555'}, label=model_name)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(len(subjects) - 0.5, color='gray', lw=0.5, ls='--')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
axes[0].set_title(f'Mean over parcels of Y_hat_eeg vs Y_partial (in-sample, {fit_space})')
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=len(models), frameon=False)
axes[-1].set_xticks(x, x_labels, rotation=45, ha='right')
fig.suptitle(f'EEG HRF model comparison ({select_chromo}, {eeg_reg_type})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, f'model_cmp_rmse_r2_bar{out_tag}.png'), dpi=150)
plt.show()

#%% brain surface maps: per-parcel R2 / RMSE averaged across subjects, one view per model
import xarray as xr
import cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices
parcel_df = metrics_df.groupby(['model', 'parcel'])[['rmse', 'r2']].mean()

# R2 on a linear scale from 0 to the max over models. Parcels with R2 <= 0 are pinned to the
# lowest colormap bin, which is black (one bin below 0, so no positive parcel lands on it).
from matplotlib.colors import ListedColormap
r2_max = np.nanmax(parcel_df['r2'])
r2_black = -r2_max / 255
r2_cmap = ListedColormap(np.vstack([[0, 0, 0, 1], plt.get_cmap('YlOrRd')(np.linspace(0, 1, 255))]))

def r2_to_surf(r2):
    """R2 > 0 unchanged; R2 <= 0 -> black bin; NaN stays NaN (gray)."""
    return r2.where((r2 > 0) | r2.isna(), r2_black)

# RMSE: subtract each model's mean over parcels, so the map shows which parcels fit worse (+)
# or better (-) than that model's average; shared symmetric limits across models
rmse_dm = parcel_df['rmse'] - parcel_df['rmse'].groupby('model').transform('mean')
rmse_lim = np.nanpercentile(np.abs(rmse_dm), 98)
surf_cfg = {
    'r2': {'vals': r2_to_surf(parcel_df['r2']),
           'cmap': r2_cmap, 'clim': (r2_black, r2_max),
           'label': 'R$^2$ (black: R$^2$ ≤ 0)', 'bar_title': 'R2'},
    'rmse': {'vals': rmse_dm, 'cmap': 'seismic', 'clim': (-rmse_lim, rmse_lim),
             'label': f'RMSE − parcel mean ({y_unit})', 'bar_title': f'RMSE - mean ({y_unit})'},
}
surf_paths = {}
for metric, cfg in surf_cfg.items():
    for model_name in models:
        val_by_parcel = cfg['vals'].loc[model_name].to_dict()
        vertex_vals = np.array([val_by_parcel.get(p, np.nan) for p in vertex_parcel])
        X_surf = xr.DataArray(
            np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
            dims=['vertex', 'chromo'],
            coords={'chromo': ['HbO', 'HbR'],
                    'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
        )
        surf_path = os.path.join(plot_dir, f'surf_{metric}_{model_name}{out_tag}')
        image_recon_multi_view(
            X_ts=X_surf, head=head, cmap=cfg['cmap'], clim=cfg['clim'],
            view_type='hbo_brain', title_str=f'{model_name} {cfg["bar_title"]}',
            SAVE=True, filename=surf_path, wdw_size=(1600, 800),
        )
        surf_paths[(metric, model_name)] = surf_path + '.png'

fig, axes = plt.subplots(len(models), len(surf_cfg), figsize=(8 * len(surf_cfg), 4 * len(models)),
                         squeeze=False)
for c_i, metric in enumerate(surf_cfg):
    for r_i, model_name in enumerate(models):
        ax = axes[r_i, c_i]
        ax.imshow(plt.imread(surf_paths[(metric, model_name)]))
        ax.axis('off')
        ax.set_title(f'{model_name}: {surf_cfg[metric]["label"]}')
fig.suptitle(f'Per-parcel fit of Y_hat_eeg vs Y_partial, mean over {len(subjects)} subjects ({select_chromo}, {fit_space})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, f'model_cmp_rmse_r2_surf{out_tag}.png'), dpi=150)
plt.show()

#%% brain surface map: RMSE difference AR-IRLS - iRRR per parcel, mean over subjects
# positive (red) = iRRR fits Y_partial better than AR-IRLS in that parcel
rmse_diff = parcel_df.loc['AR-IRLS', 'rmse'] - parcel_df.loc['iRRR', 'rmse']
diff_lim = np.nanpercentile(np.abs(rmse_diff), 98)
print(f"RMSE AR-IRLS - iRRR: median = {np.nanmedian(rmse_diff):.3g} {y_unit}, "
      f"iRRR lower in {np.mean(rmse_diff > 0) * 100:.1f}% of {len(rmse_diff)} parcels")
val_by_parcel = rmse_diff.to_dict()
vertex_vals = np.array([val_by_parcel.get(p, np.nan) for p in vertex_parcel])
X_surf = xr.DataArray(
    np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
    dims=['vertex', 'chromo'],
    coords={'chromo': ['HbO', 'HbR'],
            'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
)
surf_path = os.path.join(plot_dir, f'surf_rmse_diff_AR-IRLS_minus_iRRR{out_tag}')
image_recon_multi_view(
    X_ts=X_surf, head=head, cmap='seismic', clim=(-diff_lim, diff_lim),
    view_type='hbo_brain', title_str=f'RMSE AR-IRLS - iRRR ({y_unit})',
    SAVE=True, filename=surf_path, wdw_size=(1600, 800),
)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(plt.imread(surf_path + '.png'))
ax.axis('off')
ax.set_title(f'RMSE AR-IRLS − iRRR ({y_unit}), mean over {len(subjects)} subjects '
             f'(red: iRRR lower; {select_chromo}, {fit_space})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, f'model_cmp_rmse_diff_AR-IRLS_iRRR_surf{out_tag}.png'), dpi=150)
plt.show()


#%% HRF at the parcel nearest to Cz: EEG-informed models
len_delay = 15 # Delay time in HRF (sec); must match run_model_cont_EEG_fNIRS.py
# nearest brain vertex (among the modeled parcels) to the Cz landmark of the standard head
cz_pos = head.landmarks.sel(label='Cz').pint.dequantify().values
vertex_pos = head.brain.vertices.pint.dequantify().values
is_modeled = np.isin(vertex_parcel, parcels)
cz_dist = np.linalg.norm(vertex_pos[is_modeled] - cz_pos, axis=1)
cz_parcel = vertex_parcel[is_modeled][np.argmin(cz_dist)]
print(f"Parcel nearest to Cz: {cz_parcel} ({cz_dist.min():.1f} mm)")

def load_hrf(prefix, key='betas_eeg'):
    with open(get_betas_path(prefix), 'rb') as f:
        return pickle.load(f)[key].sel(chromo=select_chromo, parcel=cz_parcel)

eeg_hrf = {m: [] for m in ['AR-IRLS', 'iRRR']}
for subject in subjects:
    data_dir = os.path.join(eeg_der_dir, subject)
    eeg_hrf['AR-IRLS'].append(load_hrf(os.path.join(data_dir, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')).values)
    eeg_hrf['iRRR'].append(load_hrf(os.path.join(data_dir, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}')).values)
eeg_hrf = {m: np.array(v) for m, v in eeg_hrf.items()}
with open(group_betas_file, 'rb') as f:
    group_hrf = pickle.load(f)['betas_eeg'].sel(chromo=select_chromo, parcel=cz_parcel).values
n_delay = eeg_hrf['AR-IRLS'].shape[1]
delay_t = np.arange(n_delay) * (len_delay / n_delay)

# mean +/- SEM across subjects
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
for model_name, hrf in eeg_hrf.items():
    hrf_mean, hrf_sem = hrf.mean(0), hrf.std(0, ddof=1) / np.sqrt(len(hrf))
    ax.plot(delay_t, hrf_mean, color=model_colors[model_name], lw=2, label=f'{model_name} (mean ± SEM)')
    ax.fill_between(delay_t, hrf_mean - hrf_sem, hrf_mean + hrf_sem, color=model_colors[model_name], alpha=0.2, lw=0)
ax.plot(delay_t, group_hrf, color=model_colors['iRRR_group'], lw=2, label='iRRR_group')
ax.set_xlabel('Delay (s)')
ax.set_ylabel(f'{select_chromo} per unit EEG (a.u.)')
ax.axhline(0, color='gray', lw=0.5)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize='small', frameon=False)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.set_title(f'EEG-informed HRF at the parcel nearest to Cz: {cz_parcel} (n={len(subjects)} subjects)')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, f'model_cmp_hrf_Cz_{cz_parcel}.png'), dpi=150)
plt.show()

#%% Single subject parcel HRF: the n_cz_parcels parcels nearest to Cz
select_subject = 'sub-723'
n_cz_parcels = 4
# parcel distance to Cz = its nearest brain vertex (same measure as cz_parcel above)
cz_parcel_dist = pd.Series(cz_dist).groupby(vertex_parcel[is_modeled]).min().nsmallest(n_cz_parcels)
print(f"{n_cz_parcels} parcels nearest to Cz:\n{cz_parcel_dist.round(1).to_string()}")

data_dir = os.path.join(eeg_der_dir, select_subject)
subj_hrf = dict()
for model_name, prefix in [('AR-IRLS', f'{select_subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}'),
                           ('iRRR', f'{select_subject}_{eeg_reg_type}_iRRR_{hp_flag}')]:
    with open(get_betas_path(os.path.join(data_dir, prefix)), 'rb') as f:
        subj_hrf[model_name] = pickle.load(f)['betas_eeg'].sel(chromo=select_chromo)

# superior view of each parcel (red) on the brain, with the Cz landmark marked
from cedalion.vis.anatomy.image_recon import image_recon
from cedalion.vis.blocks import plot_labeled_points
cz_parcel_pngs = dict()
for parcel in cz_parcel_dist.index:
    vertex_vals = np.where(vertex_parcel == parcel, 1.0, np.nan)  # NaN -> gray
    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    p0, _, _ = image_recon(X_surf, head, cmap='Reds', clim=(0, 1), view_type='hbo_brain',
                           view_position='superior', off_screen=True, wdw_size=(800, 800))
    plot_labeled_points(p0, head.landmarks.sel(label=['Cz']))
    cz_parcel_pngs[parcel] = os.path.join(plot_dir, f'surf_parcel_{parcel}.png')
    p0.screenshot(cz_parcel_pngs[parcel])
    p0.close()

fig, axes = plt.subplots(2, n_cz_parcels, figsize=(4 * n_cz_parcels, 7.6), squeeze=False,
                         gridspec_kw={'height_ratios': [1, 1.1]})
for ax in axes[0, 1:]:
    ax.sharey(axes[0, 0])
for ax, surf_ax, (parcel, dist) in zip(axes[0], axes[1], cz_parcel_dist.items()):
    surf_ax.imshow(plt.imread(cz_parcel_pngs[parcel]))
    surf_ax.axis('off')
    for model_name, hrf in subj_hrf.items():
        ax.plot(delay_t, hrf.sel(parcel=parcel).values, color=model_colors[model_name], lw=2, label=model_name)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.set_title(f'{parcel}\n({dist:.1f} mm from Cz)', fontsize='medium')
    ax.set_xlabel('Delay (s)')
axes[0, 0].set_ylabel(f'{select_chromo} per unit EEG (a.u.)')
axes[0, 0].legend(loc='upper right', fontsize='small', frameon=False)
axes[1, 0].text(0, 0.5, 'Parcel location\n(superior view,\nred; Cz marked)', transform=axes[1, 0].transAxes,
                ha='right', va='center', fontsize='small')
fig.suptitle(f'{select_subject}: EEG-informed HRF at the {n_cz_parcels} parcels nearest to Cz')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, f'{select_subject}_hrf_{n_cz_parcels}_parcels_nearest_Cz.png'), dpi=150)
plt.show()

#%% ===== event-based model comparison, scored inside the mnt trials only =====
# Models, all fit on the same shared Y_all (drift and GSR already OLS-regressed out):
#   1. Event-based: stimulus-locked Gaussian-basis HRFs for mnt-correct / mnt-incorrect
#      (run_model_EEG_inform_parcel_based.py, onlyStim)
#   2. mnt AR-IRLS / 3. mnt iRRR: delayed Cz EEG regressors per trial type, zeroed outside that
#      trial type's trials (run_model_cont_EEG_city_mnt.py, cont_EEG_cz_mnt_correct_incorrect)
# The cont mnt models only predict inside the trial windows, so all three are scored only on the
# time points inside the mnt_correct or mnt_incorrect trials (onset -> onset + duration).
import glob
event_reg_type = 'event-based_onParcel_onlyStim'  # must match run_model_EEG_inform_parcel_based.py
mnt_reg_type = 'cont_EEG_cz_mnt_correct_incorrect'  # must match run_model_cont_EEG_city_mnt.py
# same trial selection as run_model_cont_EEG_city_mnt.py
trial_type_selectors = {
    'mnt_correct': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] == 0),
    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),
}
ev_models = ['Event-based', 'mnt AR-IRLS', 'mnt iRRR']
ev_model_colors = {'Event-based': '#eda100', 'mnt AR-IRLS': '#2a78d6', 'mnt iRRR': '#eb6834'}
ev_plot_dir = os.path.join(eeg_der_dir, 'HRF_surf', 'group', f'{mnt_reg_type}_model_cmp')
os.makedirs(ev_plot_dir, exist_ok=True)
rmse_scale = 1e6 if y_unit == 'µM' else 1  # match the cont EEG section's RMSE unit

def get_trial_masks(subject, Y_da):
    """Bool mask over Y_da.time per trial type, True inside that type's trials.
    Y_all keeps each run's original fNIRS sample index in 'samples' (runs in run-01/02/03 order),
    so samples * dt is the run time in the same frame as the events.tsv onsets."""
    samples = Y_da.samples.values
    dt = np.median(np.diff(Y_da.time.values))
    seg_starts = np.r_[0, np.where(np.diff(samples) != 1)[0] + 1]
    seg_stops = np.r_[seg_starts[1:], len(samples)]
    ev_files = sorted(glob.glob(os.path.join(project_path, subject, 'nirs', f'{subject}_task-gradCPT_run-*_events.tsv')))
    assert len(ev_files) == len(seg_starts), f"{subject}: {len(ev_files)} events.tsv for {len(seg_starts)} runs in Y_all"
    masks = {tt: np.zeros(len(samples), dtype=bool) for tt in trial_type_selectors}
    for seg_start, seg_stop, ev_file in zip(seg_starts, seg_stops, ev_files):
        ev_df = pd.read_csv(ev_file, sep='\t')
        run_t = samples[seg_start:seg_stop] * dt
        for tt, selector in trial_type_selectors.items():
            trials_df = ev_df[selector(ev_df)]
            for onset, duration in zip(trials_df['onset'].values, trials_df['duration'].values):
                masks[tt][seg_start:seg_stop] |= (run_t >= onset) & (run_t < onset + duration)
    return masks

ev_rows = []
ev_subjects = []
for mnt_betas_file in sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', 'betas', f'sub-*_{mnt_reg_type}_{NOISE_MODEL}_{hp_flag}_betas.pkl'))):
    mnt_prefix = get_prefix_from_betas_path(mnt_betas_file)
    data_dir = os.path.dirname(mnt_prefix)
    subject = os.path.basename(data_dir)
    if subject in excluded_subj:
        continue
    mnt_irrr_prefix = os.path.join(data_dir, f'{subject}_{mnt_reg_type}_iRRR_{hp_flag}')
    event_prefix = os.path.join(data_dir, f'{subject}_{event_reg_type}_{NOISE_MODEL}_{hp_flag}')
    Y_all_path = get_shared_Y_all_path(data_dir, subject, hp_flag)
    req_files = [Y_all_path, get_dm_all_path(mnt_prefix),
                 get_betas_path(mnt_irrr_prefix), get_stats_path(mnt_irrr_prefix),
                 get_betas_path(event_prefix), get_dm_all_path(event_prefix)]
    if not all(os.path.exists(f) for f in req_files):
        print(f"{subject}: missing event-based or mnt AR-IRLS / iRRR results, skipping.")
        continue
    print(f"Processing {subject} (event-based comparison)")

    with gzip.open(Y_all_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(get_dm_all_path(mnt_prefix), 'rb') as f:
        mnt_dm = pickle.load(f)
    with open(mnt_betas_file, 'rb') as f:
        mnt_betas = pickle.load(f)['betas']
    with open(get_betas_path(mnt_irrr_prefix), 'rb') as f:
        mnt_irrr_betas = pickle.load(f)['betas']
    with open(get_stats_path(mnt_irrr_prefix), 'rb') as f:
        mnt_irrr_mu = pickle.load(f)['intercept']  # (parcel, 1)
    with gzip.open(get_dm_all_path(event_prefix), 'rb') as f:
        event_dm = pickle.load(f)
    with open(get_betas_path(event_prefix), 'rb') as f:
        event_betas = pickle.load(f)['betas']

    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    parcels, Y_partial = Y_da.parcel.values, Y_da.values
    trial_masks = get_trial_masks(subject, Y_da)

    def get_X_B(dm, betas):
        X_da = dm.common.sel(chromo=select_chromo).transpose('time', 'regressor')
        assert np.allclose(X_da.time.values, Y_da.time.values), f"{subject}: DM / Y_all time mismatch"
        B = betas.sel(chromo=select_chromo, parcel=parcels, regressor=X_da.regressor.values) \
                 .transpose('regressor', 'parcel').values
        return X_da.values, B

    # all regressors of each model (both trial types); none of the fits has an intercept except iRRR
    X_event, B_event = get_X_B(event_dm, event_betas)
    X_mnt, B_mnt = get_X_B(mnt_dm, mnt_betas)
    _, B_mnt_irrr = get_X_B(mnt_dm, mnt_irrr_betas)
    mnt_irrr_mu_s = pd.Series(mnt_irrr_mu.ravel(), index=mnt_irrr_betas.parcel.values)[parcels].values
    Y_hat_ev = {
        'Event-based': X_event @ B_event,
        'mnt AR-IRLS': X_mnt @ B_mnt,
        'mnt iRRR': X_mnt @ B_mnt_irrr + mnt_irrr_mu_s,
    }
    # whiten the whole series (AR-IRLS models only) before masking to the trial windows
    Y_fit_space = {m: to_fit_space(Y_partial, Y_hat, m) for m, Y_hat in Y_hat_ev.items()}
    for tt, mask in trial_masks.items():
        print(f"  {tt}: {mask.sum()} time points")
        for model_name, (Y_m, Y_hat_m) in Y_fit_space.items():
            rmse, r2 = fit_metrics(Y_m[mask], Y_hat_m[mask])
            ev_rows.append(pd.DataFrame({'subject': subject, 'trial_type': tt, 'model': model_name,
                                         'parcel': parcels, 'n_time': mask.sum(),
                                         'rmse': rmse * rmse_scale, 'r2': r2}))
    ev_subjects.append(subject)

ev_metrics_df = pd.concat(ev_rows, ignore_index=True)
if is_save:
    ev_metrics_df.to_csv(os.path.join(ev_plot_dir, f'model_cmp_rmse_r2_per_parcel_mnt_trials{out_tag}.csv'), index=False)

#%% summarize: mean over parcels per subject, then mean +/- SEM across subjects, per trial type
ev_subj_df = ev_metrics_df.groupby(['trial_type', 'subject', 'model'])[['rmse', 'r2']].mean().reset_index()
ev_summary_df = ev_subj_df.groupby(['trial_type', 'model'])[['rmse', 'r2']].agg(['mean', 'sem'])
for tt in trial_type_selectors:
    print(f"--- {tt}: R2 per subject (mean over parcels)")
    print(ev_subj_df[ev_subj_df.trial_type == tt].pivot(index='subject', columns='model', values='r2')[ev_models]
          .to_string(float_format='%.4f'))
print(ev_summary_df.to_string(float_format='%.4g'))

#%% bar plots: rows = metric, columns = trial type; per-subject bars plus the mean +/- SEM
x_labels = ev_subjects + [f'Mean\n(n={len(ev_subjects)})']
x = np.arange(len(x_labels))
bar_w = 0.8 / len(ev_models)
fig, axes = plt.subplots(len(metric_info), len(trial_type_selectors), sharex=True, squeeze=False,
                         figsize=(max(12, 2 * len(x_labels)), 7))
for c_i, tt in enumerate(trial_type_selectors):
    tt_subj_df = ev_subj_df[ev_subj_df.trial_type == tt]
    n_time = ev_metrics_df[ev_metrics_df.trial_type == tt].groupby('subject')['n_time'].first()[ev_subjects]
    for r_i, (metric, ylabel) in enumerate(metric_info.items()):
        ax = axes[r_i, c_i]
        for m_i, model_name in enumerate(ev_models):
            subj_vals = tt_subj_df[tt_subj_df.model == model_name].set_index('subject')[metric][ev_subjects].values
            vals = np.append(subj_vals, ev_summary_df.loc[(tt, model_name), (metric, 'mean')])
            err = np.append(np.full(len(ev_subjects), np.nan), ev_summary_df.loc[(tt, model_name), (metric, 'sem')])
            ax.bar(x + (m_i - (len(ev_models) - 1) / 2) * bar_w, vals, bar_w, yerr=err,
                   color=ev_model_colors[model_name], edgecolor='white', linewidth=2,
                   error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': '#555555'}, label=model_name)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(len(ev_subjects) - 0.5, color='gray', lw=0.5, ls='--')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        if c_i == 0:
            ax.set_ylabel(ylabel)
    axes[0, c_i].set_title(f'{tt} trials (time points per subject: {n_time.min()}–{n_time.max()})')
    axes[-1, c_i].set_xticks(x, x_labels, rotation=45, ha='right')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.95),
           ncols=len(ev_models), frameon=False)
fig.suptitle(f'Event-based model comparison inside mnt trials only, mean over parcels ({select_chromo}, in-sample, {fit_space})')
fig.tight_layout(rect=(0, 0, 1, 0.92))
if is_save:
    fig.savefig(os.path.join(ev_plot_dir, f'model_cmp_rmse_r2_bar_mnt_trials{out_tag}.png'), dpi=150)
plt.show()

#%% HRF at the parcel nearest to Cz (cz_parcel): event-based models, per trial type
def plot_ev_hrf(subject_list, parcel, title, save_name):
    """Event-based model HRFs at one parcel, columns = trial type. The event-based HRF (uM per trial,
    vs time from onset) and the mnt cont EEG HRFs (per unit EEG, vs delay) have different units, so
    they get separate rows. Mean +/- SEM across subject_list (a single line for one subject)."""
    ev_hrf = {tt: {m: [] for m in ev_models} for tt in trial_type_selectors}
    for subject in subject_list:
        data_dir = os.path.join(eeg_der_dir, subject)
        with open(get_betas_path(os.path.join(data_dir, f'{subject}_{event_reg_type}_{NOISE_MODEL}_{hp_flag}')), 'rb') as f:
            event_hrf_da = pickle.load(f)['hrf_estimate'].sel(chromo=select_chromo, parcel=parcel)
        mnt_hrf = dict()
        for model_name, prefix in [('mnt AR-IRLS', f'{subject}_{mnt_reg_type}_{NOISE_MODEL}_{hp_flag}'),
                                   ('mnt iRRR', f'{subject}_{mnt_reg_type}_iRRR_{hp_flag}')]:
            with open(get_betas_path(os.path.join(data_dir, prefix)), 'rb') as f:
                mnt_hrf[model_name] = pickle.load(f)['betas_eeg_per_type']
        for tt in trial_type_selectors:
            # event-based trial types are named 'mnt-correct' / 'mnt-incorrect'
            ev_hrf[tt]['Event-based'].append(event_hrf_da.sel(trial_type=tt.replace('_', '-'))
                                             .pint.to('micromolar').pint.dequantify().values)
            for model_name, hrf_per_type in mnt_hrf.items():
                ev_hrf[tt][model_name].append(hrf_per_type[tt].sel(chromo=select_chromo, parcel=parcel).values)
    event_t = event_hrf_da.time.values
    n_delay = len(ev_hrf['mnt_correct']['mnt AR-IRLS'][0])
    ev_delay_t = np.arange(n_delay) * (len_delay / n_delay)

    is_group = len(subject_list) > 1
    fig, axes = plt.subplots(2, len(trial_type_selectors), figsize=(6.5 * len(trial_type_selectors), 8), squeeze=False)
    for c_i, tt in enumerate(trial_type_selectors):
        for r_i, (row_models, t_axis) in enumerate([(['Event-based'], event_t),
                                                     (['mnt AR-IRLS', 'mnt iRRR'], ev_delay_t)]):
            ax = axes[r_i, c_i]
            for model_name in row_models:
                hrf = np.array(ev_hrf[tt][model_name])
                hrf_mean = hrf.mean(0)
                ax.plot(t_axis, hrf_mean, color=ev_model_colors[model_name], lw=2,
                        label=f'{model_name} (mean ± SEM)' if is_group else model_name)
                if is_group:
                    hrf_sem = hrf.std(0, ddof=1) / np.sqrt(len(hrf))
                    ax.fill_between(t_axis, hrf_mean - hrf_sem, hrf_mean + hrf_sem,
                                    color=ev_model_colors[model_name], alpha=0.2, lw=0)
            ax.axhline(0, color='gray', lw=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize='small', frameon=False)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
        axes[0, c_i].set_title(f'{tt}: event-based HRF')
        axes[0, c_i].set_xlabel('Time from trial onset (s)')
        axes[1, c_i].set_title(f'{tt}: trial-masked cont EEG HRF')
        axes[1, c_i].set_xlabel('Delay (s)')
    axes[0, 0].set_ylabel(f'{select_chromo} (µM)')
    axes[1, 0].set_ylabel(f'{select_chromo} per unit EEG (a.u.)')
    fig.suptitle(title)
    fig.tight_layout()
    if is_save:
        fig.savefig(os.path.join(ev_plot_dir, save_name), dpi=150)
    plt.show()

plot_ev_hrf(ev_subjects, cz_parcel,
            f'Event-based model HRFs at the parcel nearest to Cz: {cz_parcel} (n={len(ev_subjects)} subjects)',
            f'model_cmp_hrf_Cz_{cz_parcel}.png')

#%% single subject: event-based model HRFs at one parcel
select_subject = 'sub-723'
select_ev_parcel = 'SalVentAttnA_FrMed_5_LH'
plot_ev_hrf([select_subject], select_ev_parcel,
            f'{select_subject}: event-based model HRFs at {select_ev_parcel}',
            f'{select_subject}_hrf_{select_ev_parcel}.png')


#%% ===== AR-IRLS vs AR-iRRR, both scored on the AR-whitened Y_partial =====
# AR-iRRR (run_model_cont_EEG_fNIRS_iRRR_AR.py) fits iRRR on Y_partial prewhitened per parcel with the
# AR filter re-estimated from the AR-IRLS residuals (saved as betas_dict['ar_filters']); X is not
# whitened, so its prediction X @ C + mu is already in the whitened space. Both models are scored
# against the same target f * Y_partial, using those saved filters:
#   AR-IRLS: f * (X @ B_ar)        AR-iRRR: X @ C + mu
arw_reg_type = f'{eeg_reg_type}_iRRR_AR'  # must match run_model_cont_EEG_fNIRS_iRRR_AR.py
arw_models = ['AR-IRLS', 'AR-iRRR']
arw_model_colors = {'AR-IRLS': '#2a78d6', 'AR-iRRR': '#e87ba4'}

arw_rows = []
arw_subjects = []
for arw_betas_file in sorted(glob.glob(os.path.join(eeg_der_dir, 'sub-*', 'betas', f'sub-*_{arw_reg_type}_{hp_flag}_betas.pkl'))):
    arw_prefix = get_prefix_from_betas_path(arw_betas_file)
    data_dir = os.path.dirname(arw_prefix)
    subject = os.path.basename(data_dir)
    if subject in excluded_subj:
        continue
    ar_irls_prefix = os.path.join(data_dir, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    Y_all_path = get_Y_all_path(ar_irls_prefix)
    req_files = [Y_all_path, get_dm_all_path(ar_irls_prefix), get_betas_path(ar_irls_prefix), get_stats_path(arw_prefix)]
    if not all(os.path.exists(f) for f in req_files):
        print(f"{subject}: missing AR-IRLS or AR-iRRR results, skipping.")
        continue
    print(f"Processing {subject} (AR-IRLS vs AR-iRRR)")

    with gzip.open(Y_all_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(get_dm_all_path(ar_irls_prefix), 'rb') as f:
        dm_all = pickle.load(f)
    with open(get_betas_path(ar_irls_prefix), 'rb') as f:
        ar_betas = pickle.load(f)['betas']
    with open(arw_betas_file, 'rb') as f:
        arw_betas_dict = pickle.load(f)
    with open(get_stats_path(arw_prefix), 'rb') as f:
        arw_mu = pickle.load(f)['intercept']  # (parcel, 1)

    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    parcels, regressors = Y_da.parcel.values, X_da.regressor.values
    Y_partial, X_np = Y_da.values, X_da.values

    def get_B(betas):
        return betas.sel(chromo=select_chromo, parcel=parcels, regressor=regressors) \
                    .transpose('regressor', 'parcel').values

    ar_filters = [arw_betas_dict['ar_filters'][p] for p in parcels]
    arw_mu_s = pd.Series(arw_mu.ravel(), index=arw_betas_dict['betas'].parcel.values)[parcels].values
    Y_white = apply_ar_filters(Y_partial, ar_filters)  # first p samples of each parcel are NaN
    Y_hat_arw = {
        'AR-IRLS': apply_ar_filters(X_np @ get_B(ar_betas), ar_filters),
        'AR-iRRR': X_np @ get_B(arw_betas_dict['betas']) + arw_mu_s,
    }
    for model_name, Y_hat in Y_hat_arw.items():
        rmse, r2 = fit_metrics(Y_white, Y_hat)
        arw_rows.append(pd.DataFrame({'subject': subject, 'model': model_name, 'parcel': parcels,
                                      'rmse': rmse * rmse_scale, 'r2': r2}))
    arw_subjects.append(subject)

arw_metrics_df = pd.concat(arw_rows, ignore_index=True)
if is_save:
    arw_metrics_df.to_csv(os.path.join(plot_dir, 'model_cmp_AR-IRLS_vs_AR-iRRR_rmse_r2_per_parcel.csv'), index=False)

#%% summarize: mean over parcels per subject, then mean +/- SEM across subjects
arw_subj_df = arw_metrics_df.groupby(['subject', 'model'])[['rmse', 'r2']].mean().reset_index()
arw_summary_df = arw_subj_df.groupby('model')[['rmse', 'r2']].agg(['mean', 'sem']).reindex(arw_models)
print(arw_subj_df.pivot(index='subject', columns='model', values='r2')[arw_models].to_string(float_format='%.4f'))
print(arw_summary_df.to_string(float_format='%.4g'))

#%% bar plots: per-subject mean over parcels, plus the cross-subject mean +/- SEM
x_labels = arw_subjects + [f'Mean\n(n={len(arw_subjects)})']
x = np.arange(len(x_labels))
bar_w = 0.8 / len(arw_models)
fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.1 * len(x_labels)), 7), sharex=True)
for ax, (metric, ylabel) in zip(axes, metric_info.items()):
    for m_i, model_name in enumerate(arw_models):
        subj_vals = arw_subj_df[arw_subj_df.model == model_name].set_index('subject')[metric][arw_subjects].values
        vals = np.append(subj_vals, arw_summary_df.loc[model_name, (metric, 'mean')])
        err = np.append(np.full(len(arw_subjects), np.nan), arw_summary_df.loc[model_name, (metric, 'sem')])
        ax.bar(x + (m_i - (len(arw_models) - 1) / 2) * bar_w, vals, bar_w, yerr=err,
               color=arw_model_colors[model_name], edgecolor='white', linewidth=2,
               error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': '#555555'}, label=model_name)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(len(arw_subjects) - 0.5, color='gray', lw=0.5, ls='--')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
axes[0].set_title('Mean over parcels of whitened Y_hat_eeg vs whitened Y_partial (in-sample)')
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=len(arw_models), frameon=False)
axes[-1].set_xticks(x, x_labels, rotation=45, ha='right')
fig.suptitle(f'AR-IRLS vs AR-iRRR ({select_chromo}, {eeg_reg_type})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, 'model_cmp_AR-IRLS_vs_AR-iRRR_rmse_r2_bar.png'), dpi=150)
plt.show()

#%% brain surface maps: R2 per model (same color scale) and RMSE AR-IRLS - AR-iRRR, mean over subjects
def render_parcel_surface(val_by_parcel, cmap, clim, title_str, surf_path):
    """Render per-parcel values on the icbm152 brain (parcels not in val_by_parcel are gray)."""
    vertex_vals = np.array([val_by_parcel.get(p, np.nan) for p in vertex_parcel])
    X_surf = xr.DataArray(
        np.stack([vertex_vals, np.zeros(n_vertex)], axis=-1),
        dims=['vertex', 'chromo'],
        coords={'chromo': ['HbO', 'HbR'],
                'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))},
    )
    image_recon_multi_view(
        X_ts=X_surf, head=head, cmap=cmap, clim=clim,
        view_type='hbo_brain', title_str=title_str,
        SAVE=True, filename=surf_path, wdw_size=(1600, 800),
    )
    return surf_path + '.png'

arw_parcel_df = arw_metrics_df.groupby(['model', 'parcel'])[['rmse', 'r2']].mean()
# R2: linear 0 -> max over both models, R2 <= 0 in black (same scheme as the cont EEG section)
arw_r2_max = np.nanmax(arw_parcel_df['r2'])
arw_r2_black = -arw_r2_max / 255
arw_surf_paths = []
for model_name in arw_models:
    r2 = arw_parcel_df.loc[model_name, 'r2']
    arw_surf_paths.append((f'{model_name}: R$^2$ (black: R$^2$ ≤ 0)', render_parcel_surface(
        r2.where((r2 > 0) | r2.isna(), arw_r2_black).to_dict(), r2_cmap, (arw_r2_black, arw_r2_max),
        f'{model_name} R2', os.path.join(plot_dir, f'surf_r2_{model_name}_whitened'))))
# RMSE difference: positive (red) = AR-iRRR fits the whitened Y_partial better than AR-IRLS
arw_rmse_diff = arw_parcel_df.loc['AR-IRLS', 'rmse'] - arw_parcel_df.loc['AR-iRRR', 'rmse']
arw_diff_lim = np.nanpercentile(np.abs(arw_rmse_diff), 98)
print(f"RMSE AR-IRLS - AR-iRRR: median = {np.nanmedian(arw_rmse_diff):.3g} {y_unit}, "
      f"AR-iRRR lower in {np.mean(arw_rmse_diff > 0) * 100:.1f}% of {len(arw_rmse_diff)} parcels")
arw_surf_paths.append((f'RMSE AR-IRLS − AR-iRRR ({y_unit}; red: AR-iRRR lower)', render_parcel_surface(
    arw_rmse_diff.to_dict(), 'seismic', (-arw_diff_lim, arw_diff_lim),
    f'RMSE AR-IRLS - AR-iRRR ({y_unit})', os.path.join(plot_dir, 'surf_rmse_diff_AR-IRLS_minus_AR-iRRR_whitened'))))

fig, axes = plt.subplots(len(arw_surf_paths), 1, figsize=(10, 5 * len(arw_surf_paths)))
for ax, (title, surf_png) in zip(axes, arw_surf_paths):
    ax.imshow(plt.imread(surf_png))
    ax.axis('off')
    ax.set_title(title)
fig.suptitle(f'AR-IRLS vs AR-iRRR on whitened Y_partial, mean over {len(arw_subjects)} subjects ({select_chromo})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, 'model_cmp_AR-IRLS_vs_AR-iRRR_surf.png'), dpi=150)
plt.show()
