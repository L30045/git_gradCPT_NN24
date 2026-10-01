#%% load library
# Compare how well the EEG-informed HRF models reconstruct Y_partial (fNIRS after the drift and
# GSR OLS stages), using betas from:
#   1. AR-IRLS (run_model_cont_EEG_fNIRS.py, 3-stage_bspline-test)
#   2. per-subject iRRR (run_model_cont_EEG_fNIRS_iRRR.py)
#   3. group iRRR (run_model_cont_EEG_fNIRS_iRRR_group.py)
# All three share the same Y_all / dm_all, so only the betas differ. RMSE and R2 are computed
# per subject per parcel (in-sample) and summarized with bar plots.
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

rows = []
subjects = []
y_unit = None
for subject in group_stats['subjects']:
    data_dir = os.path.join(eeg_der_dir, subject)
    ar_irls_prefix = os.path.join(data_dir, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    irrr_prefix = os.path.join(data_dir, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}')
    req_files = [f'{ar_irls_prefix}_Y_all.pkl.gz', f'{ar_irls_prefix}_dm_all.pkl.gz',
                 f'{ar_irls_prefix}_betas.pkl', f'{irrr_prefix}_betas.pkl', f'{irrr_prefix}_stats.pkl']
    if not all(os.path.exists(f) for f in req_files):
        print(f"{subject}: missing AR-IRLS or iRRR results, skipping.")
        continue
    print(f"Processing {subject}")

    with gzip.open(f'{ar_irls_prefix}_Y_all.pkl.gz', 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(f'{ar_irls_prefix}_dm_all.pkl.gz', 'rb') as f:
        dm_all = pickle.load(f)
    with open(f'{ar_irls_prefix}_betas.pkl', 'rb') as f:
        ar_betas = pickle.load(f)['betas']
    with open(f'{irrr_prefix}_betas.pkl', 'rb') as f:
        irrr_betas = pickle.load(f)['betas']
    with open(f'{irrr_prefix}_stats.pkl', 'rb') as f:
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

    def get_B(betas):
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
        rmse, r2 = fit_metrics(Y_partial, Y_hat)
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
    metrics_df.to_csv(os.path.join(plot_dir, 'model_cmp_rmse_r2_per_parcel.csv'), index=False)

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
axes[0].set_title('Mean over parcels of Y_hat_eeg vs Y_partial (in-sample)')
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.12), ncols=len(models), frameon=False)
axes[-1].set_xticks(x, x_labels, rotation=45, ha='right')
fig.suptitle(f'EEG HRF model comparison ({select_chromo}, {eeg_reg_type})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, 'model_cmp_rmse_r2_bar.png'), dpi=150)
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
        surf_path = os.path.join(plot_dir, f'surf_{metric}_{model_name}')
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
fig.suptitle(f'Per-parcel fit of Y_hat_eeg vs Y_partial, mean over {len(subjects)} subjects ({select_chromo})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, 'model_cmp_rmse_r2_surf.png'), dpi=150)
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
surf_path = os.path.join(plot_dir, 'surf_rmse_diff_AR-IRLS_minus_iRRR')
image_recon_multi_view(
    X_ts=X_surf, head=head, cmap='seismic', clim=(-diff_lim, diff_lim),
    view_type='hbo_brain', title_str=f'RMSE AR-IRLS - iRRR ({y_unit})',
    SAVE=True, filename=surf_path, wdw_size=(1600, 800),
)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(plt.imread(surf_path + '.png'))
ax.axis('off')
ax.set_title(f'RMSE AR-IRLS − iRRR ({y_unit}), mean over {len(subjects)} subjects '
             f'(red: iRRR lower; {select_chromo})')
fig.tight_layout()
if is_save:
    fig.savefig(os.path.join(plot_dir, 'model_cmp_rmse_diff_AR-IRLS_iRRR_surf.png'), dpi=150)
plt.show()
