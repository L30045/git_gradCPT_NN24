#%% AR noise structure of every parcel: BIC order and coefficients across parcels, runs and subjects
# Parcel-wise version of check_AR_structure.py (BIC method only). For every subject and gradCPT run separately,
# fit the event-based OLS model of run_Bayes_image-space_gaussian-kernel_mnt-correct.py on all sensitive parcels:
#
#     y_p = X h_p + N b_p + e_p      (X: mnt-correct Gaussian kernels, N: Legendre drift + GSR of that run)
#
# then select the AR order of each parcel's residuals with cedalion.math.ar_model.bic_arfit (orders 0..pmax,
# stops at the first BIC increase; the order selection inside Cedalion AR-IRLS) and keep its AR coefficients.
# Preprocessing as in the Bayes script: 0.02 Hz highpass, truncation to [first onset - t_pre, last onset + t_post],
# event regressors highpassed like y (hpDesign), y standardized per run and parcel.
#
# To compare coefficient vectors of different order, each is zero-padded to pmax (an AR(P) model is an AR(pmax)
# model with phi_k = 0 for k > P). Distances below are Euclidean distances between these padded vectors.
#
# Outputs (derivatives/eeg/check_AR_structure/all_parcels/):
#   AR_bic_parcels.nc                     order (subject, run, parcel) and phi (subject, run, parcel, lag)
#   surf/group_order_by_run.png           group median AR order per parcel, one brain per run + all runs
#   surf/group_phi1_by_run.png            group mean lag-1 coefficient per parcel, same layout
#   surf/<subject>_order_by_run.png       per-subject AR order per parcel, one brain per run
#   order_across_parcels.png              distribution of the order over parcels, per subject and run
#   phi_distance.png                      coefficient distance: across parcels vs across runs vs across subjects
#   phi_sd_by_lag.png                     SD of each lag's coefficient across parcels / runs / subjects
#   phi_heatmap/<subject>_phi_heatmap.png parcel x lag coefficients, one panel per run, parcels sorted by network
import os
import sys
import glob
import gzip
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
git_path = os.path.join(modeling_path, os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *
import cedalion.io
import cedalion.dot
import cedalion.models.glm as glm
from cedalion.math.ar_model import bic_arfit
from cedalion.vis.anatomy.image_recon import image_recon_multi_view

#%% settings
select_chromo = 'HbO'
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']
l_cutoff = 0.02      # highpass cutoff (Hz), as in the Bayes script
pmax = 30            # max AR order (Cedalion glm.fit default ar_order)
is_refit = False     # False: reuse AR_bic_parcels.nc if it exists
is_plot_subject_surf = True   # per-subject surface maps (one render per subject x run, ~10 s each)
n_lag_show = 15      # lags shown in the heatmaps and the per-lag SD figure
order_clim = (0, 15)
phi1_clim = (0, 1)
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
cfg_GLM['drift_order'] = DRIFT_ORDER
run_colors = {'gradcpt1': '#2a78d6', 'gradcpt2': '#eb6834', 'gradcpt3': '#1baf7a'}
cmp_colors = {'across parcels': '#2a78d6', 'across runs': '#eb6834', 'across subjects': '#1baf7a'}

der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
out_dir = os.path.join(project_path, 'derivatives', 'eeg', 'check_AR_structure', 'all_parcels')
surf_dir = os.path.join(out_dir, 'surf')
heat_dir = os.path.join(out_dir, 'phi_heatmap')
for d in (out_dir, surf_dir, heat_dir):
    os.makedirs(d, exist_ok=True)
result_path = os.path.join(out_dir, 'AR_bic_parcels.nc')
gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])

#%% subjects: folders with both image-space input files, not excluded
subjects = []
for subj_dir in sorted(glob.glob(os.path.join(der_dir, 'sub-*'))):
    subject = os.path.basename(subj_dir)
    if subject in excluded_subj:
        continue
    if not all(os.path.exists(p) for p in get_image_space_paths(subject)):
        print(f"{subject}: missing input files, skipping.")
        continue
    subjects.append(subject)
print(f"{len(subjects)} subjects: {subjects}")

#%% sensitive parcels (same mask as the Bayes script; GSR is the mean over these parcels)
Adot = cedalion.io.load_Adot('/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/Adot_v26.nc')
Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
Adot_brain = Adot_brain.sel(vertex=~Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH', 'Background+FreeSurfer_Defined_Medial_Wall_RH']))
sensitivity_mask = (np.log10(Adot_brain[:, :, 1].sum('channel')) > -2).drop_vars('wavelength')
sensitive_parcels = np.unique(Adot_brain.sel(vertex=sensitivity_mask.values).parcel.values)
print(f"{len(sensitive_parcels)} sensitive parcels")


#%% per subject, per run: OLS on all parcels -> BIC order and coefficients per parcel
def ols_residuals(run, stim):
    """One run: highpass, truncate, hpDesign Gaussian kernels + drift + GSR, OLS on every parcel.
    Returns residuals (T, parcel) of the standardized data and the parcel names (non-finite parcels dropped)."""
    run = run.sel(parcel=run.parcel.isin(sensitive_parcels)).sel(chromo=[select_chromo]).pint.to('micromolar')
    run.time.attrs['units'] = units.s
    fs = 1 / np.median(np.diff(run.time.values))
    sos = butter(4, l_cutoff, btype='highpass', fs=fs, output='sos')
    run_units = run.pint.units
    run = xr.apply_ufunc(sosfiltfilt, sos, run.pint.dequantify(), input_core_dims=[[], ['time']],
                         output_core_dims=[['time']], exclude_dims={'time'},
                         ).transpose(*run.dims).pint.quantify(run_units).assign_coords(time=run.time)
    run.time.attrs['units'] = units.s

    mnt = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)].copy()
    mnt['trial_type'] = 'mnt-correct'
    t_start = max(stim['onset'].min() - cfg_HRF_basis['t_pre'].magnitude, run.time.values[0])
    t_stop = min(stim['onset'].max() + cfg_HRF_basis['t_post'].magnitude, run.time.values[-1])
    mnt = mnt[(mnt['onset'] >= t_start) & (mnt['onset'] <= t_stop)]

    ev_dm = glm.design_matrix.hrf_regressors(run, mnt, gauss_kernels)
    common = ev_dm.common.fillna(0)
    common = common.copy(data=sosfiltfilt(sos, np.asarray(common.values), axis=common.dims.index('time')))
    ev_dm.common = common.sel(time=slice(t_start, t_stop))
    run = run.sel(time=slice(t_start, t_stop))
    t0 = run.time.values[0]
    run = run.assign_coords(time=run.time.values - t0)
    run.time.attrs['units'] = units.s
    ev_dm.common = ev_dm.common.assign_coords(time=run.time.values)

    Y_all, dm_all, runs_updated = model.concatenate_runs_dms([run], [ev_dm])
    dm_all &= model.reduce(model.operator.and_, model.get_drift_legendre_regressors(runs_updated, cfg_GLM))
    dm_all &= model.reduce(model.operator.and_, model.get_global_mean_regressor(runs_updated))
    X = dm_all.common.fillna(0).sel(chromo=select_chromo).transpose('time', 'regressor').values
    Y = Y_all.sel(chromo=select_chromo).transpose('time', 'parcel').pint.dequantify()
    ok = np.isfinite(Y.values).all(0) & (Y.values.std(0) > 0)
    Y_v = Y.values[:, ok]
    Y_v = (Y_v - Y_v.mean(0)) / Y_v.std(0)
    return Y_v - X @ np.linalg.lstsq(X, Y_v, rcond=None)[0], Y.parcel.values[ok]


def bic_ar(e):
    """Order and AR coefficients (intercept dropped) from Cedalion's bic_arfit, and whether pmax was reached."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fit = bic_arfit(pd.Series(e), pmax=pmax)
    phi = np.asarray(fit.params)[1:]
    return len(phi), any('pmax was reached' in str(x.message) for x in w), phi


if os.path.exists(result_path) and not is_refit:
    res = xr.load_dataset(result_path)
    print(f"loaded {result_path}")
else:
    order = np.full((len(subjects), len(run_keys), len(sensitive_parcels)), np.nan)
    capped = np.zeros_like(order, dtype=bool)
    phi = np.full(order.shape + (pmax,), np.nan)
    p_index = {p: i for i, p in enumerate(sensitive_parcels)}
    t_begin = time.time()
    for s_i, subject in enumerate(subjects):
        stim_path, ts_path = get_image_space_paths(subject)
        with gzip.open(stim_path, 'rb') as f:
            all_stims = pickle.load(f)['stims']
        with open(ts_path, 'rb') as f:
            raw_runs = pickle.load(f)['parcel_ts']
        for r_k, run_key in enumerate(run_keys):
            ev_file = os.path.join(project_path, subject, 'nirs',
                                   f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv")
            if not os.path.exists(ev_file):
                print(f"{subject} {run_key}: no events.tsv, skipping.")
                continue
            try:   # first onset + first 3 reaction times
                r_i = model.match_run_keys_to_image_runs(subject, [run_key], all_stims, project_path)[run_key]
            except ValueError as err:
                print(f"{err} Skipping.")
                continue
            E, parcels = ols_residuals(raw_runs[r_i], all_stims[r_i])
            for j, parcel in enumerate(parcels):
                P, reached, phi_p = bic_ar(E[:, j])
                k = p_index[parcel]
                order[s_i, r_k, k] = P
                capped[s_i, r_k, k] = reached
                phi[s_i, r_k, k, :] = 0.0
                phi[s_i, r_k, k, :P] = phi_p
            print(f"{subject} {run_key}: {len(parcels)} parcels, median order "
                  f"{np.nanmedian(order[s_i, r_k]):.0f}  [{time.time() - t_begin:.0f} s]", flush=True)
    res = xr.Dataset(
        {'order': (('subject', 'run', 'parcel'), order),
         'capped': (('subject', 'run', 'parcel'), capped),
         'phi': (('subject', 'run', 'parcel', 'lag'), phi)},
        coords={'subject': subjects, 'run': run_keys, 'parcel': sensitive_parcels, 'lag': np.arange(1, pmax + 1)},
        attrs={'method': 'cedalion bic_arfit on event-based OLS residuals', 'pmax': pmax, 'chromo': select_chromo})
    res.to_netcdf(result_path)
    print(f"saved {result_path}")

subjects = list(res.subject.values)
n_s = len(subjects)
net = pd.Series([p.split('_')[0] for p in res.parcel.values], index=res.parcel.values)
parcel_sorted = net.sort_values(kind='stable').index.values   # parcels grouped by network

#%% text summary
o = res['order']
print(f"BIC order over all subject x run x parcel: median {float(o.median()):.0f}, "
      f"IQR {float(o.quantile(0.25)):.0f}-{float(o.quantile(0.75)):.0f}, "
      f"range {float(o.min()):.0f}-{float(o.max()):.0f}; reached pmax: {int(res['capped'].sum())}")
same_runs = (o.max('run') == o.min('run')) & o.notnull().all('run')
print(f"subject x parcel pairs whose runs all share one order: {float(same_runs.sum()) / float(o.notnull().all('run').sum()):.1%}")

#%% surface maps of the order and the lag-1 coefficient
head = cedalion.dot.get_standard_headmodel('icbm152')
vertex_parcel = head.brain.vertices.parcel.values
n_vertex = head.brain.nvertices


def render_surf(vals, parcels, title, path, clim, cmap='viridis'):
    """One brain-surface render (5 views) of one value per parcel; parcels without a value stay NaN."""
    by_parcel = dict(zip(parcels, vals))
    v = np.array([by_parcel.get(p, np.nan) for p in vertex_parcel])
    X_surf = xr.DataArray(np.stack([v, np.zeros(n_vertex)], axis=-1), dims=['vertex', 'chromo'],
                          coords={'chromo': ['HbO', 'HbR'], 'is_brain': ('vertex', np.ones(n_vertex, dtype=bool))})
    image_recon_multi_view(X_ts=X_surf, head=head, cmap=cmap, clim=clim, view_type='hbo_brain',
                           title_str=title, SAVE=True, filename=path, wdw_size=(1600, 800))
    plt.close('all')
    return path + '.png'


def stitch(pngs, titles, suptitle, path, n_col=2):
    """Put several surface renders into one figure."""
    n_row = int(np.ceil(len(pngs) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(8 * n_col, 4.2 * n_row), squeeze=False)
    for a in axes.flat:
        a.axis('off')
    for a, png, t in zip(axes.flat, pngs, titles):
        a.imshow(plt.imread(png))
        a.set_title(t, fontsize=13)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


tmp_dir = os.path.join(surf_dir, 'single')
os.makedirs(tmp_dir, exist_ok=True)
for var, stat, clim, label in [('order', 'median', order_clim, 'AR order (BIC)'),
                               ('phi1', 'mean', phi1_clim, 'lag-1 AR coefficient')]:
    da = res['order'] if var == 'order' else res['phi'].sel(lag=1)
    pngs, titles = [], []
    for run_key in run_keys + ['all runs']:
        sel = da if run_key == 'all runs' else da.sel(run=run_key)
        dims = ['subject', 'run'] if run_key == 'all runs' else ['subject']
        g = sel.median(dims) if stat == 'median' else sel.mean(dims)
        n = int(sel.notnull().any('parcel').sum())
        pngs.append(render_surf(g.values, g.parcel.values, label,
                                os.path.join(tmp_dir, f"group_{var}_{run_key.replace(' ', '-')}"), clim))
        titles.append(f"{run_key} ({stat} of {n} {'subject-runs' if run_key == 'all runs' else 'subjects'})")
    stitch(pngs, titles, f'Group {stat} {label} per parcel ({select_chromo})',
           os.path.join(surf_dir, f'group_{var}_by_run.png'))

if is_plot_subject_surf:
    for subject in subjects:
        pngs, titles = [], []
        for run_key in run_keys:
            v = res['order'].sel(subject=subject, run=run_key)
            if v.isnull().all():
                continue
            pngs.append(render_surf(v.values, v.parcel.values, 'AR order (BIC)',
                                    os.path.join(tmp_dir, f'{subject}_order_{run_key}'), order_clim))
            titles.append(f'{run_key} (median {float(v.median()):.0f})')
        stitch(pngs, titles, f'{subject}: AR order (BIC) per parcel ({select_chromo})',
               os.path.join(surf_dir, f'{subject}_order_by_run.png'))

#%% figure: distribution of the order over parcels, per subject and run
width = 0.8 / len(run_keys)
x = np.arange(n_s)
fig, ax = plt.subplots(figsize=(max(8, 0.6 * n_s + 2), 4.2), constrained_layout=True)
for j, run_key in enumerate(run_keys):
    data, pos = [], []
    for i, subject in enumerate(subjects):
        v = res['order'].sel(subject=subject, run=run_key).values
        v = v[np.isfinite(v)]
        if len(v):
            data.append(v)
            pos.append(x[i] + (j - (len(run_keys) - 1) / 2) * width)
    bp = ax.boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True, showfliers=False,
                    medianprops={'color': 'white', 'lw': 1.5}, whiskerprops={'color': run_colors[run_key]},
                    capprops={'color': run_colors[run_key]})
    for b in bp['boxes']:
        b.set(facecolor=run_colors[run_key], edgecolor=run_colors[run_key])
    ax.plot([], [], color=run_colors[run_key], lw=6, label=run_key)
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=60, ha='right')
ax.set(ylabel='AR order (samples)', title=f'BIC AR order across parcels, per subject and run '
                                          f'(box: IQR, whiskers: 1.5 IQR; {select_chromo})')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eeeeee', lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, fontsize=9, loc='upper left')
fig.savefig(os.path.join(out_dir, 'order_across_parcels.png'), dpi=150)
plt.close(fig)

#%% coefficient distances: across parcels (same subject, run) vs across runs (same subject, parcel)
#   vs across subjects (same run key, parcel). Per group, the mean pairwise distance of the padded phi vectors.
phi_v = res['phi'].values   # (subject, run, parcel, lag)
d_parcel = [pdist(phi_v[s, r][np.isfinite(phi_v[s, r, :, 0])]).mean()
            for s in range(n_s) for r in range(len(run_keys)) if np.isfinite(phi_v[s, r, :, 0]).sum() > 1]
d_run = [pdist(phi_v[s, :, p][np.isfinite(phi_v[s, :, p, 0])]).mean()
         for s in range(n_s) for p in range(phi_v.shape[2]) if np.isfinite(phi_v[s, :, p, 0]).sum() > 1]
d_subj = [pdist(phi_v[:, r, p][np.isfinite(phi_v[:, r, p, 0])]).mean()
          for r in range(len(run_keys)) for p in range(phi_v.shape[2]) if np.isfinite(phi_v[:, r, p, 0]).sum() > 1]
dists = {'across parcels': d_parcel, 'across runs': d_run, 'across subjects': d_subj}
units_txt = {'across parcels': 'one point per subject x run', 'across runs': 'one point per subject x parcel',
             'across subjects': 'one point per run x parcel'}
for k, v in dists.items():
    print(f"phi distance {k}: median {np.median(v):.3f} (IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}, n = {len(v)})")

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
bp = ax.boxplot(list(dists.values()), patch_artist=True, showfliers=False, widths=0.5,
                medianprops={'color': 'white', 'lw': 1.5})
for b, k in zip(bp['boxes'], dists):
    b.set(facecolor=cmp_colors[k], edgecolor=cmp_colors[k])
for w_i, k in enumerate(np.repeat(list(dists), 2)):
    bp['whiskers'][w_i].set_color(cmp_colors[k])
    bp['caps'][w_i].set_color(cmp_colors[k])
ax.set_xticks(range(1, len(dists) + 1))
ax.set_xticklabels([f'{k}\n({units_txt[k]})' for k in dists], fontsize=9)
ax.set(ylabel='Mean pairwise distance of AR coefficients',
       title=f'How much the AR coefficients differ ({select_chromo}, zero-padded to lag {pmax})')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eeeeee', lw=0.6)
ax.set_axisbelow(True)
fig.savefig(os.path.join(out_dir, 'phi_distance.png'), dpi=150)
plt.close(fig)

#%% SD of each lag's coefficient across parcels / runs / subjects (median over the groups)
sd = {'across parcels': res['phi'].std('parcel').median(['subject', 'run']),
      'across runs': res['phi'].std('run').median(['subject', 'parcel']),
      'across subjects': res['phi'].std('subject').median(['run', 'parcel'])}
lags = np.arange(1, n_lag_show + 1)
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
for k, v in sd.items():
    ax.plot(lags, v.sel(lag=lags), color=cmp_colors[k], lw=2, marker='o', ms=4, label=k)
ax.set(xlabel='Lag (samples)', ylabel='SD of AR coefficient (median over groups)', xticks=lags,
       title=f'Spread of each AR coefficient ({select_chromo})')
ax.set_ylim(bottom=0)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eeeeee', lw=0.6)
ax.legend(frameon=False)
fig.savefig(os.path.join(out_dir, 'phi_sd_by_lag.png'), dpi=150)
plt.close(fig)

#%% per-subject heatmaps: parcel (sorted by network) x lag, one panel per run
net_sorted = net[parcel_sorted].values
net_edges = np.flatnonzero(net_sorted[1:] != net_sorted[:-1]) + 1
net_mid = (np.r_[0, net_edges] + np.r_[net_edges, len(net_sorted)]) / 2
net_names = net_sorted[np.r_[0, net_edges]]
v_lim = float(np.nanpercentile(np.abs(res['phi'].sel(lag=lags).values), 99))
for subject in subjects:
    fig, axes = plt.subplots(1, len(run_keys), figsize=(4 * len(run_keys) + 1, 9), sharey=True,
                             constrained_layout=True)
    for a, run_key in zip(axes, run_keys):
        m = res['phi'].sel(subject=subject, run=run_key, parcel=parcel_sorted, lag=lags).values
        im = a.imshow(m, aspect='auto', cmap='RdBu_r', vmin=-v_lim, vmax=v_lim, interpolation='nearest',
                      extent=(lags[0] - 0.5, lags[-1] + 0.5, len(parcel_sorted), 0))
        for e in net_edges:
            a.axhline(e, color='#666666', lw=0.4)
        o_med = res['order'].sel(subject=subject, run=run_key).median()
        a.set(title=f'{run_key} (median order {float(o_med):.0f})' if np.isfinite(o_med) else f'{run_key} (missing)',
              xlabel='Lag (samples)')
    axes[0].set_yticks(net_mid)
    axes[0].set_yticklabels(net_names, fontsize=7)
    axes[0].set_ylabel('Parcel (grouped by network)')
    fig.colorbar(im, ax=axes, shrink=0.6, label='AR coefficient φ')
    fig.suptitle(f'{subject}: BIC AR coefficients per parcel ({select_chromo}; 0 beyond the selected order)')
    fig.savefig(os.path.join(heat_dir, f'{subject}_phi_heatmap.png'), dpi=120)
    plt.close(fig)
print(f"saved figures to {out_dir}")
