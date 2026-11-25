#%% Do runs (and subjects) share the same AR noise structure?
# For every subject and every gradCPT run separately, fit the event-based OLS model of
# run_Bayes_image-space_gaussian-kernel_mnt-correct.py on one image-space parcel:
#
#     y = X h + N b + e      (X: mnt-correct Gaussian kernels, N: Legendre drift + GSR of that run)
#
# then pick the AR order of the OLS residuals in two ways and keep the AR(P) coefficients of each:
#   seq: the rule of the Bayes script (start at p_start, add one lag while the last lag of an AR(P) fit is
#        significant, up to pmax)
#   bic: cedalion.math.ar_model.bic_arfit, the order selection inside Cedalion AR-IRLS (orders 0..pmax,
#        stops at the first order whose BIC is higher than the previous one; AutoReg with an intercept)
# Preprocessing is the same as the Bayes script: 0.02 Hz highpass, truncation to
# [first onset - t_pre, last onset + t_post], event regressors highpassed like y (hpDesign), y standardized per run.
#
# Outputs (derivatives/eeg/check_AR_structure/<parcel>/):
#   AR_structure.csv              one row per subject x run: order, flags and coefficients of both methods
#   AR_order_by_subject.png       bar plot of the AR order of each run, grouped by subject (one panel per method)
#   AR_coef_by_subject_<m>.png    one subplot per subject, AR coefficients vs lag, one line per run
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
import statsmodels.api as sm
from scipy.signal import butter, sosfiltfilt
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
git_path = os.path.join(modeling_path, os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *
import cedalion.io
import cedalion.models.glm as glm
from cedalion.math.ar_model import bic_arfit

#%% settings
select_parcel = os.environ.get('AR_PARCEL', 'DorsAttnB_PostC_2_RH')
select_chromo = 'HbO'
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']
l_cutoff = 0.02      # highpass cutoff (Hz), as in the Bayes script
p_start = 10         # AR-order rule of the Bayes script
pmax = 30            # max order of both methods (Cedalion glm.fit default ar_order)
methods = {'seq': f'sequential last-lag test (from {p_start})', 'bic': 'BIC (Cedalion bic_arfit)'}
p_alpha = 0.05
is_save = True
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
cfg_GLM['drift_order'] = DRIFT_ORDER
run_colors = {'gradcpt1': '#2a78d6', 'gradcpt2': '#eb6834', 'gradcpt3': '#1baf7a'}

der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
out_dir = os.path.join(project_path, 'derivatives', 'eeg', 'check_AR_structure', select_parcel)
os.makedirs(out_dir, exist_ok=True)
gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])


def select_ar_order(e, p_start, pmax, alpha):
    """Same rule as select_ar_order in run_Bayes_image-space_gaussian-kernel_mnt-correct.py, for one run:
    smallest P >= p_start whose last lag is not significant (OLS of e[t] on e[t-1..t-P]), capped at pmax.
    Returns P, the last-lag p-value of AR(P) and the AR(P) coefficients."""
    P = p_start
    while True:
        lags = np.column_stack([e[P - k - 1:len(e) - k - 1] for k in range(P)])   # column k = lag k+1
        fit = sm.OLS(e[P:], lags).fit()
        if fit.pvalues[-1] >= alpha or P == pmax:
            return P, fit.pvalues[-1], fit.params
        P += 1


def bic_ar_order(e, pmax):
    """Order and AR coefficients (intercept dropped) from Cedalion's bic_arfit, and whether it reached pmax."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fit = bic_arfit(pd.Series(e), pmax=pmax)
    reached = any('pmax was reached' in str(x.message) for x in w)
    phi = np.asarray(fit.params)[1:]   # params[0] is the intercept
    return len(phi), reached, phi


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
if select_parcel not in sensitive_parcels:
    raise ValueError(f"{select_parcel} is not among the {len(sensitive_parcels)} sensitive parcels.")


#%% per subject, per run: OLS fit -> AR order and coefficients
def fit_run(run, stim):
    """One run: highpass, truncate, hpDesign Gaussian kernels + drift + GSR, OLS; returns the residuals."""
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
    y = Y_all.sel(parcel=select_parcel, chromo=select_chromo).pint.dequantify().values
    y = (y - y.mean()) / y.std()
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0], len(mnt)


rows = []
t_begin = time.time()
for subject in subjects:
    stim_path, ts_path = get_image_space_paths(subject)
    with gzip.open(stim_path, 'rb') as f:
        all_stims = pickle.load(f)['stims']
    with open(ts_path, 'rb') as f:
        raw_runs = pickle.load(f)['parcel_ts']
    for run_key in run_keys:
        ev_file = os.path.join(project_path, subject, 'nirs', f"{subject}_task-gradCPT_run-{run_key[-1]:0>2}_events.tsv")
        if not os.path.exists(ev_file):
            print(f"{subject} {run_key}: no events.tsv, skipping.")
            continue
        # first onset + first 3 reaction times: in sub-656/668/670 two runs have first onsets < 0.01 s apart,
        # so the first onset alone maps two run keys to one run
        try:
            r_i = model.match_run_keys_to_image_runs(subject, [run_key], all_stims, project_path)[run_key]
        except ValueError as err:
            print(f"{err} Skipping.")
            continue
        e, n_trials = fit_run(raw_runs[r_i], all_stims[r_i])
        P_s, p_last, phi_s = select_ar_order(e, p_start, pmax, p_alpha)
        P_b, bic_reached, phi_b = bic_ar_order(e, pmax)
        rows.append({'subject': subject, 'run': run_key, 'n_samples': len(e), 'n_mnt_correct': n_trials,
                     'seq_order': P_s, 'seq_last_lag_p': p_last,
                     'seq_capped': bool(P_s == pmax and p_last < p_alpha),
                     'bic_order': P_b, 'bic_capped': bic_reached,
                     **{f'seq_phi_{k + 1}': phi_s[k] for k in range(P_s)},
                     **{f'bic_phi_{k + 1}': phi_b[k] for k in range(P_b)}})
        print(f"{subject} {run_key}: T = {len(e)}, seq order = {P_s} (last-lag p = {p_last:.3g}), "
              f"BIC order = {P_b}{' (pmax reached)' if bic_reached else ''}  [{time.time() - t_begin:.0f} s]",
              flush=True)

df = pd.DataFrame(rows)
if is_save:
    df.to_csv(os.path.join(out_dir, 'AR_structure.csv'), index=False)

#%% summary: does the order agree within subject / across subjects?
for m, m_label in methods.items():
    order_tab = df.pivot(index='subject', columns='run', values=f'{m}_order')
    order_tab['same_within_subject'] = order_tab[run_keys].nunique(axis=1) == 1
    print(f"\n--- {m_label} ---")
    print(order_tab.to_string())
    print(f"subjects whose runs all share one AR order: {order_tab['same_within_subject'].sum()} / {len(order_tab)}")
    print(f"distinct AR orders across all runs: {sorted(df[f'{m}_order'].unique())}; "
          f"capped at pmax = {pmax}: {df[f'{m}_capped'].sum()} / {len(df)} runs")
print(f"\nseq vs BIC order, Spearman r = {df['seq_order'].corr(df['bic_order'], method='spearman'):.2f}")

#%% figure 1: AR order of each run, grouped by subject (one panel per method)
n_s = len(subjects)
width = 0.8 / len(run_keys)
x = np.arange(n_s)
fig, axes = plt.subplots(len(methods), 1, figsize=(max(8, 0.55 * n_s + 2), 3.6 * len(methods)),
                         sharex=True, sharey=True, constrained_layout=True)
for ax, (m, m_label) in zip(axes, methods.items()):
    for j, run_key in enumerate(run_keys):
        sub = df[df['run'] == run_key].set_index('subject').reindex(subjects)
        ax.bar(x + (j - (len(run_keys) - 1) / 2) * width, sub[f'{m}_order'], width=width * 0.9,
               color=run_colors[run_key], label=run_key, edgecolor='white', linewidth=1)
    ax.axhline(pmax, color='#999999', lw=0.8, ls='--', label=f'pmax = {pmax}')
    if m == 'seq':
        ax.axhline(p_start, color='#bbbbbb', lw=0.8, ls=':', label=f'p_start = {p_start}')
    ax.set(ylabel='AR order (samples)', ylim=(0, pmax + 4), title=m_label)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#eeeeee', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc='upper left')
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(subjects, rotation=60, ha='right')
fig.suptitle(f'AR order of the event-based OLS residuals per run ({select_parcel}, {select_chromo})')
if is_save:
    fig.savefig(os.path.join(out_dir, 'AR_order_by_subject.png'), dpi=150)

#%% figure 2: AR coefficients vs lag, one subplot per subject, one line per run (one figure per method)
n_col = 5
n_row = int(np.ceil(n_s / n_col))
for m, m_label in methods.items():
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.2 * n_col, 2.6 * n_row), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    for a, subject in zip(axes.flat, subjects):
        for run_key in run_keys:
            r = df[(df['subject'] == subject) & (df['run'] == run_key)]
            if r.empty:
                continue
            P = int(r[f'{m}_order'].iloc[0])
            phi = r[[f'{m}_phi_{k + 1}' for k in range(P)]].values[0]
            a.plot(np.arange(1, P + 1), phi, color=run_colors[run_key], lw=1.5, marker='o', ms=3,
                   label=f'{run_key} (P = {P})')
        a.axhline(0, color='#cccccc', lw=0.8, zorder=0)
        a.set_title(subject, fontsize=10)
        a.spines[['top', 'right']].set_visible(False)
        a.legend(frameon=False, fontsize=7)
    for a in axes.flat[n_s:]:
        a.axis('off')
    for a in axes[-1]:
        a.set_xlabel('Lag (samples)')
    for a in axes[:, 0]:
        a.set_ylabel('AR coefficient φ')
    fig.suptitle(f'AR coefficients of the event-based OLS residuals, {m_label} ({select_parcel}, {select_chromo})')
    if is_save:
        fig.savefig(os.path.join(out_dir, f'AR_coef_by_subject_{m}.png'), dpi=150)
if is_save:
    print(f"saved to {out_dir}")
plt.show()
