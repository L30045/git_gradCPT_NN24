#%% Why is the continuous-EEG (contEEG) mnt-correct HRF flat?
# In the Gaussian-kernel model the HRF is B_gauss h. In the contEEG model (run_model_cont_EEG_city_mnt.py,
# mnt-correct only, as in cmp_hrf_gaussian-kernel_vs_cont-EEG_mnt-correct.py) the B-spline basis times the
# weights gives a delay kernel h(d) per EEG sample per volt, and a trial's predicted response is that kernel
# convolved with the trial's Cz EEG (onset -> onset + duration). Three possible reasons for a flat response:
#   1. the B-spline basis cannot represent an HRF shape      -> fit the onlyStim HRF with the 20 B-splines
#   2. the fitted B-spline weights are small / not different from 0 -> kernel with 95% CI, t of each weight
#   3. the EEG weighting cancels across trials               -> per-trial weight w_i = sum of Cz over the
#      trial (uV*s). For a kernel that is smooth over the 0.8 s trial, trial i's response is ~ w_i h(d), so
#      the average-trial response is ~ mean(w) h(d): it is flat if mean(w) is small relative to its spread.
#      The kernel needed to reproduce the onlyStim HRF is then HRF_onlyStim / mean(w).
# Needs the curves saved by cmp_hrf_gaussian-kernel_vs_cont-EEG_mnt-correct.py. Saves a PNG in this folder.
#%% load library
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELING = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODELING)
from params_setting import *

SUBJ_ID = 723
PARCELS = ['DorsAttnB_PostC_2_RH', 'SomMotA_25_LH']
CONT_HP = 0.02
subject = f'sub-{SUBJ_ID}'
cmp_path = os.path.join(project_path, 'derivatives', 'eeg', subject, 'cmp_HRF_DM', f'{subject}_cmp_HRF_DM_mnt-correct.pkl')
fig_path = os.path.join(MODELING, f'diag_contEEG_flat_hrf_{subject}.png')


def run_script(fname, replacements, cut_at=None):
    path = os.path.join(MODELING, fname)
    src = open(path).read()
    if cut_at is not None:
        src = src[:src.index(cut_at)]
    for old, new in replacements:
        assert src.count(old) == 1, f'{fname}: {old[:80]!r} found {src.count(old)} times'
        src = src.replace(old, new)
    ns = {'__name__': '__main__', '__file__': path}
    cwd = os.getcwd(); os.chdir(MODELING)
    try:
        exec(compile(src, path, 'exec'), ns)
    finally:
        os.chdir(cwd)
    return ns


#%% contEEG fit (mnt-correct only, AR-IRLS), same as the comparison script
fname = 'run_model_cont_EEG_city_mnt.py'
src = open(os.path.join(MODELING, fname)).read()
a = src.index('#%% find subjects with fNIRS and enough EEG epochs')
b = src.index('# check if any of subject in subj_id_array is in the excluded_subj')
ns = run_script(fname, [
    (src[a:b], f'subj_id_array = [{SUBJ_ID}]\n'),
    ('is_save = True', 'is_save = False'),
    ('fnirs_hp_cutoff = 0.01', f'fnirs_hp_cutoff = {CONT_HP}'),
    ("    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),\n", ''),
    ("    dm_all.common = dm_all.common.sel(chromo=[select_chromo])\n",
     f"    dm_all.common = dm_all.common.sel(chromo=[select_chromo])\n    Y_all = Y_all.sel(parcel={PARCELS!r})\n"),
], cut_at='    #%% spectral whitening of Y and X')

fs = ns['eeg_sfreq']
B = ns['bspline_basis']                 # (n_delay, 20)
d_t = ns['delay_times']                 # delay of each basis row (s)
names = [f'mnt_correct_bspline{i}' for i in range(B.shape[1])]
res = ns['results']
cov_all = res.sm.cov_params()

#%% 3. per-trial EEG weights
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']
w = []
for r_i, rk in enumerate(run_keys):
    sig = ns['eeg_reg_value_list'][r_i] - ns['eeg_offsets']['mnt_correct'][r_i]  # per-trial baseline-corrected Cz
    t = ns['eeg_sample_nirs_t'][r_i]
    stim = ns['all_stims'][ns['run_key_to_run_idx'][rk]]
    mc = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)]
    for on, dur in zip(mc['onset'].values, mc['duration'].values):
        idx = (t >= on) & (t < on + dur)
        w.append(sig[idx].sum() / fs * 1e6)  # uV*s
w = np.array(w)
print(f'per-trial EEG weight w (sum of Cz over the trial, uV*s): mean {w.mean():.3f}, SD {w.std():.3f}, '
      f'mean/SD {w.mean() / w.std():.2f}, negative {np.mean(w < 0) * 100:.0f}% of {len(w)} trials')

#%% 1-2. B-spline flexibility and fitted kernels
with open(cmp_path, 'rb') as f:
    cmp = pickle.load(f)['curves']
w_mean_Vsample = w.mean() * 1e-6 * fs  # mean trial weight in the kernel's units (V * EEG sample)
rows = []
fig, axs = plt.subplots(len(PARCELS) + 1, 3, figsize=(17, 4 * (len(PARCELS) + 1)), constrained_layout=True)

a = axs[0, 0]
for j in range(B.shape[1]):
    a.plot(d_t, B[:, j], lw=1)
a.set(xlabel='Delay (s)', ylabel='basis value', title=f'{B.shape[1]} cubic B-splines on [-2, 18) s (sum = 1 at every delay)')
a = axs[0, 1]
a.hist(w, bins=30, color='#2a6fdb')
a.axvline(0, color='k', lw=0.8); a.axvline(w.mean(), color='#d1495b', lw=2, label=f'mean {w.mean():.2f}')
a.set(xlabel='Cz summed over the trial (uV*s)', ylabel='mnt-correct trials',
      title=f'3. per-trial EEG weight w: mean/SD = {w.mean() / w.std():.2f}, {np.mean(w < 0) * 100:.0f}% negative')
a.legend(frameon=False)
axs[0, 2].axis('off')

for row, p in enumerate(PARCELS, start=1):
    c = cmp[p]
    # 1. can the B-splines represent the onlyStim HRF shape?
    h_stim = np.interp(d_t, c['onlyStim']['t'], c['onlyStim']['h'])
    coef = np.linalg.lstsq(B, h_stim, rcond=None)[0]
    fit = B @ coef
    r2 = 1 - np.sum((h_stim - fit) ** 2) / np.sum((h_stim - h_stim.mean()) ** 2)
    a = axs[row, 0]
    a.plot(d_t, h_stim, color='k', lw=2, label='onlyStim HRF (Gaussian kernels)')
    a.plot(d_t, fit, color='#2a6fdb', lw=1.5, ls='--', label=f'best fit with the 20 B-splines (R2 = {r2:.4f})')
    a.axhline(0, color='#ccc', lw=0.8)
    a.set(xlabel='Delay (s)', ylabel='HbO (uM)', title=f'{p}\n1. B-spline basis vs the onlyStim HRF')
    a.legend(frameon=False, fontsize=8)

    # 2. fitted kernel h(d) = B beta with 95% CI, t of each weight
    beta = res.sm.params.sel(parcel=p, chromo='HbO', regressor=names).values
    cov = cov_all.sel(parcel=p, chromo='HbO', regressor_r=names, regressor_c=names).values
    h = B @ beta
    se_h = np.sqrt(np.einsum('ij,jk,ik->i', B, cov, B))
    t_beta = beta / np.sqrt(np.diag(cov))
    a = axs[row, 1]
    a.fill_between(d_t, (h - 1.96 * se_h) * 1e6, (h + 1.96 * se_h) * 1e6, color='#2a6fdb', alpha=0.2, lw=0)
    a.plot(d_t, h * 1e6, color='#2a6fdb', lw=2, label='fitted kernel h(d) = B-splines x weights')
    h_need = h_stim / w_mean_Vsample  # uM per (V * sample): kernel that reproduces the onlyStim HRF for a mean-weight trial
    a.plot(d_t, h_need, color='k', lw=1.5, ls='--', label='kernel needed for the onlyStim HRF (/ mean w)')
    a.axhline(0, color='#ccc', lw=0.8)
    a.set(xlabel='Delay (s)', ylabel='uM per (V * EEG sample)',
          title=f'2. fitted kernel (95% CI); |t| of the 20 weights: max {np.abs(t_beta).max():.1f}, '
                f'median {np.median(np.abs(t_beta)):.1f}')
    a.legend(frameon=False, fontsize=8)

    # average-trial response vs response of a +1 SD trial
    resp_mean = h * w_mean_Vsample * 1e6
    resp_sd = h * (w.std() * 1e-6 * fs) * 1e6
    a = axs[row, 2]
    a.plot(c['onlyStim']['t'], c['onlyStim']['h'], color='k', lw=2, label='onlyStim HRF')
    a.plot(d_t, resp_mean, color='#2a6fdb', lw=2, label=f'contEEG, trial with mean w ({w.mean():.2f} uV*s)')
    a.plot(d_t, resp_sd, color='#2a6fdb', lw=1, ls=':', label=f'contEEG, trial with w = SD ({w.std():.2f} uV*s)')
    a.axhline(0, color='#ccc', lw=0.8)
    a.set(xlim=(-2, 18), xlabel='Time from onset (s)', ylabel='HbO (uM)',
          title='3. predicted response ~ w x h(d)')
    a.legend(frameon=False, fontsize=8)
    rows.append(dict(parcel=p, bspline_R2_onlyStim=round(r2, 4), max_abs_t_weight=round(np.abs(t_beta).max(), 2),
                     F_p=float(np.squeeze(ns['stats_dict']['f_test_full_noEEG_mnt_correct'].sel(parcel=p, chromo='HbO').item().pvalue)),
                     kernel_peak_fit=np.abs(h).max() * 1e6, kernel_peak_needed=np.abs(h_need).max()))
fig.suptitle(f'{subject}: why the contEEG mnt-correct HRF is flat')
fig.savefig(fig_path, dpi=120)
print(pd.DataFrame(rows).to_string(index=False))
print(f'saved {fig_path}')
