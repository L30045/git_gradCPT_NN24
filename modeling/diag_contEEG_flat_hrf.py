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
# A second figure shows one example mnt-correct trial: the fitted fNIRS Y, the Cz EEG and the delay-0 EEG
# regressor (masked, baseline-corrected Cz, low-passed and read at the fNIRS samples like the model's
# regressors, with a one-hot basis at delay 0 instead of the B-splines), and the B-spline regressors (actual DM
# columns) next to the same splines convolved with the rectified Cz (no sign cancellation).
# Needs the curves saved by cmp_hrf_gaussian-kernel_vs_cont-EEG_mnt-correct.py. Saves PNGs in this folder.
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
fig2_path = os.path.join(MODELING, f'diag_contEEG_flat_hrf_{subject}_example_trial.png')
EXAMPLE = None          # (run_key, onset in fNIRS time); None: the trial whose |w| is closest to the median |w|
SPLINE_CENTERS = [0, 2.5, 5, 7.5, 10]  # s, B-spline regressors to show (nearest centres)


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
trial_list = []  # (run index, onset, duration) of each w
for r_i, rk in enumerate(run_keys):
    sig = ns['eeg_reg_value_list'][r_i]
    t = ns['eeg_sample_nirs_t'][r_i]
    stim = ns['all_stims'][ns['run_key_to_run_idx'][rk]]
    mc = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)]
    for on, dur in zip(mc['onset'].values, mc['duration'].values):
        w0, w1 = ns['eeg_trial_window'] if ns['eeg_trial_window'] is not None else (0.0, dur)
        idx = (t >= on + w0) & (t < on + w1)
        bl = (t >= on + ns['eeg_baseline'][0]) & (t < on + ns['eeg_baseline'][1]) if ns['eeg_baseline'] is not None else None
        off = sig[bl].mean() if bl is not None and bl.any() else 0.0
        w.append((sig[idx] - off).sum() / fs * 1e6)  # uV*s, EEG over the trial window, own baseline
        trial_list.append((r_i, on, dur))
w = np.array(w)
print(f'per-trial EEG weight w (sum of Cz over the trial window {ns["eeg_trial_window"]}, uV*s): mean {w.mean():.3f}, SD {w.std():.3f}, '
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

#%% example trial: fNIRS, delay-0 EEG regressor and B-spline regressors
model = ns['model']
fnirs_sfreq = ns['fnirs_sfreq']
t_pre_c = ns['t_pre']
win = (-2.0, 18.0)
if EXAMPLE is None:
    cand = [i for i, (r_i, on, dur) in enumerate(trial_list)
            if on + win[0] >= ns['fnirs_sample_t'][r_i][0] and on + win[1] <= ns['fnirs_sample_t'][r_i][-1]]
    ex = cand[np.argmin(np.abs(np.abs(w[cand]) - np.median(np.abs(w))))]
else:
    ex = [i for i, (r_i, on, _) in enumerate(trial_list)
          if run_keys[r_i] == EXAMPLE[0] and np.isclose(on, EXAMPLE[1], atol=0.05)][0]
r_i, on0, dur0 = trial_list[ex]
print(f'example trial: {run_keys[r_i]}, onset {on0:.2f} s (fNIRS time), w = {w[ex]:.2f} uV*s')

t_eeg = ns['eeg_sample_nirs_t'][r_i]
cz_raw = ns['eeg_reg_value_list'][r_i]
mask = ns['eeg_masks']['mnt_correct'][r_i]
cz_reg = (cz_raw - ns['eeg_offsets']['mnt_correct'][r_i]) * mask   # what the model convolves (all mnt-correct trials)
t_f = ns['fnirs_sample_t'][r_i]
seg_start = int(sum(len(x) for x in ns['fnirs_sample_t'][:r_i]))
seg = slice(seg_start, seg_start + len(t_f))
in_f = (t_f >= on0 + win[0]) & (t_f <= on0 + win[1])
in_e = (t_eeg >= on0 + win[0]) & (t_eeg <= on0 + win[1])

# delay-0 regressor: one-hot basis at delay 0, same lowpass / sampling as the B-spline regressors
k0 = int(round(t_pre_c * fs))
onehot = np.zeros((B.shape[0], 1)); onehot[k0, 0] = 1.0
x_d0 = model.sample_cont_EEG_bspline_regressor(cz_reg, t_eeg, onehot, t_pre_c, t_f, fnirs_sfreq / 2)[:, 0]
# B-spline regressors: actual DM columns, and the same splines on the rectified masked Cz
centers = d_t[B.argmax(0)]
sel = sorted({int(np.argmin(np.abs(centers - c))) for c in SPLINE_CENTERS})
X_dm = ns['dm_all'].common.sel(chromo='HbO').transpose('time', 'regressor').values[seg]  # V * EEG sample
X_rect = model.sample_cont_EEG_bspline_regressor(np.abs(cz_reg), t_eeg, B[:, sel], t_pre_c, t_f, fnirs_sfreq / 2)
to_uVs = 1e6 / fs  # V * EEG sample -> uV * s

onsets = [on for (ri, on, _) in trial_list if ri == r_i and win[0] <= on - on0 <= win[1]]
fig2 = plt.figure(figsize=(16, 13), constrained_layout=True)
gs = fig2.add_gridspec(4, 3)
ax_y = fig2.add_subplot(gs[0, :2])
ax_e = fig2.add_subplot(gs[1, :2], sharex=ax_y)
ax_b = fig2.add_subplot(gs[2, :2], sharex=ax_y)
ax_r = fig2.add_subplot(gs[3, :2], sharex=ax_y)
ax_z = fig2.add_subplot(gs[1, 2])
ax_t = fig2.add_subplot(gs[2:, 2])
for ax in [ax_y, ax_e, ax_b, ax_r]:
    for on in onsets:
        ax.axvline(on - on0, color='#d1495b' if on == on0 else '#e8a0aa', lw=1.5 if on == on0 else 0.8, ls='-' if on == on0 else '--')
    ax.axhline(0, color='#cccccc', lw=0.8)
for p_i, p in enumerate(PARCELS):
    y = ns['Y_all'].sel(parcel=p, chromo='HbO').pint.dequantify().values[seg] * 1e6
    ax_y.plot(t_f[in_f] - on0, y[in_f], color=f'C{p_i}', lw=1.2, label=p)
ax_y.set(ylabel='HbO (uM)', title=f'fNIRS Y (drift / GSR removed) around the example trial ({run_keys[r_i]}, onset {on0:.1f} s; '
                                  f'red: this and other mnt-correct onsets)')
ax_y.legend(frameon=False, fontsize=8)

ax_e.plot(t_eeg[in_e] - on0, cz_raw[in_e] * 1e6, color='#bbbbbb', lw=0.6, label='raw Cz')
ax_e.plot(t_eeg[in_e] - on0, cz_reg[in_e] * 1e6, color='k', lw=0.9, label='Cz used by the model (in mnt-correct trials, baseline-corrected)')
ax_e.plot(t_f[in_f] - on0, x_d0[in_f] * 1e6, 'o-', color='#2a6fdb', ms=2.5, lw=1,
          label='delay-0 EEG regressor (low-passed, at fNIRS samples)')
ax_e.set(ylabel='Cz (uV)', title='EEG and the delay-0 regressor')
ax_e.legend(frameon=False, fontsize=8, loc='upper right')

for k, j in enumerate(sel):
    ax_b.plot(t_f[in_f] - on0, X_dm[in_f, j] * to_uVs, color=f'C{k}', lw=1.8, label=f'B-spline {j} (centre {centers[j]:.1f} s)')
ax_b.set(ylabel='uV * s', title='B-spline EEG regressors (DM columns): Cz in mnt-correct trials convolved with each B-spline')
ax_b.legend(frameon=False, fontsize=8, ncol=len(sel), loc='upper right')
for k, j in enumerate(sel):
    ax_r.plot(t_f[in_f] - on0, X_rect[in_f, k] * to_uVs, color=f'C{k}', lw=1.8, ls='--', label=f'B-spline {j}, rectified Cz')
ax_r.set(xlim=win, xlabel='Time from the example onset (s)', ylabel='uV * s',
         title='Same B-splines convolved with |Cz| (no sign cancellation): the size the regressors would have without cancellation')
ax_r.legend(frameon=False, fontsize=8, ncol=len(sel), loc='upper right')

zw = (t_eeg >= on0 - 0.5) & (t_eeg <= on0 + 1.5)
ax_z.axvspan(-0.2, 0, color='#f4a261', alpha=0.3, lw=0, label='baseline')
ax_z.axvspan(0, dur0, color='#2a9d8f', alpha=0.15, lw=0, label='trial (kept)')
ax_z.plot(t_eeg[zw] - on0, cz_raw[zw] * 1e6, color='#999999', lw=0.8, label='raw Cz')
ax_z.plot(t_eeg[zw] - on0, cz_reg[zw] * 1e6, color='k', lw=1.2, label='used by the model')
ax_z.axhline(0, color='#cccccc', lw=0.8)
ax_z.set(xlabel='Time from onset (s)', ylabel='Cz (uV)', title=f'example trial: w = {w[ex]:.2f} uV*s')
ax_z.legend(frameon=False, fontsize=7)

# summary: size of the regressors over all samples of all runs
X_all = ns['dm_all'].common.sel(chromo='HbO').transpose('time', 'regressor').values * to_uVs
X_rect_all, X_d0_all = [], []
for rr in range(len(run_keys)):
    czr = (ns['eeg_reg_value_list'][rr] - ns['eeg_offsets']['mnt_correct'][rr]) * ns['eeg_masks']['mnt_correct'][rr]
    X_rect_all.append(model.sample_cont_EEG_bspline_regressor(np.abs(czr), ns['eeg_sample_nirs_t'][rr], B,
                                                              t_pre_c, ns['fnirs_sample_t'][rr], fnirs_sfreq / 2))
    X_d0_all.append(model.sample_cont_EEG_bspline_regressor(czr, ns['eeg_sample_nirs_t'][rr], onehot, t_pre_c,
                                                            ns['fnirs_sample_t'][rr], fnirs_sfreq / 2)[:, 0])
X_rect_all = np.vstack(X_rect_all) * to_uVs
X_d0_all = np.concatenate(X_d0_all) * 1e6
ratio = X_all.std(0) / X_rect_all.std(0)
ax_t.bar(np.arange(B.shape[1]), ratio, color='#2a6fdb')
ax_t.set(xlabel='B-spline index', ylabel='SD(regressor) / SD(rectified-Cz regressor)',
         title=f'all runs: B-spline regressor SD relative to\nthe no-cancellation version (median {np.median(ratio):.2f})')
ax_t.grid(alpha=0.3, axis='y')
fig2.suptitle(f'{subject}: example mnt-correct trial, fNIRS vs delay-0 EEG regressor vs B-spline EEG regressors')
fig2.savefig(fig2_path, dpi=110)
print(f'delay-0 regressor SD {X_d0_all.std():.3f} uV; B-spline regressors SD median {np.median(X_all.std(0)):.4f} uV*s, '
      f'rectified-Cz version {np.median(X_rect_all.std(0)):.4f} uV*s (ratio median {np.median(ratio):.2f})')
print(f'saved {fig2_path}')
