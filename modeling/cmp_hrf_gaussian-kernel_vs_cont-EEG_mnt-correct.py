#%% Compare mnt-correct HRFs from three design matrices (AR-IRLS, one subject, selected parcels)
#   1. onlyStim  run_model_EEG_inform_parcel_based.py: 20 Gaussian kernels (-2 to 18 s), every mnt-correct
#                trial with amplitude 1
#   2. onlyEEG   run_model_EEG_inform_parcel_based.py: the same kernels, each trial scaled by its Cz N2-P3
#                peak-to-peak / run mean (DM from build_dm_gaussian-kernel_mnt-correct_EEG-informed.py)
#   3. contEEG   run_model_cont_EEG_city_mnt.py: continuous Cz EEG, zeroed outside the mnt-correct trials
#                (onset -> onset + duration; all other periods, incl. mnt-incorrect and city trials, masked),
#                delayed over [-2, 18) s and projected onto 20 B-splines. Run here with mnt-correct only
#                (the script's mnt-incorrect set is dropped), the same 0.02 Hz fNIRS highpass as the shared
#                Y_all, and only its AR-IRLS fit (fixAR-IRLS / AR-iRRR skipped).
#                Within each trial, Cz is baseline-corrected (mean over -0.2-0 s removed, eeg_baseline).
# All three are fit with cedalion glm.fit(noise_model='ar_irls') on Y that went through the same 3-stage
# preprocessing (highpass, OLS drift, OLS GSR over all sensitive parcels). Models 1-2 use the shared Y_all of
# run_model_cont_EEG_fNIRS.py (window: first onset -> last onset + 15 s); model 3 builds its own Y (window:
# first onset - 2 s -> last onset + 18 s), so its Y differs slightly at the run edges.
#
# The HRFs are in different units, so each model is converted to the response it predicts for an average
# mnt-correct trial (uM):
#   onlyStim: B h_stim
#   onlyEEG:  B h_eeg * mean scale over mnt-correct trials (~1)
#   contEEG:  the delay kernel h(d) (per EEG sample, per V) convolved with the average within-trial Cz EEG
#             (each trial's Cz from onset to onset + duration, averaged over mnt-correct trials)
# Each curve is a linear map of the model's HRF betas, curve = M beta (M: time x regressor; M = B for the
# Gaussian kernels, M = G = average Cz EEG convolved with each B-spline for contEEG). All three are
# baseline-corrected the same way, by subtracting the mean over t < 0 from every row of M:
#   M_bc = M - mean_{t<0} M,   curve = M_bc beta,   var = diag(M_bc Sigma M_bc^T)
# with Sigma the AR-IRLS covariance of the HRF betas, so the baseline subtraction is included in the variance.
# CIs are pointwise, curve +- 1.96 sqrt(var). The block average of Y around the mnt-correct onsets (shared
# Y_all, not a model; overlapping trials included; baseline-corrected the same way) is shown for reference.
#
# 4. Bayes      Bayesian model/run_Bayes_image-space_gaussian-kernel_mnt-correct.py, run here as it is (current
#                settings: run 1 (gradcpt1) only, no smoothness penalty, 0.02 Hz highpass) for each parcel with
#                is_save = False: joint HRF + drift/GSR + AR(P) Student-t model, PyMC NUTS, 20 Gaussian
#                kernels (-2 to 18 s), mnt-correct trials, no 3-stage preprocessing. Its posterior 'hrf' (B h,
#                standardized units) times the script's data std ('scale') gives uM. The draws are cached in
#                <out_dir>/<subject>_Bayes_hrf_<parcel>.pkl (delete to refit). The older saved posteriors in
#                <derivatives>/eeg/<subject>/bayes/ (2026-11-02, previous model version) are not used: their
#                'hrf' is not on the current script's scale. Each draw is baseline-corrected (mean over t < 0
#                removed); the band is the 2.5-97.5% posterior interval of the corrected draws.
#
# Nothing is saved by the two model scripts (is_save = False). Outputs: a PNG in this folder and a pickle
# with the curves in <derivatives>/eeg/<subject>/cmp_HRF_DM/.
#%% load library
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

MODELING = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODELING)
from params_setting import *

#%% settings
SUBJ_ID = 723
PARCELS = ['DorsAttnB_PostC_2_RH', 'SomMotA_25_LH']
HRF_WIN = (-2.0, 18.0)   # s, block-average window
CONT_HP = 0.02           # fNIRS highpass for the cont EEG model (shared Y_all uses 0.02 Hz)
BAYES_SCRIPT = os.path.join('Bayesian model', 'run_Bayes_image-space_gaussian-kernel_mnt-correct.py')
RUN_BAYES = True         # fit the Bayes model if no cached posterior exists (slow: NUTS, 4 chains)
subject = f'sub-{SUBJ_ID}'
out_dir = os.path.join(project_path, 'derivatives', 'eeg', subject, 'cmp_HRF_DM')
fig_path = os.path.join(MODELING, f'cmp_HRF_DM_{subject}_mnt-correct.png')


def run_script(fname, replacements, cut_at=None):
    """exec a model script with string replacements (each must match exactly once) and return its namespace"""
    path = os.path.join(MODELING, fname)
    src = open(path).read()
    if cut_at is not None:
        assert src.count(cut_at) == 1, cut_at
        src = src[:src.index(cut_at)]
    for old, new in replacements:
        assert src.count(old) == 1, f'{fname}: {old[:80]!r} found {src.count(old)} times'
        src = src.replace(old, new)
    ns = {'__name__': '__main__', '__file__': path}
    cwd = os.getcwd()
    os.chdir(MODELING)
    try:
        exec(compile(src, path, 'exec'), ns)
    finally:
        os.chdir(cwd)
    return ns


def single_subject(fname):
    src = open(os.path.join(MODELING, fname)).read()
    a = src.index('#%% find subjects with fNIRS and enough EEG epochs')
    b = src.index('# check if any of subject in subj_id_array is in the excluded_subj')
    return (src[a:b], f'subj_id_array = [{SUBJ_ID}]\n')


#%% 1-2. Gaussian-kernel models (run_model_EEG_inform_parcel_based.py)
gauss = {}
for mt in ['onlyStim', 'onlyEEG']:
    print(f'===== Gaussian kernels, {mt} =====')
    ns = run_script('run_model_EEG_inform_parcel_based.py', [
        single_subject('run_model_EEG_inform_parcel_based.py'),
        ("model_type = 'onlyStim'", f"model_type = '{mt}'"),
        ('is_overwrite = False', 'is_overwrite = True'),
        ('is_save = True', 'is_save = False'),
        ("    Y_all = Y_all.sel(chromo=[select_chromo])\n",
         f"    Y_all = Y_all.sel(chromo=[select_chromo]).sel(parcel={PARCELS!r})\n"),
    ])
    gauss[mt] = dict(betas_dict=ns['betas_dict'], stats=ns['stats_dict'])
Y_shared = ns['Y_all']
with open(get_gauss_eeg_dm_path(os.path.join(project_path, 'derivatives', 'eeg', subject)), 'rb') as f:
    dm_saved = pickle.load(f)
mean_scale = np.concatenate([v['scale'] for v in dm_saved['trial_info'].values()]).mean()
print(f'mean EEG scale over mnt-correct trials: {mean_scale:.3f} ({dm_saved["eeg_amp_feature"]})')

#%% 3. continuous EEG model, mnt-correct only (run_model_cont_EEG_city_mnt.py)
print('===== continuous Cz EEG, mnt-correct only =====')
ns = run_script('run_model_cont_EEG_city_mnt.py', [
    single_subject('run_model_cont_EEG_city_mnt.py'),
    ('is_save = True', 'is_save = False'),
    ('fnirs_hp_cutoff = 0.01', f'fnirs_hp_cutoff = {CONT_HP}'),
    ("    'mnt_incorrect': lambda df: (df['trial_type'] == 'mnt') & (df['response_code'] != 0),\n", ''),
    ("    dm_all.common = dm_all.common.sel(chromo=[select_chromo])\n",
     f"    dm_all.common = dm_all.common.sel(chromo=[select_chromo])\n    Y_all = Y_all.sel(parcel={PARCELS!r})\n"),
], cut_at='    #%% spectral whitening of Y and X')
assert ns['trial_type_prefixes'] == ['mnt_correct']

# masked EEG check: the regressor EEG is nonzero only inside mnt-correct trials
fs_eeg = ns['eeg_sfreq']
t_pre_c = ns['t_pre']
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']
segs = []
for r_i, rk in enumerate(run_keys):
    sig = ns['eeg_reg_value_list'][r_i] - ns['eeg_offsets']['mnt_correct'][r_i]  # per-trial baseline-corrected Cz
    mask = ns['eeg_masks']['mnt_correct'][r_i]
    t = ns['eeg_sample_nirs_t'][r_i]
    stim = ns['all_stims'][ns['run_key_to_run_idx'][rk]]
    mc = stim[(stim['trial_type'] == 'mnt') & (stim['response_code'] == 0)]
    print(f'{rk}: {len(mc)} mnt-correct trials, EEG kept in {mask.mean() * 100:.1f}% of the cropped samples')
    for on, dur in zip(mc['onset'].values, mc['duration'].values):
        idx = np.where((t >= on) & (t < on + dur))[0]
        if len(idx):
            segs.append(sig[idx])
n_seg = max(len(s) for s in segs)
eeg_avg = np.zeros(n_seg)  # average within-trial Cz (V), trials of different length zero-padded
for s in segs:
    eeg_avg[:len(s)] += s
eeg_avg /= len(segs)
print(f'average within-trial Cz over {len(segs)} trials: mean {eeg_avg.mean() * 1e6:.2f} uV, '
      f'duration {n_seg / fs_eeg:.2f} s')

B = ns['bspline_basis']                       # (n_delay, n_bspline)
G = np.column_stack([np.convolve(eeg_avg, B[:, j]) for j in range(B.shape[1])])  # avg-trial response basis
tau = -t_pre_c + np.arange(G.shape[0]) / fs_eeg
step = max(1, int(round(fs_eeg / 50)))      # plot at ~50 Hz
G, tau = G[::step], tau[::step]
reg_names = [f'mnt_correct_bspline{i}' for i in range(B.shape[1])]
cont_res = ns['results']
cont_cov = cont_res.sm.cov_params()
delay_t = ns['delay_times']

#%% observed block average around the mnt-correct onsets (shared Y_all)
samples = Y_shared.samples.values
dt = np.median(np.diff(Y_shared.time.values))
bounds = np.concatenate([[0], np.where(np.diff(samples) <= 0)[0] + 1, [len(samples)]])
t_ep = np.arange(HRF_WIN[0], HRF_WIN[1] + dt / 2, dt)


def block_average(parcel):
    y = Y_shared.sel(parcel=parcel, chromo='HbO').pint.dequantify().values * 1e6
    eps = []
    for r_i, rk in enumerate(run_keys):
        seg = slice(bounds[r_i], bounds[r_i + 1])
        t_orig = samples[seg] * dt  # fNIRS run time (0 at the run start)
        ev = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                      f'{subject}_task-gradCPT_run-{rk[-1]:0>2}_events.tsv'), sep='\t')
        for on in ev[(ev['trial_type'] == 'mnt') & (ev['response_code'] == 0)]['onset'].values:
            if on + HRF_WIN[0] >= t_orig[0] and on + HRF_WIN[1] <= t_orig[-1]:
                eps.append(np.interp(on + t_ep, t_orig, y[seg]))
    eps = np.array(eps)
    eps = eps - eps[:, t_ep < 0].mean(1, keepdims=True)  # same baseline as the model curves
    return eps.mean(0), eps.std(0) / np.sqrt(len(eps)), len(eps)


#%% collect curves (uM) and stats
def pval(test):
    v = test.sel(parcel=p, chromo='HbO') if hasattr(test, 'sel') else test
    v = v.item() if hasattr(v, 'item') else v
    return float(np.squeeze(v.pvalue))


def baseline_curve(M, t, beta, cov, scale=1.0):
    """curve = scale * M_bc beta and its pointwise SE, with M_bc = M - mean of M over t < 0 (uM)"""
    M_bc = (M - M[t < 0].mean(0, keepdims=True)) * scale
    h = M_bc @ beta
    se = np.sqrt(np.einsum('ij,jk,ik->i', M_bc, cov, M_bc))
    return h * 1e6, se * 1e6


out = {}
for p in PARCELS:
    d = {}
    for mt, scale, ftest in [('onlyStim', 1.0, 'f_test_stim_basis'), ('onlyEEG', mean_scale, 'f_test_eeg_basis')]:
        bd = gauss[mt]['betas_dict']
        t_h = bd['hrf_estimate'].time.values
        names = [r for r in bd['betas'].regressor.values if str(r).startswith('HRF mnt-correct-')]
        beta = bd['betas'].sel(parcel=p, chromo='HbO', regressor=names).values
        cov = bd['cov_params'].sel(parcel=p, chromo='HbO', regressor_r=names, regressor_c=names).values
        M = bd['basis_da'].transpose('time', 'component').values  # kernels in regressor order
        assert M.shape == (len(t_h), len(names))
        h, se = baseline_curve(M, t_h, beta, cov, scale)
        # the parcel script's hrf_estimate is baseline-corrected the same way: check
        h_ref = bd['hrf_estimate'].sel(trial_type='mnt-correct', parcel=p, chromo='HbO').pint.dequantify().values * 1e6 * scale
        assert np.allclose(h, h_ref, atol=1e-9 + 1e-6 * np.abs(h_ref).max()), f'{mt} {p}: HRF mismatch'
        d[mt] = dict(t=t_h, h=h, se=se, p=pval(gauss[mt]['stats'][ftest]))
    beta = cont_res.sm.params.sel(parcel=p, chromo='HbO', regressor=reg_names).values
    cov = cont_cov.sel(parcel=p, chromo='HbO', regressor_r=reg_names, regressor_c=reg_names).values
    r, se = baseline_curve(G, tau, beta, cov)
    kern = ns['betas_eeg_per_type']['mnt_correct'].sel(parcel=p, chromo='HbO').values
    d['contEEG'] = dict(t=tau, h=r, se=se, kernel_t=delay_t, kernel=kern,
                        p=pval(ns['stats_dict']['f_test_full_noEEG_mnt_correct']))
    ba, ba_se, n_ep = block_average(p)
    d['block_average'] = dict(t=t_ep, h=ba, se=ba_se, n=n_ep)
    out[p] = d


#%% 4. Bayesian HRF (saved posterior, run 1)
def bayes_curve(parcel):
    """posterior HRF in uM from the current Bayes script (cached), baseline-corrected per draw"""
    cache = os.path.join(out_dir, f'{subject}_Bayes_hrf_{parcel}.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            bres = pickle.load(f)
    elif RUN_BAYES:
        print(f'===== Bayes model, {parcel} =====')
        bns = run_script(BAYES_SCRIPT, [
            ("subject = 'sub-723'", f"subject = '{subject}'"),
            ("select_parcel = 'DorsAttnB_PostC_2_RH'", f"select_parcel = '{parcel}'"),
            ('is_save = True ', 'is_save = False'),
        ])
        post = bns['idata'].posterior
        bres = dict(t=bns['hrf_t'], draws=post['hrf'].stack(s=('chain', 'draw')).values.T * bns['scale'],
                    scale=bns['scale'], run_keys=bns['run_keys'], P=bns['P'],
                    divergences=int(bns['idata'].sample_stats['diverging'].sum()),
                    is_smooth_penalty=bns['is_smooth_penalty'])
        os.makedirs(out_dir, exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(bres, f)
    else:
        print(f'{parcel}: no cached Bayes posterior and RUN_BAYES = False')
        return None
    t_b, draws = bres['t'], bres['draws']
    draws = draws - draws[:, t_b < 0].mean(1, keepdims=True)
    lo, hi = np.percentile(draws, [2.5, 97.5], 0)
    print(f"{parcel}: Bayes posterior {draws.shape[0]} draws, runs {bres['run_keys']}, AR({bres['P']}), "
          f"{bres['divergences']} divergences, data std {bres['scale']:.4g} uM")
    return dict(t=t_b, h=draws.mean(0), lo=lo, hi=hi, p=None)


for p in PARCELS:
    for k in ['onlyStim', 'onlyEEG', 'contEEG']:
        c = out[p][k]
        c['lo'], c['hi'] = c['h'] - 1.96 * c['se'], c['h'] + 1.96 * c['se']
    b = bayes_curve(p)
    if b is not None:
        out[p]['Bayes'] = b
MODELS = ['onlyStim', 'onlyEEG', 'contEEG', 'Bayes']

print('\nmnt-correct HRF (average trial), peak in 2-10 s and F-test of the HRF regressors:')
for p in PARCELS:
    for k in [m for m in MODELS if m in out[p]]:
        c = out[p][k]
        w = (c['t'] >= 2) & (c['t'] <= 10)
        i = np.argmax(c['h'][w])
        ptxt = f'F-test p = {c["p"]:.3g}' if c['p'] is not None else 'posterior mean'
        print(f'  {p:22s} {k:9s} peak {c["h"][w][i]:+.4f} uM at {c["t"][w][i]:5.2f} s, {ptxt}')

#%% figure
colors = {'onlyStim': 'k', 'onlyEEG': '#d1495b', 'contEEG': '#2a6fdb', 'Bayes': '#2a9d8f'}
labels = {'onlyStim': 'GLM AR-IRLS | Gaussian kernels, unmodulated (onlyStim)',
          'onlyEEG': f'GLM AR-IRLS | Gaussian kernels x N2-P3 ptp (onlyEEG), x{mean_scale:.2f}',
          'contEEG': 'GLM AR-IRLS | continuous Cz EEG in mnt-correct trials (contEEG)',
          'Bayes': 'Bayesian NUTS, joint drift/GSR + AR(P) Student-t | Gaussian kernels, run 1 only'}
band = {'onlyStim': '95% CI', 'onlyEEG': '95% CI', 'contEEG': '95% CI', 'Bayes': '95% cred. int.'}
fig, axs = plt.subplots(len(PARCELS), 2, figsize=(16, 5 * len(PARCELS)), constrained_layout=True, squeeze=False)
for row, p in enumerate(PARCELS):
    a = axs[row, 0]
    ba = out[p]['block_average']
    a.plot(ba['t'], ba['h'], color='#999999', lw=1, label=f'block average of Y (n = {ba["n"]})')
    for k in [m for m in MODELS if m in out[p]]:
        c = out[p][k]
        a.fill_between(c['t'], c['lo'], c['hi'], color=colors[k], alpha=0.12, lw=0)
        stat = f'F p = {c["p"]:.2g}' if c['p'] is not None else 'posterior mean'
        a.plot(c['t'], c['h'], color=colors[k], lw=2, label=f'{labels[k]} ({stat}; band {band[k]})')
    a.axhline(0, color='#cccccc', lw=0.8); a.axvline(0, color='#cccccc', lw=0.8)
    a.set(xlim=HRF_WIN, xlabel='Time from mnt-correct onset (s)', ylabel='HbO (uM)',
          title=f'{p}: predicted response to an average mnt-correct trial\n(baseline t < 0 removed, pointwise bands)')
    a.plot([], [], color=colors['contEEG'], lw=1, ls=':', label='GLM AR-IRLS | contEEG delay kernel h(d), per unit EEG (right panel only)')
    a.legend(fontsize=7, frameon=False, loc='upper right')
    a.grid(alpha=0.3)

    a = axs[row, 1]
    for k in [m for m in MODELS if m in out[p]]:
        c = out[p][k]
        w = (c['t'] >= 0) & (c['t'] <= 12)
        a.plot(c['t'], c['h'] / np.abs(c['h'][w]).max(), color=colors[k], lw=2, label=labels[k])
    c = out[p]['contEEG']
    w = (c['kernel_t'] >= 0) & (c['kernel_t'] <= 12)
    a.plot(c['kernel_t'], c['kernel'] / np.abs(c['kernel'][w]).max(), color=colors['contEEG'], lw=1, ls=':',
           label='GLM AR-IRLS | contEEG delay kernel h(d), per unit EEG')
    a.axhline(0, color='#cccccc', lw=0.8); a.axvline(0, color='#cccccc', lw=0.8)
    a.set(xlim=HRF_WIN, xlabel='Time / delay (s)', ylabel='normalized to max |.| in 0-12 s', title=f'{p}: shape')
    a.grid(alpha=0.3)
fig.suptitle(f'{subject} HbO, mnt-correct HRF: GLM (AR-IRLS) with three design matrices vs Bayesian model')
fig.savefig(fig_path, dpi=130)
print(f'saved {fig_path}')

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f'{subject}_cmp_HRF_DM_mnt-correct.pkl'), 'wb') as f:
    pickle.dump(dict(curves=out, parcels=PARCELS, mean_scale=mean_scale, eeg_avg=eeg_avg, fs_eeg=fs_eeg,
                     cont_hp=CONT_HP), f)
print(f'saved {os.path.join(out_dir, f"{subject}_cmp_HRF_DM_mnt-correct.pkl")}')
