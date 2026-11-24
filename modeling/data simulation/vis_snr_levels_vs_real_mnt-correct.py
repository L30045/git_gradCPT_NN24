#%% Simulated mnt-correct HRFs at several SNR levels vs the real mnt-correct response
# Inputs: the simulations made by simulate_mnt-correct_HRF.py with SIM_AMP_MULT = 0.25, 0.5, 1, 2, 4 (same seed, so
# the trials, jitter and time scales are identical and only the amplitude differs; x2 = +6 dB SNR), and the real
# data of the same subject and parcel.
#
# Figure 1 (one panel per SNR level): trial average +/- 95% CI of the simulated mnt-correct trials (the series the
#   model sees, highpassed at l_cutoff, baseline -2-0 s), the mean injected HRF, and the real mnt-correct trial
#   average.
# Figure 2: left, the mean injected HRF of every level with the real mnt-correct trial average; right, the per-trial
#   SNR (peak amplitude / SD of the highpassed background, 0-18 s) of every level.
#%% load library
import os
import sys
import glob
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import t as t_dist
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(modeling_path)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from params_setting import *
from sim_utils import true_trial_hrfs

#%% settings
subject = 'sub-723'
select_parcel = 'DorsAttnB_PostC_2_RH'
select_chromo = 'HbO'
# simulations of one family that differ only in A (one per SNR level)
sim_glob = 'sim_mnt-correct_10pct-city_dgfit-end15_A*_cv0.25_slogsd0.15_jit3_seed0'
l_cutoff = 0.02
epoch_win = (-2.0, 18.0)
is_save = True

sim_root = os.path.join(project_path, 'derivatives', 'eeg', subject, 'simulation')
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'HRF_surf', subject,
                        'SNR_levels_' + sim_glob.replace('_A*', ''))
c_real, c_true = '#7b3fb5', '#1f1f1f'


#%% load
def epochs(run, onsets):
    """Highpassed, baseline-corrected epochs (trials, lag) of the selected parcel, uM."""
    t = run.time.values
    fs = 1 / np.median(np.diff(t))
    x = sosfiltfilt(butter(4, l_cutoff, btype='highpass', fs=fs, output='sos'),
                    run.sel(parcel=select_parcel, chromo=select_chromo).pint.to('micromolar').pint.dequantify().values)
    lags = np.arange(int(round(epoch_win[0] * fs)), int(round(epoch_win[1] * fs)) + 1)
    idx = np.array([np.argmin(np.abs(t - o)) for o in onsets], dtype=int)[:, None] + lags[None, :]
    idx = idx[(idx[:, 0] >= 0) & (idx[:, -1] < len(t))]
    ep = x[idx]
    return lags / fs, ep - ep[:, lags < 0].mean(1, keepdims=True)


def mean_ci(e):
    m = e.mean(0)
    h = t_dist.ppf(0.975, len(e) - 1) * e.std(0, ddof=1) / np.sqrt(len(e))
    return m, m - h, m + h


der = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data', subject)
with gzip.open(os.path.join(der, f'{subject}_preprocessed_results_{NOISE_MODEL}_v26.pkl'), 'rb') as f:
    real_stims = pickle.load(f)['stims']
with open(os.path.join(der, f'{subject}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}_v26.pkl'), 'rb') as f:
    real_runs = pickle.load(f)['parcel_ts']
real_ep = []
for st, run in zip(real_stims, real_runs):
    lag, e = epochs(run, st.loc[(st['trial_type'] == 'mnt') & (st['response_code'] == 0), 'onset'].values)
    real_ep.append(e)
real_ep = np.concatenate(real_ep)

levels = []
for d in glob.glob(os.path.join(sim_root, sim_glob)):
    truth = pd.read_csv(os.path.join(d, f'{subject}_sim_ground_truth.csv'))
    if 'amp_mult' not in truth.columns:
        continue
    with gzip.open(os.path.join(d, f'{subject}_sim_stims.pkl'), 'rb') as f:
        sim_stims = pickle.load(f)['stims']
    with open(os.path.join(d, f'{subject}_sim_parcel_ts.pkl'), 'rb') as f:
        sim_runs = pickle.load(f)['parcel_ts']
    ep = np.concatenate([epochs(run, st.loc[st['is_sim'], 'onset'].values)[1] for st, run in zip(sim_stims, sim_runs)])
    levels.append(dict(name=os.path.basename(d), mult=float(truth['amp_mult'].iloc[0]), truth=truth, ep=ep,
                       true_mean=true_trial_hrfs(truth, lag).mean(0), snr_db=np.median(truth['snr_db'])))
levels.sort(key=lambda L: L['mult'])
if not levels:
    raise FileNotFoundError(f"no simulations with amp_mult in {sim_root}/{sim_glob}")
print(f"{len(levels)} SNR levels, {len(real_ep)} real mnt-correct trials")
for L in levels:
    print(f"  x{L['mult']:g}: median SNR {L['snr_db']:.1f} dB, {len(L['ep'])} simulated trials  ({L['name']})")

cmap = plt.get_cmap('viridis')
lvl_col = [cmap(0.3 + 0.6 * i / max(len(levels) - 1, 1)) for i in range(len(levels))]


def style(ax):
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


#%% figure 1: trial average of each level vs the real mnt-correct response
rm, rlo, rhi = mean_ci(real_ep)
fig, axs = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.3), constrained_layout=True, squeeze=False)
for ax, L, col in zip(axs[0], levels, lvl_col):
    m, lo, hi = mean_ci(L['ep'])
    ax.fill_between(lag, rlo, rhi, color=c_real, alpha=0.15, lw=0)
    ax.plot(lag, rm, color=c_real, lw=1.5, label=f'real mnt-correct (n = {len(real_ep)})')
    ax.fill_between(lag, lo, hi, color=col, alpha=0.25, lw=0)
    ax.plot(lag, m, color=col, lw=2, label=f'simulated mnt-correct (n = {len(L["ep"])})')
    ax.plot(lag, L['true_mean'], color=c_true, ls='--', lw=1.5, label='mean injected HRF')
    ax.set(title=f'A x {L["mult"]:g}  (median SNR {L["snr_db"]:.1f} dB)', xlabel='Lag from onset (s)')
    style(ax)
axs[0, 0].set_ylabel(f'{select_chromo} (µM)')
axs[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
fig.suptitle(f'{subject} {select_parcel} {select_chromo}: trial average ± 95% CI by SNR level '
             f'(highpassed {l_cutoff} Hz, baseline −2–0 s); each panel has its own y-axis')
if is_save:
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f'{subject}_{select_parcel}_trial_avg_by_SNR.png')
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


#%% figure 2: injected HRFs of all levels with the real response, and the SNR distributions
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, gridspec_kw={'width_ratios': [1.4, 1]})
ax = axs[0]
ax.fill_between(lag, rlo, rhi, color=c_real, alpha=0.15, lw=0)
ax.plot(lag, rm, color=c_real, lw=1.5, label=f'real mnt-correct trial average ± 95% CI (n = {len(real_ep)})')
for L, col in zip(levels, lvl_col):
    ax.plot(lag, L['true_mean'], color=col, lw=2, label=f'injected, A x {L["mult"]:g} ({L["snr_db"]:.1f} dB)')
ax.set(xlabel='Lag from onset (s)', ylabel=f'{select_chromo} (µM)', title='Mean injected HRF by SNR level')
ax.legend(frameon=False, fontsize=8)
style(ax)

ax = axs[1]
data = [L['truth']['snr_db'].values for L in levels]
parts = ax.violinplot(data, positions=range(len(levels)), showmedians=True, widths=0.8)
for b, col in zip(parts['bodies'], lvl_col):
    b.set_facecolor(col)
    b.set_alpha(0.5)
for k in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
    parts[k].set_color('#555555')
    parts[k].set_linewidth(1)
for i, L in enumerate(levels):
    ax.annotate(f'{L["snr_db"]:.1f}', (i, L['snr_db']), xytext=(30, 6), textcoords='offset points',
                fontsize=8, va='center')
ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
ax.set_xticks(range(len(levels)), [f'x{L["mult"]:g}' for L in levels])
ax.set(xlabel='Amplitude factor', ylabel='Per-trial SNR (dB)',
       title='SNR = peak amplitude / background SD (0–18 s)')
ax.grid(alpha=0.25, lw=0.6, axis='y')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.suptitle(f'{subject} {select_parcel} {select_chromo}: simulated SNR levels vs the real mnt-correct response')
if is_save:
    out = os.path.join(plot_dir, f'{subject}_{select_parcel}_injected_HRF_and_SNR_by_level.png')
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
