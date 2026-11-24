#%% Compare Bayesian HRF fits across simulated SNR levels
# Inputs:
# - simulations from data simulation/simulate_mnt-correct_HRF.py with SIM_AMP_MULT = 0.25, 0.5, 1, 2, 4. Same seed,
#   so trials, jitter and time scales are identical and only the amplitude (hence SNR, 6 dB per x2) differs.
#   Per-trial ground truth: <sim dir>/<subject>_sim_ground_truth.csv.
# - posteriors from run_Bayes_image-space_gaussian-kernel_mnt-correct.py on each simulation, for the 4 variants
#   ridge / no ridge x design highpassed like y / not: <sim dir>/bayes/<subject>_<model_name>_<parcel>_idata.nc.
#
# For every (SNR level, variant): truth = mean injected HRF of the trials in the fitted runs (locked to the labeled
# onset, so the jitter is in it); estimate = posterior of B h in uM. Metrics:
#   peak ratio      posterior-mean peak / true peak (0-18 s), with the 95% interval of the per-draw peak
#   coverage        fraction of lags where the true mean is inside the 95% credible band
#   corr, RMSE      posterior mean vs true mean (RMSE also relative to the true peak)
#   latency         posterior median peak latency
#   R-hat, ESS      worst over the HRF lags; divergences
#
# Outputs (derivatives/eeg/run_Bayes_image-space_gaussian-kernel_mnt-correct/simulated data/<subject>/
#          SNR_comparison_<sim family>/):
#   <model base>_metrics_by_SNR.csv
#   <subject>_<parcel>_HRF_by_SNR.png       grid: SNR level (rows) x variant (columns), truth vs posterior
#   <subject>_<parcel>_metrics_by_SNR.png   metrics vs median per-trial SNR (dB), one line per variant
#%% load library
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import t as t_dist
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, modeling_path)
sys.path.insert(0, os.path.join(modeling_path, 'data simulation'))
from params_setting import *
from sim_utils import true_trial_hrfs

#%% settings (match the Bayesian runs)
subject = 'sub-723'
select_parcel = 'DorsAttnB_PostC_2_RH'
sim_glob = 'sim_mnt-correct_10pct-city_dgfit-end15_A*_cv0.25_slogsd0.15_jit3_seed0'   # one simulation per SNR level
run_keys = ['gradcpt1']
is_hp_fNIRS = True
is_smooth_penalty = False
peak_win = (0.0, 18.0)   # lag window (s) for the peak amplitude and latency
is_save = True

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
smooth_flag = 'smooth' if is_smooth_penalty else 'noSmooth'
model_base = f'Bayes-HRF-AR_mnt-correct_gauss_1-stage_{"-".join(run_keys)}_{hp_flag}_{smooth_flag}'
sim_root = os.path.join(project_path, 'derivatives', 'eeg', subject, 'simulation')
plot_dir = os.path.join(project_path, 'derivatives', 'eeg', 'run_Bayes_image-space_gaussian-kernel_mnt-correct',
                        'simulated data', subject, 'SNR_comparison_' + sim_glob.replace('_A*', ''))

# variants: (label, is_ridge, is_hp_design, color, linestyle). Color = prior, line style = design.
variants = [('ridge, hpDesign', True, True, '#2a78d6', '-'),
            ('ridge, no hpDesign', True, False, '#2a78d6', '--'),
            ('no ridge, hpDesign', False, True, '#1baa7d', '-'),
            ('no ridge, no hpDesign', False, False, '#1baa7d', '--')]
c_true = '#eb6834'


def idata_path(sim_dir, is_ridge, is_hp_design):
    name = model_base + ('' if is_ridge else '_noRidge') + ('_hpDesign' if is_hp_design and is_hp_fNIRS else '')
    return os.path.join(sim_dir, 'bayes', f'{subject}_{name}_{select_parcel}_idata.nc')


#%% load every (level, variant)
rows, curves = [], {}
for sim_dir in glob.glob(os.path.join(sim_root, sim_glob)):
    sim_name = os.path.basename(sim_dir)
    truth_all = pd.read_csv(os.path.join(sim_dir, f'{subject}_sim_ground_truth.csv'))
    if 'amp_mult' not in truth_all.columns:
        print(f"skip {sim_name}: no amp_mult in the ground truth (made before the SNR levels)")
        continue
    mult = float(truth_all['amp_mult'].iloc[0])
    for label, is_ridge, is_hpd, _, _ in variants:
        p = idata_path(sim_dir, is_ridge, is_hpd)
        if not os.path.exists(p):
            print(f"missing: x{mult:g} {label}")
            continue
        idata = az.from_netcdf(p)
        cd = idata.constant_data
        if cd.attrs.get('sim_name') != sim_name or cd.attrs.get('parcel') != select_parcel:
            raise ValueError(f"{p} is for sim '{cd.attrs.get('sim_name')}', parcel '{cd.attrs.get('parcel')}'.")
        hrf_t = cd['hrf_t'].values
        y_scale = float(cd['y_scale'])
        hrf_post = idata.posterior['hrf'].stack(s=('chain', 'draw')).values.T * y_scale   # (draws, lag), uM
        est_mean = hrf_post.mean(0)
        est_lo, est_hi = np.percentile(hrf_post, [2.5, 97.5], 0)

        truth = truth_all[truth_all['run_idx'].isin(cd['run_idx'].values)]
        true_hrfs = true_trial_hrfs(truth, hrf_t)
        true_mean = true_hrfs.mean(0)
        h_ci = t_dist.ppf(0.975, len(true_hrfs) - 1) * true_hrfs.std(0, ddof=1) / np.sqrt(len(true_hrfs))

        in_pk = (hrf_t >= peak_win[0]) & (hrf_t <= peak_win[1])
        true_pk = true_mean[in_pk].max()
        pk_draws = hrf_post[:, in_pk].max(1)
        lat_draws = hrf_t[in_pk][hrf_post[:, in_pk].argmax(1)]
        rmse = np.sqrt(np.mean((est_mean - true_mean) ** 2))
        rhat = az.rhat(idata, var_names=['hrf'])['hrf'].values
        ess = az.ess(idata, var_names=['hrf'])['hrf'].values
        rows.append({
            'amp_mult': mult, 'sim_name': sim_name, 'variant': label, 'is_ridge': is_ridge, 'is_hp_design': is_hpd,
            'n_trials': len(true_hrfs),
            'median_snr_db': float(np.median(truth['snr_db'])),
            'true_peak_uM': true_pk,
            'est_peak_uM': est_mean[in_pk].max(),
            'peak_ratio': est_mean[in_pk].max() / true_pk,
            'peak_ratio_lo': np.percentile(pk_draws, 2.5) / true_pk,
            'peak_ratio_hi': np.percentile(pk_draws, 97.5) / true_pk,
            'coverage': np.mean((true_mean >= est_lo) & (true_mean <= est_hi)),
            'corr': np.corrcoef(est_mean, true_mean)[0, 1],
            'rmse_uM': rmse,
            'rmse_rel': rmse / true_pk,
            'true_latency_s': hrf_t[in_pk][true_mean[in_pk].argmax()],
            'est_latency_s': float(np.median(lat_draws)),
            'rhat_max': float(np.nanmax(rhat)),
            'ess_bulk_min': float(np.nanmin(ess)),
            'divergences': int(idata.sample_stats['diverging'].sum()),
        })
        curves[(mult, label)] = dict(t=hrf_t, true=true_mean, true_lo=true_mean - h_ci, true_hi=true_mean + h_ci,
                                     est=est_mean, lo=est_lo, hi=est_hi)

if not rows:
    raise FileNotFoundError(f"no fitted SNR levels in {sim_root}/{sim_glob}")
df = pd.DataFrame(rows).sort_values(['amp_mult', 'is_ridge', 'is_hp_design'], ascending=[True, False, False])
mults = sorted(df['amp_mult'].unique())
snr_of = df.groupby('amp_mult')['median_snr_db'].first()
pd.set_option('display.width', 220)
print(df[['amp_mult', 'median_snr_db', 'variant', 'peak_ratio', 'peak_ratio_lo', 'peak_ratio_hi', 'coverage',
          'corr', 'rmse_rel', 'est_latency_s', 'rhat_max', 'ess_bulk_min', 'divergences']].round(3).to_string(index=False))
if is_save:
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f'{model_base}_metrics_by_SNR.csv')
    df.to_csv(out, index=False)
    print(f"saved {out}")


def style(ax):
    ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


#%% figure 1: HRF grid, SNR level (rows) x variant (columns)
fig, axs = plt.subplots(len(mults), len(variants), figsize=(4.2 * len(variants), 2.9 * len(mults)),
                        constrained_layout=True, squeeze=False, sharex=True, sharey='row')
for i, m in enumerate(mults):
    for j, (label, _, _, col, _) in enumerate(variants):
        ax = axs[i, j]
        c = curves.get((m, label))
        if c is None:
            ax.set_axis_off()
            continue
        r = df[(df['amp_mult'] == m) & (df['variant'] == label)].iloc[0]
        ax.fill_between(c['t'], c['true_lo'], c['true_hi'], color=c_true, alpha=0.25, lw=0)
        ax.plot(c['t'], c['true'], color=c_true, lw=2, label='truth: trial mean, 95% CI')
        ax.fill_between(c['t'], c['lo'], c['hi'], color=col, alpha=0.2, lw=0)
        ax.plot(c['t'], c['est'], color=col, lw=2, label='posterior mean, 95% CrI')
        ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
        ax.set_title(f'peak {r.peak_ratio:.0%} of truth, coverage {r.coverage:.2f}', fontsize=9)
        style(ax)
        if i == 0:
            ax.annotate(label, (0.5, 1.25), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f'A x {m:g} ({snr_of[m]:.1f} dB)\nHbO (µM)')
        if i == len(mults) - 1:
            ax.set_xlabel('Lag (s)')
axs[0, 0].legend(frameon=False, fontsize=7, loc='lower left')
fig.suptitle(f'{subject} {select_parcel}: Bayesian HRF vs ground truth by SNR (median per-trial SNR in dB; '
             f'{int(df["n_trials"].iloc[0])} trials, {"-".join(run_keys)}); rows share the y-axis')
if is_save:
    out = os.path.join(plot_dir, f'{subject}_{select_parcel}_HRF_by_SNR.png')
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


#%% figure 2: metrics vs SNR
panels = [('peak_ratio', 'Posterior peak / true peak', 1.0),
          ('coverage', 'Coverage of the true mean (95% CrI)', 0.95),
          ('corr', 'Correlation with the true mean', 1.0),
          ('rmse_rel', 'RMSE / true peak', 0.0)]
fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), constrained_layout=True)
dx = np.linspace(-0.45, 0.45, len(variants))   # small offset so the error bars do not overlap
for ax, (key, ylabel, ref) in zip(axs, panels):
    for k, (label, _, _, col, ls) in enumerate(variants):
        d = df[df['variant'] == label].sort_values('median_snr_db')
        if d.empty:
            continue
        x = d['median_snr_db'].values + dx[k]
        if key == 'peak_ratio':
            ax.errorbar(x, d[key], yerr=[d[key] - d['peak_ratio_lo'], d['peak_ratio_hi'] - d[key]], color=col,
                        ls=ls, lw=1.8, marker='o', ms=5, capsize=2, elinewidth=0.8, label=label)
        else:
            ax.plot(x, d[key], color=col, ls=ls, lw=1.8, marker='o', ms=5, label=label)
    ax.axhline(ref, color='#888888', lw=1, ls=':', zorder=0)
    ax.set(xlabel='Median per-trial SNR (dB)', ylabel=ylabel)
    ax.set_xticks(snr_of.values, [f'{v:.0f}' for v in snr_of.values])
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title('bars: 95% interval of the per-draw peak', fontsize=9)
fig.suptitle(f'{subject} {select_parcel}: Bayesian HRF recovery vs SNR (blue = ridge prior, green = no ridge; '
             f'solid = design highpassed like y, dashed = not)')
if is_save:
    out = os.path.join(plot_dir, f'{subject}_{select_parcel}_metrics_by_SNR.png')
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
