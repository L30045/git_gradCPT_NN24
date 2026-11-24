#%% Group-average HRF from the per-subject fits of run_Bayes_image-space_gaussian-kernel_mnt-correct.py
# For each subject, load the saved posterior (InferenceData), check convergence of the HRF weights h and their
# prior scale tau_h, and take the posterior mean HRF in uM (hrf * y_scale). Subjects whose sampler did not
# converge (max R-hat above rhat_max) are left out. The group HRF is the mean of the subjects' posterior means,
# with a t-based 95% CI across subjects (between-subject spread; the within-subject posterior widths are
# not propagated).
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import t as t_dist
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from params_setting import project_path

#%% settings
select_parcel = 'DorsAttnB_PostC_2_RH'
model_name = 'Bayes-HRF-AR_mnt-correct_gauss_1-stage_gradcpt1_Hp_noSmooth_hpDesign'
rhat_max = 1.2       # subjects with any R-hat (h, tau_h) above this are excluded as not converged
peak_win = (0.0, 18.0)
is_save = True

der = os.path.join(project_path, 'derivatives', 'eeg')
out_dir = os.path.join(der, 'run_Bayes_image-space_gaussian-kernel_mnt-correct', 'group', model_name)

#%% load per-subject posteriors
paths = sorted(glob.glob(os.path.join(der, 'sub-*', 'bayes', f'sub-*_{model_name}_{select_parcel}_idata.nc')))
rows, hrfs = [], {}
hrf_t = None
for p in paths:
    idata = az.from_netcdf(p)
    subject = idata.constant_data.attrs['subject']
    if idata.constant_data.attrs.get('sim_name'):
        continue
    s = az.summary(idata, var_names=['h', 'tau_h'])
    y_scale = float(idata.constant_data['y_scale'])
    t_i = idata.constant_data['hrf_t'].values
    if hrf_t is None:
        hrf_t = t_i
    elif not np.allclose(hrf_t, t_i):
        raise ValueError(f"{subject}: lag axis differs from the first subject.")
    hrf_post = idata.posterior['hrf'].stack(s=('chain', 'draw')).values.T * y_scale   # (draws, lag), uM
    hrfs[subject] = hrf_post.mean(0)
    rows.append({'subject': subject, 'max_rhat': s['r_hat'].max(), 'min_ess_bulk': s['ess_bulk'].min(),
                 'divergences': int(idata.sample_stats['diverging'].sum())})
qc = pd.DataFrame(rows).set_index('subject')
qc['included'] = qc['max_rhat'] <= rhat_max
print(qc.round(3).to_string())

keep = qc.index[qc['included']].tolist()
excluded = qc.index[~qc['included']].tolist()
H = np.stack([hrfs[s] for s in keep])           # (subjects, lag)
n = len(keep)
g_mean = H.mean(0)
g_ci = t_dist.ppf(0.975, n - 1) * H.std(0, ddof=1) / np.sqrt(n)

in_pk = (hrf_t >= peak_win[0]) & (hrf_t <= peak_win[1])
i_pk = np.flatnonzero(in_pk)[g_mean[in_pk].argmax()]
print(f"\n{n} subjects included, excluded: {excluded}")
print(f"group peak {g_mean[i_pk]:.4f} uM (95% CI {g_mean[i_pk] - g_ci[i_pk]:.4f} to "
      f"{g_mean[i_pk] + g_ci[i_pk]:.4f}) at {hrf_t[i_pk]:.1f} s")

#%% figure
c_grp, c_sub = '#2a78d6', '#b4b4b4'
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for k, s in enumerate(keep):
    ax.plot(hrf_t, hrfs[s], color=c_sub, lw=0.8, alpha=0.8, label='Single subject (posterior mean)' if k == 0 else None)
ax.fill_between(hrf_t, g_mean - g_ci, g_mean + g_ci, color=c_grp, alpha=0.2, lw=0)
ax.plot(hrf_t, g_mean, color=c_grp, lw=2.2, label=f'Group mean, 95% CI (n = {n})')
ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
ax.grid(alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set(xlabel='Lag from mnt-correct onset (s)', ylabel='HbO (µM)',
       title=f'{select_parcel} HbO, gradcpt1: group-average Bayesian HRF')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.text(0.01, 0.01, f"Excluded (R-hat > {rhat_max}): {', '.join(excluded) if excluded else 'none'}",
        transform=ax.transAxes, fontsize=8, color='#666666', va='bottom')

if is_save:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'group_{select_parcel}_HRF.png')
    fig.savefig(path, dpi=150)
    qc.to_csv(os.path.join(out_dir, f'group_{select_parcel}_subject_qc.csv'))
    print(f"saved {path}")
plt.show()
