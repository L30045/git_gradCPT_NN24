# %% [markdown]
# # EEG features in vs out of the zone: our data vs the literature table
#
# Checks each row of the survey table "EEG features: in the zone vs. out of the zone" (slide 13 of
# *EEG & Pupil Features of Sustained Attention*) against gradCPT_NN24.
#
# **Zone label.** Esterman et al. (2013): within each run, trials with `VTC_smoothed` above the run median are
# *out of the zone*, the rest *in the zone*.
#
# **Per row** (the literature direction is for out of the zone relative to in the zone):
#
# | Row | Literature | Our measure | Trials |
# |---|---|---|---|
# | Frontal-midline theta, transition-locked | lower out (gradCPT) | `fz_theta_trial` (4-8 Hz Morlet, 0-0.8 s); Fz theta time course | city_correct |
# | Theta variability across the run | high-variability people have a bigger d' drop (between-subject) | subject SD of `fz_theta_trial` vs d'(in) - d'(out); within-subject `fz_theta_trial_sd9` | city_correct / all |
# | Posterior alpha power | higher out | `oz_alpha_win`, `pz_alpha_win` (10 s centred window) | all |
# | P3 amplitude (no-go) | smaller out | `pz_P3_meanamp`, `cz_P3_meanamp` (0.25-0.80 s); Pz ERP | clean mnt_correct |
# | P1 / N1 amplitude | smaller out | `oz_P1`, `oz_N1` (80-130 / 140-200 ms); Oz ERP | city_correct |
# | CPP build-up rate | slower and more variable out | `pz_cpp_slope` (mean and within-zone SD); response-locked Pz | city_correct |
# | Trial-to-trial response variability | higher out | theta ITPC at Oz / Pz over all trials of a zone (lower ITPC = more temporal jitter); proxy | city_correct |
# | Responses to background distractors | smaller out (fMRI) | not testable: needs a distractor manipulation | - |
#
# **Statistics.** Each feature is z-scored within subject; the per-subject zone means are compared with a paired
# t-test across subjects (Cohen's dz = mean difference / SD of differences). The zone split is not corrected for
# time-on-task, as in the literature; the time-on-task-partialled correlation with `VTC_smoothed` from
# `EEG_extra_features_vs_VTC.py` / `EEG_features_vs_behavior.py` is listed next to it.
#
# Inputs: the two feature tables (`extract_EEG_trial_features.py`, `extract_extra_trial_features.py`) and the
# preprocessed .fif files (waveforms, cached in `OUT_DIR/cache`). Outputs: `<data_save_path>/xSubj_results/EEG_zone_table/`.

# %% [markdown]
# ## 1. Setup

# %%
# %matplotlib inline
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

code_dir = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eeg/preproc_pipe'
sys.path.insert(0, code_dir)
from utils import data_save_path
from extract_EEG_trial_features import get_features_path, get_eeg_fif_path, load_events, RUN_KEYS, CH_NAMES
from extract_extra_trial_features import get_extra_features_path, THETA_FREQS, REJECT_PTP_UV

warnings.filterwarnings('ignore', category=RuntimeWarning)
OUT_DIR = os.path.join(data_save_path, 'xSubj_results', 'EEG_zone_table')
CACHE = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE, exist_ok=True)
MIN_TRIALS_ZONE = 10
ITPC_WIN = (0.0, 0.4)

INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
C_IN, C_OUT = '#2a78d6', '#eb6834'          # categorical slots 1 and 2
AGREE, OPPOSITE, NULL, NA = '#1baf7a', '#e34948', '#b4b3ae', '#e6e5e0'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999999',
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'font.size': 9,
                     'axes.titlesize': 9.5, 'axes.titlecolor': INK, 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'savefig.facecolor': BG})

# %% [markdown]
# ## 2. Trial table with zone labels

# %%
MAIN_COLS = ['run', 'trial_idx', 'trial_type', 'response_code', 'reaction_time', 'VTC', 'VTC_smoothed',
             'pz_P3_meanamp', 'cz_P3_meanamp', 'pz_erp_ok', 'cz_erp_ok']


def load_all():
    dfs = []
    for p in sorted(glob.glob(get_extra_features_path('sub-*'))):
        subj = os.path.basename(os.path.dirname(os.path.dirname(p)))
        x = pd.read_csv(p)
        b = pd.read_csv(get_features_path(subj), usecols=MAIN_COLS)
        dfs.append(x.merge(b, on=['run', 'trial_idx', 'trial_type', 'response_code'], validate='1:1'))
    df = pd.concat(dfs, ignore_index=True)
    df['condition'] = np.select([(df.trial_type == 'city') & (df.response_code == 1), df.trial_type == 'city',
                                 (df.trial_type == 'mnt') & (df.response_code == 0)],
                                ['city_correct', 'city_incorrect', 'mnt_correct'], 'mnt_incorrect')
    med = df.groupby(['subject', 'run']).VTC_smoothed.transform('median')
    df['zone'] = np.where(df.VTC_smoothed > med, 'out', 'in')
    return df


df = load_all()
print(f"{df.subject.nunique()} subjects, {len(df):,} trials")
df.groupby('zone').condition.value_counts().unstack()

# %% [markdown]
# ## 3. Zone contrasts of the scalar features

# %%
# (row, feature, trials, expected sign of out - in, label)
ROWS = [
    ('Frontal-midline theta', 'fz_theta_trial', 'city_correct', -1, 'Fz stim-locked theta'),
    ('Frontal-midline theta', 'cz_theta_trial', 'city_correct', -1, 'Cz stim-locked theta'),
    ('Frontal-midline theta', 'fz_theta_win', 'all', 0, 'Fz ongoing theta (10 s)'),
    ('Theta variability', 'fz_theta_trial_sd9', 'all', +1, 'Fz theta trial SD (within subject)'),
    ('Posterior alpha', 'oz_alpha_win', 'all', +1, 'Oz alpha power (10 s)'),
    ('Posterior alpha', 'pz_alpha_win', 'all', +1, 'Pz alpha power (10 s)'),
    ('P3 (no-go)', 'pz_P3_meanamp', 'mnt_correct', -1, 'Pz P3 mean amplitude'),
    ('P3 (no-go)', 'cz_P3_meanamp', 'mnt_correct', -1, 'Cz P3 mean amplitude'),
    ('P1 / N1', 'oz_P1', 'city_correct', -1, 'Oz P1 (80-130 ms)'),
    ('P1 / N1', 'oz_N1', 'city_correct', +1, 'Oz N1 (140-200 ms; less negative = smaller)'),
    ('CPP build-up', 'pz_cpp_slope', 'city_correct', -1, 'Pz CPP slope'),
    ('CPP build-up', 'pz_cpp_slope@sd', 'city_correct', +1, 'Pz CPP slope SD across trials'),
]


def zone_values(df, feat, trials):
    """Per-subject in / out value of a feature, z-scored within subject."""
    base, how = (feat.split('@') + ['mean'])[:2]
    out = []
    for subj, g in df.groupby('subject'):
        m = np.ones(len(g), bool) if trials == 'all' else (g.condition == trials).values
        if base.endswith('P3_meanamp'):
            m &= (g[f"{base.split('_')[0]}_erp_ok"] == True).values
        x = g[base].values.astype(float)
        m &= np.isfinite(x)
        if m.sum() < 2 * MIN_TRIALS_ZONE or np.nanstd(x[m]) == 0:
            continue
        z = (x - x[m].mean()) / x[m].std()
        vals = {}
        for zone in ['in', 'out']:
            k = m & (g.zone == zone).values
            if k.sum() < MIN_TRIALS_ZONE:
                break
            vals[zone] = z[k].std(ddof=1) if how == 'sd' else z[k].mean()
        if len(vals) == 2:
            out.append(dict(subject=subj, **vals))
    return pd.DataFrame(out)


def paired(v):
    d = v['out'] - v['in']
    t, p = stats.ttest_1samp(d, 0)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(n=len(d), diff=d.mean(), lo=d.mean() - half, hi=d.mean() + half, dz=d.mean() / d.std(ddof=1),
                t=t, p=p, n_pos=int((d > 0).sum()))


def verdict(r, expected):
    if expected == 0:
        return 'contrast'
    if r['p'] >= 0.05:
        return 'no effect'
    return 'agree' if np.sign(r['diff']) == expected else 'opposite'


zone_vals, res = {}, []
for row, feat, trials, exp_sign, label in ROWS:
    v = zone_values(df, feat, trials)
    zone_vals[feat] = v
    r = paired(v)
    r.update(row=row, feature=feat, label=label, trials=trials, expected=exp_sign)
    r['verdict'] = verdict(r, exp_sign)
    res.append(r)
res = pd.DataFrame(res)

# time-on-task-partialled r with VTC_smoothed from the earlier analyses
g_extra = pd.read_csv(os.path.join(data_save_path, 'xSubj_results', 'EEG_extra_features_vs_VTC', 'group_stats.csv'))
g_main = pd.read_csv(os.path.join(data_save_path, 'xSubj_results', 'EEG_features_vs_behavior', 'group_stats.csv'))
g_extra = g_extra[(g_extra.behaviour == 'VTC_smoothed') & (g_extra.stat == 'partial')].set_index('feature')
g_main = g_main[(g_main.behaviour == 'VTC_smoothed') & (g_main.r_type == 'partial')].set_index('feature')
for i, f in enumerate(res.feature):
    src = g_extra if f in g_extra.index else g_main if f in g_main.index else None
    if src is not None:
        res.loc[i, 'r_partial'], res.loc[i, 'p_partial'] = src.loc[f, 'mean'], src.loc[f, 'p']
res.to_csv(os.path.join(OUT_DIR, 'zone_contrasts.csv'), index=False)
res[['row', 'label', 'n', 'n_pos', 'diff', 'dz', 't', 'p', 'expected', 'verdict', 'r_partial', 'p_partial']].round(3)

# %% [markdown]
# ### 3.1 Theta variability, between subjects (the form Solís-Vivanco et al. tested)
# Subject theta variability = SD of `fz_theta_trial` (city_correct) after removing run means and a linear
# time-on-task trend. d' per zone from hits (city) and false alarms (mountain commissions), log-linear corrected.

# %%
def dprime(g):
    city, mnt = g[g.trial_type == 'city'], g[g.trial_type == 'mnt']
    h = ((city.response_code == 1).sum() + 0.5) / (len(city) + 1)
    fa = ((mnt.response_code != 0).sum() + 0.5) / (len(mnt) + 1)
    return stats.norm.ppf(h) - stats.norm.ppf(fa)


def detrended_sd(g, feat):
    x = g[feat].values.astype(float)
    res_ = np.full(len(x), np.nan)
    for u in g.run.unique():
        m = (g.run == u).values & np.isfinite(x)
        if m.sum() < 10:
            continue
        A = np.column_stack([np.ones(m.sum()), g.trial_idx.values[m]])
        res_[m] = x[m] - A @ np.linalg.lstsq(A, x[m], rcond=None)[0]
    return np.nanstd(res_, ddof=1)


between = []
for subj, g in df.groupby('subject'):
    cc = g[g.condition == 'city_correct']
    if cc.fz_theta_trial.notna().sum() < 50:
        continue
    between.append(dict(subject=subj, theta_sd=detrended_sd(cc, 'fz_theta_trial'),
                        d_in=dprime(g[g.zone == 'in']), d_out=dprime(g[g.zone == 'out']),
                        rt_cv=cc.reaction_time.std() / cc.reaction_time.mean()))
between = pd.DataFrame(between)
between['d_drop'] = between.d_in - between.d_out
r_drop, p_drop = stats.spearmanr(between.theta_sd, between.d_drop)
r_cv, p_cv = stats.spearmanr(between.theta_sd, between.rt_cv)
print(f"d' in {between.d_in.mean():.2f} vs out {between.d_out.mean():.2f} "
      f"(paired t p = {stats.ttest_rel(between.d_in, between.d_out)[1]:.4f}, n = {len(between)})")
print(f"theta SD vs d' drop: Spearman r = {r_drop:.2f}, p = {p_drop:.3f}")
print(f"theta SD vs RT CV:   Spearman r = {r_cv:.2f}, p = {p_cv:.3f}")

# %% [markdown]
# ## 4. Waveforms per zone (cached per subject)
# * Fz theta power time course (4-8 Hz Morlet, log10), city_correct, -0.2 to 0.8 s.
# * Oz stimulus-locked ERP, city_correct, baseline (-0.2, 0), 20 Hz low-pass.
# * Pz stimulus-locked ERP, clean mnt_correct, baseline (-0.2, 0), 10 Hz low-pass.
# * Pz response-locked CPP, city_correct, baseline = pre-onset 200 ms, 10 Hz low-pass.
# * Theta ITPC at Oz / Pz over all city_correct trials of a zone (trial count matched by random subsampling).
# Epochs with peak-to-peak > 100 uV on the channel are dropped.

# %%
T_STIM = np.arange(-0.2, 0.8, 0.004)
T_P3 = np.arange(-0.2, 1.2, 0.004)
T_RESP = np.arange(-0.6, 0.2, 0.004)


def subject_waveforms(subject, zone_tbl):
    path = os.path.join(CACHE, f'{subject}_zone_waveforms.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    acc = {k: {'in': [], 'out': []} for k in ['theta_fz', 'erp_oz', 'erp_pz_mnt', 'cpp_pz']}
    phases = {ch: {'in': [], 'out': []} for ch in ['oz', 'pz']}
    for run_key in RUN_KEYS:
        run_id = int(run_key[-1])
        fif = get_eeg_fif_path(subject, run_id)
        if not os.path.exists(fif):
            continue
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        sf = raw.info['sfreq']
        ev = load_events(subject, run_id)
        zt = zone_tbl[zone_tbl.run == run_id].set_index('trial_idx')
        chs = [c for c in CH_NAMES if c in raw.ch_names]
        data = {c: raw.get_data(picks=[c])[0] * 1e6 for c in chs}
        n = len(next(iter(data.values())))
        lp = {hz: {c: sosfiltfilt(butter(4, hz, btype='low', fs=sf, output='sos'), data[c]) for c in chs}
              for hz in (10.0, 20.0)}
        W = {c: mne.time_frequency.tfr_array_morlet(data[c][None, None], sf, THETA_FREQS, n_cycles=THETA_FREQS / 2,
                                                    output='complex', verbose=False)[0, 0] for c in chs}
        tpow = np.log10((np.abs(W['fz']) ** 2).mean(0)) if 'fz' in chs else None
        on = ev.onset.values.astype(float)

        def seg(x, t0, tt):
            idx = np.round((t0 + tt) * sf).astype(int)
            return x[idx] if idx[0] >= 0 and idx[-1] < n else None

        def clean(c, a, b):
            i0, i1 = int(round(a * sf)), int(round(b * sf))
            if i0 < 0 or i1 > n:
                return False
            s = data[c][i0:i1]
            return s.max() - s.min() <= REJECT_PTP_UV
        it = np.arange(ITPC_WIN[0], ITPC_WIN[1] + 1e-9, 0.01)
        for i in range(len(ev)):
            if i not in zt.index:
                continue
            cond, zone = zt.loc[i, 'condition'], zt.loc[i, 'zone']
            if cond == 'city_correct':
                if 'fz' in chs and clean('fz', on[i] - 0.2, on[i] + 0.8):
                    p = seg(tpow, on[i], T_STIM)
                    if p is not None:
                        acc['theta_fz'][zone].append(p)
                if 'oz' in chs and clean('oz', on[i] - 0.2, on[i] + 0.8):
                    e = seg(lp[20.0]['oz'], on[i], T_STIM)
                    if e is not None:
                        acc['erp_oz'][zone].append(e - e[T_STIM < 0].mean())
                for ch in ['oz', 'pz']:
                    if ch in chs and clean(ch, on[i] - 0.2, on[i] + 0.8):
                        z = W[ch][:, np.round((on[i] + it) * sf).astype(int)]
                        phases[ch][zone].append(z / np.abs(z))
                rt = zt.loc[i, 'reaction_time']
                if 'pz' in chs and np.isfinite(rt) and clean('pz', on[i] - 0.2, on[i] + rt + 0.2):
                    e = seg(lp[10.0]['pz'], on[i] + rt, T_RESP)
                    b = seg(lp[10.0]['pz'], on[i], np.arange(-0.2, 0, 0.004))
                    if e is not None and b is not None:
                        acc['cpp_pz'][zone].append(e - b.mean())
            elif cond == 'mnt_correct' and 'pz' in chs and clean('pz', on[i] - 0.2, on[i] + 1.2):
                e = seg(lp[10.0]['pz'], on[i], T_P3)
                if e is not None:
                    acc['erp_pz_mnt'][zone].append(e - e[T_P3 < 0].mean())
    out = {}
    for k, d in acc.items():
        for zone, lst in d.items():
            out[f'{k}_{zone}'] = np.mean(lst, 0) if len(lst) >= MIN_TRIALS_ZONE else np.array([np.nan])
            out[f'{k}_{zone}_n'] = np.array(len(lst))
    rng = np.random.default_rng(0)
    for ch, d in phases.items():
        nmin = min(len(d['in']), len(d['out']))
        for zone, lst in d.items():
            if nmin < 30:
                out[f'itpc_{ch}_{zone}'] = np.array(np.nan)
                continue
            P = np.array(lst)[rng.choice(len(lst), nmin, replace=False)]
            out[f'itpc_{ch}_{zone}'] = np.array(np.abs(P.mean(0)).mean())
    np.savez(path, **out)
    return out


wave = {}
for subj, g in df.groupby('subject'):
    wave[subj] = subject_waveforms(subj, g[['run', 'trial_idx', 'condition', 'zone', 'reaction_time']])
print(f"waveforms for {len(wave)} subjects")

# ITPC zone contrast (trial-count matched), added to the result table
for ch in ['oz', 'pz']:
    v = pd.DataFrame([dict(subject=s, **{z: float(w[f'itpc_{ch}_{z}']) for z in ['in', 'out']}) for s, w in wave.items()]).dropna()
    zone_vals[f'itpc_{ch}'] = v
    r = paired(v)
    r.update(row='Response variability', feature=f'itpc_{ch}', label=f'{ch.capitalize()} theta ITPC (lower = more jitter)',
             trials='city_correct', expected=-1)
    r['verdict'] = verdict(r, -1)
    res = pd.concat([res, pd.DataFrame([r])], ignore_index=True)
# Benjamini-Hochberg FDR across the directional contrasts (the 'contrast' row is excluded)
d_ = res.expected != 0
ps = res.loc[d_, 'p'].values
o = np.argsort(ps)
qs = np.minimum.accumulate((ps[o] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
res['q'] = np.nan
res.loc[res.index[d_][o], 'q'] = np.clip(qs, 0, 1)
res.to_csv(os.path.join(OUT_DIR, 'zone_contrasts.csv'), index=False)
print(f"{(res.q < 0.05).sum()} of {d_.sum()} directional contrasts have FDR q < 0.05")
res[['row', 'label', 'dz', 'p', 'q', 'verdict']].round(3)

# %% [markdown]
# ## 5. Figures
# Verdicts use uncorrected p < 0.05 ("nominal"); the q column gives the FDR-corrected value.

# %%
def grand(key):
    A = np.array([w[f'{key}_in'] for w in wave.values() if w[f'{key}_in'].size > 1 and w[f'{key}_out'].size > 1])
    B = np.array([w[f'{key}_out'] for w in wave.values() if w[f'{key}_in'].size > 1 and w[f'{key}_out'].size > 1])
    return A, B


def wave_panel(ax, t, key, title, ylabel, shade=None, sig_t=True):
    A, B = grand(key)
    for M, c, lab in [(A, C_IN, 'in the zone'), (B, C_OUT, 'out of the zone')]:
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.18, lw=0)
        ax.plot(t, mu, color=c, lw=2, label=lab)
    if shade:
        for a, b, lab in shade:
            ax.axvspan(a, b, color='#999999', alpha=0.12, lw=0)
            ax.text((a + b) / 2, 0.98, lab, transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=7.5, color=INK2)
    if sig_t:  # paired t per time point, uncorrected p < .05 marked under the curves
        p = stats.ttest_rel(B, A, axis=0)[1]
        y = ax.get_ylim()[0]
        ax.plot(t[p < 0.05], np.full((p < 0.05).sum(), y), 's', ms=2.5, color=INK2, label='p < .05 (uncorrected)')
    ax.axvline(0, color='#999999', lw=0.8)
    ax.set_title(f"{title}  (n = {len(A)})", loc='left')
    ax.set_ylabel(ylabel)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
wave_panel(axes[0, 0], T_STIM, 'theta_fz', 'Fz theta power, city_correct', 'log10 power (uV²)',
           shade=[(0.0, 0.4, 'Solís-Vivanco ~400 ms')])
wave_panel(axes[0, 1], T_STIM, 'erp_oz', 'Oz stimulus-locked ERP, city_correct', 'uV',
           shade=[(0.08, 0.13, 'P1'), (0.14, 0.20, 'N1')])
wave_panel(axes[1, 0], T_P3, 'erp_pz_mnt', 'Pz stimulus-locked ERP, mnt_correct', 'uV', shade=[(0.25, 0.80, 'P3 window')])
wave_panel(axes[1, 1], T_RESP, 'cpp_pz', 'Pz response-locked (CPP), city_correct', 'uV',
           shade=[(-0.25, -0.05, 'slope window')])
for ax in axes[0]:
    ax.set_xlabel('time from image transition onset (s)')
axes[1, 0].set_xlabel('time from image transition onset (s)')
axes[1, 1].set_xlabel('time from button press (s)')
axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Grand-average waveforms in vs out of the zone (mean ± SEM across subjects; zone = run-median split of '
             'smoothed VTC)', x=0.01, ha='left', color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'zone_waveforms.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
def paired_panel(ax, v, r, title):
    for _, s in v.iterrows():
        ax.plot([0, 1], [s['in'], s['out']], color='#c9c8c3', lw=1, zorder=1)
    ax.scatter(np.zeros(len(v)), v['in'], s=22, color=C_IN, edgecolor=BG, lw=1, zorder=3)
    ax.scatter(np.ones(len(v)), v['out'], s=22, color=C_OUT, edgecolor=BG, lw=1, zorder=3)
    ax.plot([0, 1], [v['in'].mean(), v['out'].mean()], color=INK, lw=2.2, marker='o', ms=7, mec=BG, mew=2, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['in', 'out'])
    ax.set_xlim(-0.35, 1.35)
    ax.grid(axis='x', visible=False)
    exp = {1: 'expected ↑', -1: 'expected ↓', 0: 'contrast'}[r['expected']]
    col = {'agree': AGREE, 'opposite': OPPOSITE}.get(r['verdict'], INK2)
    qtxt = f", q = {r['q']:.2f}" if np.isfinite(r.get('q', np.nan)) else ''
    ax.set_title(f"{title}\n{exp} · dz = {r['dz']:+.2f}, p = {r['p']:.3f}{qtxt}", loc='left', fontsize=8.5)
    ax.text(0.02, 0.02, r['verdict'], transform=ax.transAxes, fontsize=8, color=col, fontweight='bold')


panels = [f for f in res.feature]
ncol = 5
nrow = int(np.ceil((len(panels) + 1) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow), squeeze=False)
for ax, f in zip(axes.ravel(), panels):
    r = res[res.feature == f].iloc[0]
    paired_panel(ax, zone_vals[f], r, r['label'])
    ax.set_ylabel('z within subject' if not f.startswith('itpc') else 'ITPC', fontsize=8)
ax = axes.ravel()[len(panels)]
ax.scatter(between.theta_sd, between.d_drop, s=26, color=C_OUT, edgecolor=BG, lw=1)
ax.set_xlabel('Fz theta SD across run (log10)', fontsize=8)
ax.set_ylabel("d' drop (in − out)", fontsize=8)
ax.set_title(f"Theta variability, between subjects\nexpected r > 0 · r = {r_drop:+.2f}, p = {p_drop:.3f}",
             loc='left', fontsize=8.5)
for ax in axes.ravel()[len(panels) + 1:]:
    ax.set_visible(False)
fig.legend([Line2D([], [], ls='none', marker='o', color=C_IN), Line2D([], [], ls='none', marker='o', color=C_OUT),
            Line2D([], [], color=INK, lw=2, marker='o')], ['in the zone (subject)', 'out of the zone (subject)', 'group mean'],
           frameon=False, loc='upper right', ncol=3, fontsize=8)
fig.suptitle('Scalar features in vs out of the zone (per-subject means, paired across subjects)',
             x=0.01, ha='left', color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT_DIR, 'zone_paired.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# agreement summary: dz with 95% CI, oriented so that > 0 = the direction the literature reports
s = res[res.expected != 0].copy()
s['dz_lit'] = s.dz * s.expected
s['ci_lo'] = s.lo / (s['diff'] / s.dz) * s.expected
s['ci_hi'] = s.hi / (s['diff'] / s.dz) * s.expected
s[['ci_lo', 'ci_hi']] = np.sort(s[['ci_lo', 'ci_hi']].values, 1)
s = s.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.6))
y = np.arange(len(s))
cols = [{'agree': AGREE, 'opposite': OPPOSITE}.get(v, NULL) for v in s.verdict]
ax.hlines(y, s.ci_lo, s.ci_hi, color=cols, lw=2)
ax.scatter(s.dz_lit, y, s=50, color=cols, edgecolor=BG, lw=1.5, zorder=3)
ax.axvline(0, color='#999999', lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{r}: {l}" for r, l in zip(s.row, s.label)], fontsize=8)
ax.grid(axis='y', visible=False)
ax.set_xlabel("Cohen's dz, out vs in, signed so that > 0 = direction reported in the literature (95% CI)")
ax.legend([Line2D([], [], ls='none', marker='o', color=c, ms=7) for c in [AGREE, OPPOSITE, NULL]],
          ['agrees (uncorrected p < .05)', 'opposite (uncorrected p < .05)', 'no effect'], frameon=False, fontsize=8, ncol=3,
          loc='lower left', bbox_to_anchor=(0, 1.0))
ax.set_title("Do our data agree with the literature table? None of the 13 contrasts has FDR q < .05", loc='left',
             pad=24)
fig.savefig(os.path.join(OUT_DIR, 'zone_agreement.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Summary (run of 2026-11-10, 17-18 subjects)
#
# The zone split is behaviourally valid: d' 3.62 in vs 2.83 out (paired p < 1e-4).
#
# | Row | Literature (out vs in) | Ours | Verdict |
# |---|---|---|---|
# | Frontal-midline theta, transition-locked | lower | **higher** at Fz (dz = +0.57, p = 0.03, q = 0.14) and Cz (+0.79, p = 0.005, q = 0.06); the Fz theta time course is higher out of the zone over the whole trial | opposite, nominally |
# | Theta variability across the run | high-variability people have a bigger d' drop | between subjects r = -0.28 (p = 0.28, n = 17); within subject higher out (dz = +0.37, p = 0.14; partial r p = 0.014) | not replicated (between); trend in the expected direction (within) |
# | Posterior alpha power | higher | higher at Oz (dz = +0.46, p = 0.07; partial r = 0.055, p = 0.012) and Pz (+0.43, p = 0.09) | same direction, not significant |
# | P3 (no-go) | smaller | window mean unchanged (p > 0.5), but the grand average peaks lower and later out of the zone (7.8 vs 6.3 uV, p < .05 at 0.63-0.73 s) | partly: peak / latency, not window mean |
# | P1 / N1 | smaller | no effect; the gradual transitions evoke no clear P1 / N1 at Oz, so this row is weakly testable | no effect |
# | CPP build-up rate | slower, more variable | single-trial slope and its SD unchanged (p > 0.3); grand average: more negative pre-response trough and steeper rise in the zone (p < .05 at -0.56 to -0.28 and -0.22 to -0.15 s) | partly (grand average only) |
# | Trial-to-trial response variability | higher | theta ITPC lower out of the zone at Pz (dz = -0.68, p = 0.013, q = 0.09) and Oz (-0.51, p = 0.06) | agree, nominally (ITPC used as proxy) |
# | Responses to background distractors | smaller (fMRI) | not testable without a distractor manipulation | - |
#
# None of the 13 directional contrasts survives Benjamini-Hochberg FDR (smallest q: Cz theta 0.063, Pz ITPC 0.086), so
# the verdicts are nominal. The continuous analysis agrees: Pz theta ITPC (rolling, 15 trials) vs VTC_smoothed with
# time-on-task partialled is r = -0.056, p = 0.026, q = 0.115 (EEG_extra_features_vs_VTC.py).
# Caveats: the zone split is not corrected for time-on-task; the
# response-locked CPP baseline is the pre-onset interval, so its level depends on RT, which differs between zones by
# construction (VTC is RT deviation).
