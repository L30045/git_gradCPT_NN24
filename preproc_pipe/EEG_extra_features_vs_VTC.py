# %% [markdown]
# # Literature-survey features vs VTC
#
# Do the EEG and eye-tracking features from the literature survey (2026-11-06) that were not yet in the pipeline
# track the variance time course (VTC) in gradCPT? Features come from `extract_extra_trial_features.py`
# (`<subject>_extra_trial_features.csv.gz`); VTC and trial condition from the EEG feature table of
# `extract_EEG_trial_features.py`. The feature definitions are in the header of `extract_extra_trial_features.py`.
#
# **Trials.** *window* features (centred 10 s window) and *rolling* features (trial-to-trial statistics over
# 9-15 trials) use all trials. *trial* features (one value per trial: stimulus-locked theta, P1/N1, CPP, wavelet,
# baseline pupil, TEPR) use `city_correct` trials only, so every trial has a button press.
#
# **Statistics** (same two-level design as `EEG_features_vs_behavior.py`)
# 1. Per subject: Spearman r between feature and `VTC_smoothed` / `VTC`; *partial* = after removing run means and a
#    per-run linear time-on-task trend from both ranks. *quad* = standardised quadratic coefficient in
#    `VTC_smoothed ~ x + x^2` on the partialled ranks (x = feature), the inverted-U / U test motivated by
#    van den Brink et al. (2016) and the survey's "direction is not universal" takeaway.
# 2. Group: Fisher-z mean over subjects (quad: mean coefficient), one-sample t-test against 0, Benjamini-Hochberg FDR
#    within each (family, behaviour, statistic). Per-subject p-values are not used: windowed / rolling features and
#    `VTC_smoothed` are strongly autocorrelated.
#
# `ref` features (baseline pupil, pupil derivative, theta / alpha power) are already in the pipeline; they are
# re-measured here only as a point of comparison.
#
# Outputs: `<data_save_path>/xSubj_results/EEG_extra_features_vs_VTC/`.

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
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

code_dir = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eeg/preproc_pipe'
sys.path.insert(0, code_dir)
from utils import data_save_path
from extract_EEG_trial_features import get_features_path
from extract_extra_trial_features import get_extra_features_path, feature_info

warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

OUT_DIR = os.path.join(data_save_path, 'xSubj_results', 'EEG_extra_features_vs_VTC')
os.makedirs(OUT_DIR, exist_ok=True)
MIN_TRIALS = 30
MIN_SUBJ = 6
FDR_Q = 0.05
N_BINS = 5
BEHAVS = ['VTC_smoothed', 'VTC']
FAMILY_LABEL = {'eeg_spectral': 'EEG spectral shape / coherence (10 s window)',
                'eeg_trial': 'EEG stimulus-locked: theta, P1/N1, CPP',
                'eeg_dwt': 'EEG wavelet bands at Fz (Sensors 2021)',
                'eye': 'Pupil and eye movements'}
INFO = feature_info()

INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
POS, NEG, MUTED = '#e34948', '#2a78d6', '#b4b3ae'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999999',
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'font.size': 9,
                     'axes.titlesize': 10, 'axes.titlecolor': INK, 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'savefig.facecolor': BG})

# %% [markdown]
# ## 2. Data

# %%
def load_all():
    dfs = []
    for p in sorted(glob.glob(get_extra_features_path('sub-*'))):
        subj = os.path.basename(os.path.dirname(os.path.dirname(p)))
        x = pd.read_csv(p)
        b = pd.read_csv(get_features_path(subj), usecols=['run', 'trial_idx', 'trial_type', 'response_code',
                                                           'VTC', 'VTC_smoothed'])
        x = x.merge(b, on=['run', 'trial_idx', 'trial_type', 'response_code'], how='inner', validate='1:1')
        dfs.append(x)
    df = pd.concat(dfs, ignore_index=True)
    df['condition'] = np.select([(df.trial_type == 'city') & (df.response_code == 1), df.trial_type == 'city',
                                 (df.trial_type == 'mnt') & (df.response_code == 0)],
                                ['city_correct', 'city_incorrect', 'mnt_correct'], 'mnt_incorrect')
    return df.sort_values(['subject', 'run', 'trial_idx']).reset_index(drop=True)


df = load_all()
FEATS = [f for f in INFO if f in df]
cov = df.groupby('subject')[['blink_rate', 'pupil_base', 'fz_ap_exp']].apply(lambda g: g.notna().mean()).round(2)
print(f"{df.subject.nunique()} subjects, {len(df):,} trials, {len(FEATS)} features")
print(f"eye tracking available for {(cov.pupil_base > 0).sum()} subjects")
cov

# %% [markdown]
# ## 3. Statistics

# %%
def time_residual_ranks(x, run, tidx):
    r = stats.rankdata(x)
    out = np.empty_like(r, dtype=float)
    for u in np.unique(run):
        m = run == u
        A = np.column_stack([np.ones(m.sum()), tidx[m]])
        out[m] = r[m] - A @ np.linalg.lstsq(A, r[m], rcond=None)[0]
    return out


def trial_mask(g, feat):
    return (g.condition == 'city_correct').values if INFO[feat]['scope'] == 'trial' else np.ones(len(g), bool)


def pair_stats(x, y, run, tidx):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_TRIALS or np.std(x[m]) == 0 or np.std(y[m]) == 0 or len(np.unique(x[m])) < 5:
        return np.nan, np.nan, np.nan, int(m.sum())
    r_raw = stats.spearmanr(x[m], y[m])[0]
    rx = time_residual_ranks(x[m], run[m], tidx[m])
    ry = time_residual_ranks(y[m], run[m], tidx[m])
    r_part = np.corrcoef(rx, ry)[0, 1]
    zx, zy = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    q = zx ** 2 - (zx ** 2).mean()
    B = np.linalg.lstsq(np.column_stack([np.ones_like(zx), zx, q]), zy, rcond=None)[0]
    return r_raw, r_part, B[2], int(m.sum())


def per_subject_stats(df):
    rows = []
    for subj, g in df.groupby('subject'):
        run, tidx = g.run.values, g.trial_idx.values.astype(float)
        for feat in FEATS:
            if g[feat].notna().sum() == 0:
                continue
            m = trial_mask(g, feat)
            x = g[feat].values[m].astype(float)
            for behav in BEHAVS:
                r_raw, r_part, quad, n = pair_stats(x, g[behav].values[m].astype(float), run[m], tidx[m])
                rows.append(dict(subject=subj, feature=feat, behaviour=behav, raw=r_raw, partial=r_part,
                                 quad=quad if behav == 'VTC_smoothed' else np.nan, n=n))
    return pd.DataFrame(rows)


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if m.sum():
        ps = p[m]
        o = np.argsort(ps)
        ranked = np.minimum.accumulate((ps[o] * m.sum() / np.arange(1, m.sum() + 1))[::-1])[::-1]
        out = np.empty_like(ps)
        out[o] = np.clip(ranked, 0, 1)
        q[m] = out
    return q


def group_stats(subj_df):
    rows = []
    for (feat, behav), g in subj_df.groupby(['feature', 'behaviour']):
        for stat in ['raw', 'partial'] + (['quad'] if behav == 'VTC_smoothed' else []):
            v = g[stat].dropna().values
            row = dict(family=INFO[feat]['family'], feature=feat, label=INFO[feat]['label'], ref=INFO[feat]['ref'],
                       scope=INFO[feat]['scope'], behaviour=behav, stat=stat, n_subj=len(v),
                       mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, t=np.nan, p=np.nan, n_pos=np.nan)
            if len(v) >= MIN_SUBJ:
                z = v if stat == 'quad' else np.arctanh(np.clip(v, -0.999, 0.999))
                t, p = stats.ttest_1samp(z, 0)
                half = stats.t.ppf(0.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
                back = (lambda a: a) if stat == 'quad' else np.tanh
                row.update(mean=back(z.mean()), ci_lo=back(z.mean() - half), ci_hi=back(z.mean() + half),
                           t=t, p=p, n_pos=int((v > 0).sum()))
            rows.append(row)
    gdf = pd.DataFrame(rows)
    gdf['q'] = np.nan
    for _, idx in gdf.groupby(['family', 'behaviour', 'stat']).groups.items():
        gdf.loc[idx, 'q'] = bh_fdr(gdf.loc[idx, 'p'])
    gdf['sig'] = gdf.q < FDR_Q
    return gdf


# %%
subj_df = per_subject_stats(df)
subj_df.to_csv(os.path.join(OUT_DIR, 'per_subject_stats.csv'), index=False)
gdf = group_stats(subj_df)
gdf.to_csv(os.path.join(OUT_DIR, 'group_stats.csv'), index=False)
print(f"{len(subj_df):,} per-subject values, {len(gdf):,} group tests")

# %% [markdown]
# ### 3.1 FDR-significant relations (q < 0.05)

# %%
cols = ['family', 'label', 'ref', 'behaviour', 'stat', 'n_subj', 'n_pos', 'mean', 'ci_lo', 'ci_hi', 't', 'q']
sig = gdf[gdf.sig].sort_values(['behaviour', 'stat', 'p'])
print(f"{len(sig)} FDR-significant tests")
sig[cols].round(3)

# %% [markdown]
# ### 3.2 Strongest relation per feature family (VTC_smoothed, time-on-task partialled)

# %%
part = gdf[(gdf.behaviour == 'VTC_smoothed') & (gdf.stat == 'partial')].dropna(subset=['t'])
part.reindex(part.t.abs().sort_values(ascending=False).index).groupby('family').head(5)[cols].round(3)

# %% [markdown]
# ## 4. Figures

# %%
def forest(gdf, behav, stat, fname):
    """Group mean r (95% CI) for every feature, one panel per family."""
    sub = gdf[(gdf.behaviour == behav) & (gdf.stat == stat)].dropna(subset=['mean'])
    fams = [f for f in FAMILY_LABEL if f in set(sub.family)]
    heights = [max(2, 0.16 * (sub.family == f).sum() + 0.6) for f in fams]
    H = sum(heights) + 1
    fig, axes = plt.subplots(len(fams), 1, figsize=(7.5, H),
                             gridspec_kw=dict(height_ratios=heights, hspace=0.35, top=1 - 0.8 / H))
    axes = np.atleast_1d(axes)
    lim = np.nanmax(np.abs(sub[['ci_lo', 'ci_hi']].values)) * 1.1
    for ax, fam in zip(axes, fams):
        s = sub[sub.family == fam].iloc[::-1]
        y = np.arange(len(s))
        col = np.where(~s.sig, MUTED, np.where(s['mean'] > 0, POS, NEG))
        ax.hlines(y, s.ci_lo, s.ci_hi, color=col, lw=1.6)
        ax.scatter(s['mean'], y, s=28, color=col, edgecolor=BG, linewidth=1, zorder=3)
        ax.axvline(0, color='#999999', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([l + (' (ref)' if r else '') for l, r in zip(s.label, s.ref)], fontsize=7)
        for tl, sg in zip(ax.get_yticklabels(), s.sig):
            if sg:
                tl.set_fontweight('bold'); tl.set_color(INK)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.7, len(s) - 0.3)
        ax.grid(axis='y', visible=False)
        ax.set_title(f"{FAMILY_LABEL[fam]}  (n = {int(s.n_subj.max())} subjects)", loc='left', fontsize=9)
    what = {'partial': 'Spearman r, time-on-task partialled', 'raw': 'Spearman r, raw',
            'quad': 'quadratic coefficient (> 0 = U, < 0 = inverted U)'}[stat]
    axes[-1].set_xlabel(f"group mean {what}  (95% CI over subjects)")
    fig.suptitle(f"New features vs {behav}: {what}\ncoloured / bold = FDR q < {FDR_Q} within family "
                 f"(red: higher out of the zone, blue: lower)", x=0.02, ha='left', color=INK, fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()


def binned(df, gdf, feats, fname, behav='VTC_smoothed'):
    """Feature (z within subject) by within-subject VTC quintile, mean +- SEM over subjects."""
    if not feats:
        return
    ncol = min(4, len(feats))
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow), squeeze=False)
    for ax, feat in zip(axes.ravel(), feats):
        curves = []
        for _, s in df.groupby('subject'):
            m = trial_mask(s, feat)
            x, v = s[feat].values[m].astype(float), s[behav].values[m]
            ok = np.isfinite(x) & np.isfinite(v)
            if ok.sum() < MIN_TRIALS or np.std(x[ok]) == 0:
                continue
            z = (x[ok] - x[ok].mean()) / x[ok].std()
            bins = pd.qcut(stats.rankdata(v[ok]), N_BINS, labels=False)
            curves.append([z[bins == b].mean() for b in range(N_BINS)])
        C = np.array(curves)
        mu, se = C.mean(0), C.std(0, ddof=1) / np.sqrt(len(C))
        xb = np.arange(1, N_BINS + 1)
        ax.fill_between(xb, mu - se, mu + se, color=SERIES[0], alpha=0.2, lw=0)
        ax.plot(xb, mu, '-o', color=SERIES[0], lw=2, ms=7, mec=BG, mew=2)
        ax.axhline(0, color='#999999', lw=0.8)
        ax.set_xticks(xb)
        r = gdf[(gdf.feature == feat) & (gdf.behaviour == behav) & (gdf.stat == 'partial')].iloc[0]
        ref = ' (ref)' if INFO[feat]['ref'] else ''
        ax.set_title(f"{INFO[feat]['label']}{ref}\nr = {r['mean']:.3f}, q = {r.q:.3f}, n = {len(C)}", loc='left', fontsize=8.5,
                     fontweight='bold' if r.sig else 'normal')
        ax.set_xlabel('VTC quintile (1 = in the zone)', fontsize=8)
        ax.set_ylabel('feature (z within subject)', fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    fig.suptitle(f'Features by {behav} quintile (mean ± SEM across subjects); bold title = FDR q < {FDR_Q}', x=0.02, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()


def subject_dots(subj_df, gdf, feats, fname, behav='VTC_smoothed'):
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(feats) + 1.4))
    rng = np.random.default_rng(0)
    for i, feat in enumerate(feats):
        row = gdf[(gdf.feature == feat) & (gdf.behaviour == behav) & (gdf.stat == 'partial')].iloc[0]
        v = subj_df[(subj_df.feature == feat) & (subj_df.behaviour == behav)]['partial'].dropna()
        y = len(feats) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=20, color=SERIES[0], alpha=0.55, edgecolor=BG, lw=1, zorder=3)
        ax.plot([row.ci_lo, row.ci_hi], [y, y], color=INK, lw=2, zorder=4)
        ax.plot(row['mean'], y, 'o', ms=7, color=INK, mec=BG, mew=2, zorder=5)
    ax.axvline(0, color='#999999', lw=0.8)
    lab = [f"{INFO[f]['label']}{' (ref)' if INFO[f]['ref'] else ''}   q={gdf[(gdf.feature == f) & (gdf.behaviour == behav) & (gdf.stat == 'partial')].q.iloc[0]:.3f}"
           for f in feats]
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(lab[::-1], fontsize=8)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'per-subject Spearman r with {behav} (time-on-task partialled)')
    ax.legend([Line2D([], [], ls='none', marker='o', color=SERIES[0], alpha=0.55, ms=5),
               Line2D([], [], color=INK, lw=2, marker='o', ms=6)], ['subject', 'group mean, 95% CI'],
              frameon=False, loc='lower right', fontsize=8)
    ax.set_title('Subject-level r: FDR-significant features, then the strongest new features', loc='left')
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()


# %% [markdown]
# ### 4.1 All features vs smoothed VTC (time-on-task partialled)

# %%
forest(gdf, 'VTC_smoothed', 'partial', 'forest_VTC_smoothed_partial.png')

# %% [markdown]
# ### 4.2 Raw (not partialled) and unsmoothed VTC

# %%
forest(gdf, 'VTC_smoothed', 'raw', 'forest_VTC_smoothed_raw.png')
forest(gdf, 'VTC', 'partial', 'forest_VTC_partial.png')

# %% [markdown]
# ### 4.3 Non-linear (U / inverted-U) relation with smoothed VTC

# %%
forest(gdf, 'VTC_smoothed', 'quad', 'forest_VTC_smoothed_quad.png')

# %% [markdown]
# ### 4.4 Strongest features: quintile curves and subject-level r

# %%
top = part.reindex(part.t.abs().sort_values(ascending=False).index)
# FDR-significant features first (reference ones included, marked "ref"), then the strongest new features
top_feats = list(dict.fromkeys(list(top[top.sig].feature) + list(top[~top.ref].feature)))[:8]
binned(df, gdf, top_feats, 'binned_by_VTC_top_new_features.png')
subject_dots(subj_df, gdf, top_feats, 'subject_r_top_new_features.png')

# %% [markdown]
# ## 5. Summary (run of 2026-11-10; 18 subjects EEG, 12 subjects eye tracking)
#
# 97 features (91 new, 6 reference); FDR within family. Effect sizes are small throughout (|r| <= 0.07 after
# partialling run and time-on-task), similar to the band-power result in `EEG_features_vs_behavior.py`.
#
# * **Survives FDR, partialled:** only the *reference* Fz theta power (10 s centred window; r = 0.070, 16/18 subjects
#   positive, q = 0.005), i.e. higher frontal theta out of the zone. This replicates the earlier 20 s band-power
#   finding with a shorter window, and runs opposite to Solís-Vivanco et al. (2024), who report *lower* frontal-midline
#   theta out of the zone; theirs is stimulus-locked theta around transitions, ours is ongoing power.
# * **Survives FDR, raw only (not partialled):** pupil-size variation (SD of pupil in 10 s; r = 0.10, 11/12 subjects,
#   q = 0.007; partialled r = 0.067, p = 0.011, q = 0.17) and Oz stimulus-locked theta (r = 0.041, q = 0.03;
#   partialled q = 0.09). With unsmoothed VTC, pupil-size variation also survives raw (q = 0.013). Part of these
#   relations is shared with slow time-on-task drifts.
# * **Trends (p < 0.05, q < 0.2, partialled):** theta trial-to-trial SD at Cz / Fz (r = 0.057 / 0.034; more variable
#   theta out of the zone, the direction Solís-Vivanco et al. report), Fz theta/beta ratio (0.055), Fz aperiodic
#   offset (0.051), Pz theta ITPC (-0.056; less phase locking out of the zone); q = 0.077-0.115.
#   Baseline-pupil trial SD has p = 0.044 but q = 0.33.
#   The quintile curves rise mostly between quintiles 1 and 2-3, i.e. the most in-the-zone fifth differs most.
# * **No relation:** aperiodic exponent, IAF, periodic (1/f-removed) alpha / theta, median frequency, coherence,
#   P1/N1, CPP slope / amplitude, TEPR, blink / fixation / saccade / microsaccade metrics, wavelet entropy / energy.
#   Periodic theta (1/f removed) does not track VTC although total theta does, so the theta effect may be partly
#   broadband (aperiodic offset trends the same way).
# * **Inverted-U / U:** no quadratic term survives FDR (largest: Oz theta power U-shape, p = 0.003, q = 0.14). The
#   baseline pupil goes in the direction of van den Brink et al. (2016): their inverted-U in performance is a U in
#   VTC (worse at small and large pupil), and the pupil quadratic term is positive (mean 0.037, 12 subjects,
#   p = 0.044) but does not survive FDR (q > 0.3).
# * Caveats: eye-tracking n = 12 (sub-671/695/747/755/765 have no physio file; sub-763 and sub-746 runs 1-2 have no
#   fNIRS-time events.tsv to place the eye data on); Neon accuracy limits microsaccades; blinks and saccades are rare
#   during the task, so 10 s rates are often 0.
