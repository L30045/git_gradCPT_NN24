"""
Load helpers for the figures and tables saved by phasic_single_run.py.

Compute first (writes PNG/SVG + a phase summary CSV):
    python phasic_single_run.py sub-723 1 vtc save

Then in a notebook:
    import utils_phasic as up
    up.list_runs()                                # what has been computed
    up.load_summary('sub-723', 1)                 # phase-locking table
    up.show_fig('sub-723', 1, 'raster')           # one saved figure

Figures are re-encoded at reduced width (JPEG by default) before being
embedded, so the notebook stays small even though the saved PNGs are 300 dpi.
"""
import io
import os
import re
import glob
import pandas as pd

project_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/'
phasic_dir = os.path.join(project_path, 'derivatives', 'eye_tracking', 'phasic')

# figure names written by phasic_single_run.py, in script order
FIGURES = ['raster', 'spectrum', 'phase_across_trials', 'phase_stability_vs_lag', 'vs_{cov}', 'run_psd']


def _tag(subj, run_id):
    return f'{subj}_run-{int(run_id):02d}'


def fig_path(subj, run_id, name):
    return os.path.join(phasic_dir, subj, f'{_tag(subj, run_id)}_{name}.png')


def list_runs():
    """DataFrame of (subj, run) with which figures / summary exist on disk."""
    rows = {}
    for f in glob.glob(os.path.join(phasic_dir, 'sub-*', 'sub-*_run-*_*.png')) + \
             glob.glob(os.path.join(phasic_dir, 'sub-*', 'sub-*_run-*_*.csv')):
        m = re.match(r'(sub-[^_]+)_run-(\d+)_(.+)\.(png|csv)$', os.path.basename(f))
        if m:
            rows.setdefault((m.group(1), int(m.group(2))), set()).add(m.group(3))
    out = pd.DataFrame([{'subj': s, 'run': r, 'outputs': ', '.join(sorted(v))}
                        for (s, r), v in sorted(rows.items())])
    return out


def load_status(covariate='vtc'):
    """Batch status written by run_phasic_all.sh (subj, run, status, message), or None."""
    f = os.path.join(phasic_dir, f'batch_status_{covariate}.tsv')
    return pd.read_csv(f, sep='\t').fillna('') if os.path.isfile(f) else None


def available_runs(cov_name='vtc'):
    """Subject x run table: 'ok' if all figures + summary are saved, else batch failure reason / '-'."""
    need = set(n.format(cov=cov_name) for n in FIGURES) | {'phase_summary', 'run_power_behavior'}
    df = list_runs()
    rows = []
    if len(df):
        for _, r in df.iterrows():
            got = set(r['outputs'].split(', '))
            rows.append({'subj': r['subj'], 'run': r['run'],
                         'status': 'ok' if need <= got else 'incomplete'})
    st = load_status(cov_name if cov_name == 'vtc' else 'theta')
    if st is not None:
        for _, r in st[st.status == 'failed'].iterrows():
            msg = re.sub(r'^AssertionError: sub-\S+_run-\d+: ', '', r['message'])
            rows.append({'subj': r['subj'], 'run': int(r['run']), 'status': f'failed: {msg}'})
    tab = pd.DataFrame(rows).drop_duplicates(['subj', 'run'])
    return tab.pivot(index='subj', columns='run', values='status').rename(columns=lambda c: f'run-{c:02d}').fillna('-')


def load_all_summaries():
    """Concatenate every saved phase summary into one long table (subj, run, window, ...)."""
    out = []
    for f in sorted(glob.glob(os.path.join(phasic_dir, 'sub-*', 'sub-*_run-*_phase_summary.csv'))):
        m = re.match(r'(sub-[^_]+)_run-(\d+)_', os.path.basename(f))
        d = pd.read_csv(f)
        d.insert(0, 'run', int(m.group(2)))
        d.insert(0, 'subj', m.group(1))
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_summary(subj, run_id):
    """Phase-locking summary table (one row per phase window)."""
    f = os.path.join(phasic_dir, subj, f'{_tag(subj, run_id)}_phase_summary.csv')
    if not os.path.isfile(f):
        raise FileNotFoundError(f'{f} not found; run: python phasic_single_run.py {subj} {run_id} vtc save')
    return pd.read_csv(f).set_index('window')


def show_fig(subj, run_id, name, width=1400, fmt='jpeg', quality=80):
    """Display a saved figure, downscaled to `width` px and re-encoded (fmt 'jpeg' or 'png')."""
    from PIL import Image as PILImage
    from IPython.display import Image, display, Markdown
    f = fig_path(subj, run_id, name)
    if not os.path.isfile(f):
        display(Markdown(f'*Missing: `{f}` — run `python phasic_single_run.py {subj} {run_id} <cov> save`*'))
        return
    im = PILImage.open(f)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    if fmt == 'jpeg':
        im.convert('RGB').save(buf, format='JPEG', quality=quality, optimize=True)
    else:
        im.save(buf, format='PNG', optimize=True)
    display(Image(data=buf.getvalue(), format=fmt))


def show_all(subj, run_id, covariate='vtc', **kw):
    """Display every figure of one run in script order."""
    for name in FIGURES:
        show_fig(subj, run_id, name.format(cov=covariate), **kw)


# --------------------------------------------------------------------------- run-level power vs behavior
POWER_COLS = {'run_power_at_f': 'log10 power at 1.25 Hz (mm²/Hz)', 'run_snr': 'log10 SNR at 1.25 Hz'}
# behavior metric -> (label, how the per-subject legend summarises it: 'sum' of counts or 'mean')
BEHAVIOR_COLS = {'n_city_correct':  ('# city correct', 'sum'),
                 'n_mnt_correct':   ('# mnt correct', 'sum'),
                 'n_mnt_incorrect': ('# mnt incorrect', 'sum'),
                 'rt_mean':         ('mean RT (s)', 'mean'),
                 'rt_cv':           ('RT CV (SD / mean)', 'mean')}


def normalize_behavior(df, behavior_cols=BEHAVIOR_COLS):
    """Copy of df with each behavior metric divided by that subject's mean over its runs
    (1 = the subject's average run; a subject with a single run is exactly 1)."""
    out = df.copy()
    for bc in behavior_cols:
        out[bc] = df[bc] / df.groupby('subj')[bc].transform('mean')
    return out


def load_run_power_behavior():
    """One row per run: whole-run 1.25 Hz power (+ SNR) and behavior, from *_run_power_behavior.csv."""
    files = sorted(glob.glob(os.path.join(phasic_dir, 'sub-*', 'sub-*_run-*_run_power_behavior.csv')))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _rmcorr(x, y, g):
    """Repeated-measures correlation (Bakdash & Marusich 2017): Pearson r of subject-centred
    values, df = N - n_subjects - 1."""
    import numpy as np
    from scipy import stats
    d = pd.DataFrame({'x': x, 'y': y, 'g': g}).dropna()
    d = d[d.groupby('g')['x'].transform('size') > 1]
    xc = d['x'] - d.groupby('g')['x'].transform('mean')
    yc = d['y'] - d.groupby('g')['y'].transform('mean')
    dof = len(d) - d['g'].nunique() - 1
    if dof < 1 or xc.std() == 0 or yc.std() == 0:
        return np.nan, np.nan, dof
    r = np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, 2 * stats.t.sf(abs(t), dof), dof


def power_behavior_stats(df, power_cols=POWER_COLS, behavior_cols=BEHAVIOR_COLS):
    """For each power x behavior pair:
    - all runs: Spearman over every run (runs of a subject are not independent)
    - within subject: repeated-measures correlation (subject-centred, log10 power)
    - between subject: Spearman of subject means"""
    import numpy as np
    from scipy import stats
    rows = []
    for pc in power_cols:
        lp = np.log10(df[pc])
        for bc in behavior_cols:
            ok = np.isfinite(lp) & np.isfinite(df[bc])
            rho_all = stats.spearmanr(lp[ok], df.loc[ok, bc])
            r_w, p_w, dof = _rmcorr(lp, df[bc], df['subj'])
            sm = pd.DataFrame({'p': lp, 'b': df[bc], 's': df['subj']})[ok].groupby('s').mean()
            rho_b = stats.spearmanr(sm['p'], sm['b']) if len(sm) > 2 else (np.nan, np.nan)
            rows.append({'power': pc, 'behavior': bc, 'n_runs': int(ok.sum()), 'n_subj': len(sm),
                         'rho_all_runs': rho_all[0], 'p_all_runs': rho_all[1],
                         'r_within_subj': r_w, 'p_within_subj': p_w, 'df_within': dof,
                         'rho_between_subj': rho_b[0], 'p_between_subj': rho_b[1]})
    return pd.DataFrame(rows).set_index(['power', 'behavior'])


def plot_power_vs_behavior(df, power_col='run_power_at_f', behavior_cols=BEHAVIOR_COLS):
    """2 x 3 grid: log10 whole-run power vs each behavior metric divided by the subject's own mean over runs.
    Statistics in the titles use the raw (un-normalised) metric.
    One point per run, colour = subject, runs of a subject joined by a line. Each panel's legend
    gives the subject's raw total (counts) or mean (RT) of that metric; the last panel lists subjects."""
    import numpy as np
    import matplotlib.pyplot as plt
    dn = normalize_behavior(df, behavior_cols)
    st = power_behavior_stats(df, {power_col: POWER_COLS.get(power_col, power_col)}, behavior_cols)
    subjects = sorted(df['subj'].unique())
    cmap = plt.get_cmap('tab20', max(len(subjects), 1))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    lp = np.log10(df[power_col])
    for ax, (bc, (blab, agg)) in zip(axes.flat, behavior_cols.items()):
        for k, s in enumerate(subjects):
            m = (df['subj'] == s).values
            raw = df.loc[m, bc]
            tot = f'{raw.sum():.0f}' if agg == 'sum' else f'{raw.mean():.3g}'
            o = np.argsort(dn.loc[m, bc].values)
            ax.plot(dn.loc[m, bc].values[o], lp[m].values[o], '-o', ms=4, lw=0.6, color=cmap(k),
                    label=f'{s}: {tot}')
        r = st.loc[(power_col, bc)]
        ax.axvline(1, color='grey', lw=0.6, ls=':')
        ax.set_xlabel(f'{blab} / subject mean over runs')
        ax.set_ylabel(POWER_COLS.get(power_col, power_col))
        ax.set_title(f"all runs ρ = {r.rho_all_runs:.2f} (p = {r.p_all_runs:.2g})\n"
                     f"within subj r = {r.r_within_subj:.2f} (p = {r.p_within_subj:.2g}); "
                     f"between ρ = {r.rho_between_subj:.2f} (p = {r.p_between_subj:.2g})", fontsize=9)
        ax.legend(fontsize=6, ncol=2, title=f"{'total' if agg == 'sum' else 'mean'} per subject",
                  title_fontsize=7, loc='best', framealpha=0.6)
    # last panel: subject key with number of runs and trials
    ax = axes.flat[len(behavior_cols)]
    ax.axis('off')
    for k, s in enumerate(subjects):
        m = df['subj'] == s
        ax.plot([], [], '-o', color=cmap(k), label=f"{s}: {m.sum()} runs, {df.loc[m, 'n_trials'].sum()} trials")
    ax.legend(loc='center', fontsize=8, ncol=1, title='subjects', frameon=False)
    for extra in list(axes.flat)[len(behavior_cols) + 1:]:
        extra.axis('off')
    fig.suptitle(f'Whole-run pupil {POWER_COLS.get(power_col, power_col)} vs behavior '
                 f'({len(df)} runs, {len(subjects)} subjects; one point per run; x = metric / subject mean)')
    plt.tight_layout()
    return fig


def show(fig, fmt='png', dpi=72):
    """Display a matplotlib figure as a compact embedded image and close it."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    display(Image(data=buf.getvalue(), format=fmt))
