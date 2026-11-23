#%% Compare the mnt-correct HRF of AR-IRLS, fixAR-IRLS and AR-iRRR for two designs (reads saved results, refits nothing)
# - event-based: Gaussian kernels at the mnt-correct onsets, comparison/cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py (all runs)
#   -> derivatives/eeg/<subject>/<subject>_mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.pkl, HRF in µM
# - cont-EEG: Cz EEG within mnt-correct trials as the regressor, model training/run_model_cont_EEG_city_mnt.py
#   -> betas/stats of cont_EEG_cz_mnt_correct_incorrect_pre2post18_bl0.2-0_<model>_Hp0.01, HRF in µM/V
# Figures (out_dir):
#   event-based mnt-correct/, cont-EEG mnt-correct/:
#     <subject>_parcels.png     most sensitive ContA and DorsAttnA parcel, three models (µM or µM/V)
#     <subject>_networks.png    network-mean HRF of the 12 main networks
#     group_parcels.png, group_networks.png   mean ± t-based 95% CI across subjects
#   event-based_vs_cont-EEG_networks_<model>.png   group network HRF of both designs for one model (common subjects;
#     event-based on the left axis in µM, cont-EEG on the right axis in µM/V)
#   event-based mnt-correct old vs new/group_networks_<model>.png   event-based group network HRF of one model, old fit
#     (archive/pre-hpDesign_2026-11-20: unfiltered design, old ADOT) vs new fit (highpassed design, current ADOT)
# No figure is normalized.
# All HRFs are baseline-corrected: the mean over the pre-onset delays (t < 0) is subtracted per parcel.
# The legend gives the AR-iRRR rank (per subject, or its distribution across subjects).
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
modeling_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, modeling_path)   # model.py, params_setting.py
from params_setting import project_path, excluded_subj, get_betas_path, get_stats_path
import cedalion.io

#%% parameters
select_chromo = 'HbO'
parcel_networks = ['ContA', 'DorsAttnA']   # the most sensitive parcel of each is plotted
network_order = ['VisCent', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA',
                 'SalVentAttnB', 'ContA', 'ContB', 'DefaultA', 'DefaultB', 'TempPar']
models = ['AR-IRLS', 'fixAR-IRLS', 'AR-iRRR']
model_colors = {'AR-IRLS': 'C0', 'fixAR-IRLS': 'C3', 'AR-iRRR': 'C2'}
design_colors = {'event-based': 'C0', 'cont-EEG': 'C1'}
hrf_dt = 0.1   # s, the cont-EEG HRFs (saved at the 500 Hz EEG rate) are subsampled to this step

eeg_der_dir = os.path.join(project_path, 'derivatives', 'eeg')
out_dir = os.path.join(eeg_der_dir, 'cmp_HRF_AR-IRLS_fixAR-IRLS_AR-iRRR')
design_dirs = {'event-based': os.path.join(out_dir, 'event-based mnt-correct'),
               'cont-EEG': os.path.join(out_dir, 'cont-EEG mnt-correct')}
units = {'event-based': 'µM', 'cont-EEG': 'µM/V'}
design_titles = {'event-based': 'event-based mnt-correct (Gaussian kernels)',
                 'cont-EEG': 'cont-EEG mnt-correct (Cz EEG within trials)'}
for d in design_dirs.values():
    os.makedirs(d, exist_ok=True)

# event-based: comparison/cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py, trained on all runs
ev_model_name = 'mnt-correct_gauss_cmp_AR-IRLS_fixAR-IRLS_AR-iRRR'
ev_keys = {'AR-IRLS': 'ar_irls', 'fixAR-IRLS': 'fixar_irls', 'AR-iRRR': 'ar_irrr'}
# cont-EEG: model training/run_model_cont_EEG_city_mnt.py (eeg_baseline (-0.2, 0), fNIRS highpass 0.01 Hz)
ce_reg_type = 'cont_EEG_cz_mnt_correct_incorrect_pre2post18_bl0.2-0'
ce_names = {'AR-IRLS': f'{ce_reg_type}_ar_irls_Hp0.01',
            'fixAR-IRLS': f'{ce_reg_type}_fixAR-IRLS_Hp0.01',
            'AR-iRRR': f'{ce_reg_type}_iRRR_AR_Hp0.01'}

#%% most sensitive parcel of each network (forward-model sensitivity, as in cmp_AR-IRLS_fixAR-IRLS_AR-iRRR.py)
Adot = cedalion.io.load_Adot(os.path.join(project_path, 'derivatives', 'cedalion', 'fw', 'probe', 'Adot_v26.nc'))
Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
Adot_brain = Adot_brain.sel(vertex=~Adot_brain.parcel.isin(['Background+FreeSurfer_Defined_Medial_Wall_LH',
                                                            'Background+FreeSurfer_Defined_Medial_Wall_RH']).values)
sens_mask = (np.log10(Adot_brain[:, :, 1].sum('channel')) > -2).values
parcel_sensitivity = (Adot_brain.sel(vertex=sens_mask).groupby('parcel').sum('vertex')
                      .sum(['channel', 'wavelength']).to_series())

def network_of(parcels):
    return np.array([p.split('_')[0] for p in parcels])

def baseline_correct(h, t):
    """(parcel, time) array: subtract each parcel's mean over t < 0"""
    return h - h[:, t < 0].mean(axis=1, keepdims=True)

#%% load the event-based results: subject -> {'t', 'parcels', 'hrf': {model: (parcel, time) µM}, 'rank'}
def load_event(subject, root=eeg_der_dir):
    path = os.path.join(root, subject, f'{subject}_{ev_model_name}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        r = pickle.load(f)
    t = np.asarray(r['hrf_t'], float)
    hrf = {}
    for m, k in ev_keys.items():
        h = r[k]['hrf']
        hrf[m] = baseline_correct(h.sel(parcel=r['parcels']).transpose('parcel', 'time').values * 1e6, t)
    return {'t': t, 'parcels': np.asarray(r['parcels']), 'hrf': hrf, 'rank': int(r['ar_irrr']['rank'])}

#%% load the cont-EEG results: same structure, HRF in µM/V on a hrf_dt grid
def load_cont(subject):
    pre = {m: os.path.join(eeg_der_dir, subject, f'{subject}_{n}') for m, n in ce_names.items()}
    if not all(os.path.exists(get_betas_path(p)) for p in pre.values()):
        return None
    hrf, t, parcels = {}, None, None
    for m, p in pre.items():
        with open(get_betas_path(p), 'rb') as f:
            b = pickle.load(f)
        h = b['betas_eeg_per_type']['mnt_correct'].sel(chromo=select_chromo).transpose('parcel', 'regressor')
        d_t = b['basis_da'].delay_time.values
        step = int(round(hrf_dt / np.diff(d_t[:2])[0]))
        if parcels is None:
            parcels, t = h.parcel.values, d_t[::step]
        assert np.array_equal(h.parcel.values, parcels), f'{subject} {m}: parcels differ between models'
        hrf[m] = baseline_correct(h.values[:, ::step] * 1e6, t)
    with open(get_stats_path(pre['AR-iRRR']), 'rb') as f:
        rank = int(pickle.load(f)['rank_mnt_correct'])
    return {'t': t, 'parcels': parcels, 'hrf': hrf, 'rank': rank}

subjects_all = sorted(os.path.basename(d) for d in glob.glob(os.path.join(eeg_der_dir, 'sub-*')) if os.path.isdir(d))
res = {'event-based': {}, 'cont-EEG': {}}
for s in subjects_all:
    if s in excluded_subj:
        continue
    for design, loader in [('event-based', load_event), ('cont-EEG', load_cont)]:
        r = loader(s)
        if r is not None:
            res[design][s] = r
for design, r in res.items():
    print(f"{design}: {len(r)} subjects {list(r)}")

# most sensitive parcel of each network among the parcels every subject has (per design)
sel_parcels = {}
for design, r in res.items():
    common = sorted(set.intersection(*(set(x['parcels']) for x in r.values())))
    sens = parcel_sensitivity.reindex(common).dropna()
    sel_parcels[design] = {net: sens[network_of(sens.index) == net].idxmax() for net in parcel_networks}
    print(f"{design}: most sensitive parcels {sel_parcels[design]}")
assert sel_parcels['event-based'] == sel_parcels['cont-EEG'], 'the two designs pick different parcels'
sel_parcels = sel_parcels['event-based']

#%% helpers
def parcel_curve(r, m, parcel):
    return r['hrf'][m][list(r['parcels']).index(parcel)]

def network_curve(r, m, net):
    return r['hrf'][m][network_of(r['parcels']) == net].mean(axis=0)

def mean_ci(curves):
    """mean and t-based 95% CI half-width across rows (subjects)"""
    c = np.asarray(curves)
    n = c.shape[0]
    half = stats.t.ppf(0.975, n - 1) * c.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(c.shape[1])
    return c.mean(axis=0), half

def model_label(m, ranks):
    """legend entry; ranks = one int (single subject) or a list (one per subject)"""
    if m != 'AR-iRRR':
        return m
    if np.isscalar(ranks):
        return f'AR-iRRR (rank {ranks})'
    counts = pd.Series(ranks).value_counts().sort_index()
    return 'AR-iRRR (rank ' + ', '.join(f'{k}: {v}/{len(ranks)}' for k, v in counts.items()) + ')'

def style(ax, t, ylabel=None):
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)

def net_grid():
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    return fig, dict(zip(network_order, axes.ravel()))

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

#%% per-subject figures
def plot_subject(design, s, r):
    t, u, d = r['t'], units[design], design_dirs[design]
    # most sensitive parcels
    fig, axes = plt.subplots(1, len(parcel_networks), figsize=(6.5 * len(parcel_networks), 4.5))
    for ax, net in zip(axes, parcel_networks):
        p = sel_parcels[net]
        for m in models:
            ax.plot(t, parcel_curve(r, m, p), color=model_colors[m], lw=1.8, label=model_label(m, r['rank']))
        ax.set_title(f'most sensitive {net} parcel: {p}')
        style(ax, t, f'{select_chromo} ({u})')
        ax.set_xlabel('delay from mnt-correct onset (s)')
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f'{s}, {design_titles[design]}: HRF of three models (baseline-corrected)')
    save(fig, os.path.join(d, f'{s}_parcels.png'))
    # network-mean HRF
    fig, ax_of = net_grid()
    for net, ax in ax_of.items():
        n_p = int((network_of(r['parcels']) == net).sum())
        for m in models:
            ax.plot(t, network_curve(r, m, net), color=model_colors[m], lw=1.8, label=model_label(m, r['rank']))
        ax.set_title(f'{net} ({n_p} parcels)', fontsize=10)
        style(ax, t, f'{select_chromo} ({u})' if net in network_order[::4] else None)
    for net in network_order[-4:]:
        ax_of[net].set_xlabel('delay (s)')
    ax_of[network_order[0]].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{s}, {design_titles[design]}: network-mean HRF')
    save(fig, os.path.join(d, f'{s}_networks.png'))

for design, r_d in res.items():
    for s, r in r_d.items():
        plot_subject(design, s, r)
    print(f'{design}: per-subject figures saved in {design_dirs[design]}')

#%% group figures (mean ± 95% CI across subjects)
def group_curves(design, m, kind, name, subjects=None):
    r_d = res[design]
    subjects = subjects or list(r_d)
    f = parcel_curve if kind == 'parcel' else network_curve
    return r_d[subjects[0]]['t'], np.stack([f(r_d[s], m, name) for s in subjects])

def plot_band(ax, t, curves, color, label):
    mu, half = mean_ci(curves)
    line, = ax.plot(t, mu, color=color, lw=1.8, label=label)
    ax.fill_between(t, mu - half, mu + half, color=color, alpha=0.2, lw=0)
    return line

def plot_group(design):
    r_d, u, d = res[design], units[design], design_dirs[design]
    subjects, ranks = list(r_d), [r['rank'] for r in r_d.values()]
    n = len(subjects)
    fig, axes = plt.subplots(1, len(parcel_networks), figsize=(6.5 * len(parcel_networks), 4.5))
    for ax, net in zip(axes, parcel_networks):
        p = sel_parcels[net]
        for m in models:
            t, c = group_curves(design, m, 'parcel', p)
            plot_band(ax, t, c, model_colors[m], model_label(m, ranks))
        ax.set_title(f'most sensitive {net} parcel: {p}')
        style(ax, t, f'{select_chromo} ({u})')
        ax.set_xlabel('delay from mnt-correct onset (s)')
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f'{design_titles[design]}: HRF of three models, mean ± 95% CI over N = {n} subjects (baseline-corrected)')
    save(fig, os.path.join(d, 'group_parcels.png'))
    fig, ax_of = net_grid()
    for net, ax in ax_of.items():
        for m in models:
            t, c = group_curves(design, m, 'network', net)
            plot_band(ax, t, c, model_colors[m], model_label(m, ranks))
        ax.set_title(net, fontsize=10)
        style(ax, t, f'{select_chromo} ({u})' if net in network_order[::4] else None)
    for net in network_order[-4:]:
        ax_of[net].set_xlabel('delay (s)')
    ax_of[network_order[0]].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{design_titles[design]}: network-mean HRF, mean ± 95% CI over N = {n} subjects')
    save(fig, os.path.join(d, 'group_networks.png'))

for design in res:
    plot_group(design)
    print(f'{design}: group figures saved')

#%% event-based vs cont-EEG, one figure per model, on the subjects that have both designs
common_subj = sorted(set(res['event-based']) & set(res['cont-EEG']))
print(f'event-based vs cont-EEG: N = {len(common_subj)} common subjects {common_subj}')
def align_zero(ax1, ax2):
    """rescale two y axes so that 0 sits at the same height"""
    lims = [ax1.get_ylim(), ax2.get_ylim()]
    frac = max(-lo / (hi - lo) for lo, hi in lims)
    for ax, (lo, hi) in zip([ax1, ax2], lims):
        top = max(hi, -lo * (1 - frac) / frac) if frac > 0 else hi
        ax.set_ylim(-frac * top / (1 - frac), top)

for m in models:
    fig, ax_of = net_grid()
    for net, ax in ax_of.items():
        r_net, lines = {}, []
        ax2 = ax.twinx()
        for design, a in [('event-based', ax), ('cont-EEG', ax2)]:
            t, c = group_curves(design, m, 'network', net, common_subj)
            ranks = [res[design][s]['rank'] for s in common_subj]
            lab = f'{design} ({units[design]}, {"left" if a is ax else "right"} axis)' + \
                  (', ' + model_label(m, ranks)[len('AR-iRRR ('):-1] if m == 'AR-iRRR' else '')
            lines.append(plot_band(a, t, c, design_colors[design], lab))
            a.tick_params(axis='y', colors=design_colors[design], labelsize=8)
            r_net[design] = (t, c.mean(axis=0))
        align_zero(ax, ax2)
        ax2.spines[['top']].set_visible(False)
        # shape agreement of the two group means on the common delays
        (t1, y1), (t2, y2) = r_net['event-based'], r_net['cont-EEG']
        lo, hi = max(t1[0], t2[0]), min(t1[-1], t2[-1])
        grid = np.arange(lo, hi + 1e-9, hrf_dt)
        r = np.corrcoef(np.interp(grid, t1, y1), np.interp(grid, t2, y2))[0, 1]
        ax.set_title(f'{net}  (r = {r:.2f})', fontsize=10)
        style(ax, t, f'event-based ({units["event-based"]})' if net in network_order[::4] else None)
        if net in network_order[3::4]:
            ax2.set_ylabel(f'cont-EEG ({units["cont-EEG"]})')
        ax.set_xlim(lo, hi)
        if net == network_order[0]:
            ax.legend(handles=lines, fontsize=8, frameon=False, loc='upper right')
    for net in network_order[-4:]:
        ax_of[net].set_xlabel('delay from mnt-correct onset (s)')
    fig.suptitle(f'{m}: network-mean mnt-correct HRF, event-based (left axis, µM) vs cont-EEG (right axis, µM/V), '
                 f'mean ± 95% CI over N = {len(common_subj)} subjects; r = shape agreement of the group means')
    save(fig, os.path.join(out_dir, f'event-based_vs_cont-EEG_networks_{m}.png'))
#%% event-based: old fit (unfiltered design, old ADOT; archived 2026-11-20) vs new fit (highpassed design, current ADOT)
old_root = os.path.join(eeg_der_dir, 'archive', 'pre-hpDesign_2026-11-20')
old_new_dir = os.path.join(out_dir, 'event-based mnt-correct old vs new')
os.makedirs(old_new_dir, exist_ok=True)
res_old = {s: r for s in res['event-based'] if (r := load_event(s, old_root)) is not None}
on_subj = sorted(set(res_old) & set(res['event-based']))
print(f'event-based old vs new: N = {len(on_subj)} subjects {on_subj}')
versions = {'old: unfiltered design, old ADOT': (res_old, '#7f7f7f'),
            'new: highpassed design, current ADOT': (res['event-based'], None)}
for m in models:
    fig, ax_of = net_grid()
    for net, ax in ax_of.items():
        curves = {}
        for v, (r_v, color) in versions.items():
            t = r_v[on_subj[0]]['t']
            c = np.stack([network_curve(r_v[s], m, net) for s in on_subj])
            ranks = [r_v[s]['rank'] for s in on_subj]
            lab = v + (', ' + model_label(m, ranks)[len('AR-iRRR ('):-1] if m == 'AR-iRRR' else '')
            plot_band(ax, t, c, color or model_colors[m], lab)
            curves[v] = (t, c.mean(axis=0))
        (t1, y1), (t2, y2) = curves.values()
        r = np.corrcoef(y1, np.interp(t1, t2, y2))[0, 1]
        n_old, n_new = (int((network_of(r_v[on_subj[0]]['parcels']) == net).sum()) for r_v, _ in versions.values())
        ax.set_title(f'{net} ({n_old} / {n_new} parcels, r = {r:.2f})', fontsize=10)
        style(ax, t1, f'{select_chromo} (µM)' if net in network_order[::4] else None)
    for net in network_order[-4:]:
        ax_of[net].set_xlabel('delay from mnt-correct onset (s)')
    ax_of[network_order[0]].legend(fontsize=8, frameon=False)
    fig.suptitle(f'{m}: event-based mnt-correct network-mean HRF, old vs new fit, mean ± 95% CI over N = {len(on_subj)} '
                 f'subjects (baseline-corrected; parcels old / new; r = shape agreement of the group means)')
    save(fig, os.path.join(old_new_dir, f'group_networks_{m}.png'))
print(f'saved to {out_dir}')
