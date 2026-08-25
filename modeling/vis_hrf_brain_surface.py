#%% Imports
import cedalion
import pickle
import sys
import gzip
import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

from cedalion.io.forward_model import load_Adot
import pyvista as pv 
from cedalion import dot
from cedalion.vis.anatomy import image_recon_multi_view
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import pint
from tqdm import tqdm

sys.path.append('/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/cedalion_pipeline/')
import gradCPT_funcs as gcpt

pv.set_jupyter_backend('static')

#%%
SPLIT_HEMISPHERE = True
head = dot.get_standard_headmodel('icbm152')

Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = load_Adot(Adot_path  + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values) # 561 x 15002 x 2
mask = Adot_brain.parcel.isin(['Background+FreeSurfer_Defined_Medial_Wall_LH',
                                'Background+FreeSurfer_Defined_Medial_Wall_RH'])

Adot_brain = Adot_brain.sel(vertex=~mask) # 561 x 14102 x 2
intensity = np.log10(Adot_brain[:,:,1].sum('channel'))
mask = intensity > -2
sensitivity_mask = mask.drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values) # 561 x 8156 x 2
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex') # 561 x 429 x 2
Adot_parcel = Adot_parcel.assign_coords({'is_brain': ('parcel', np.ones(len(Adot_parcel.parcel), dtype=bool))}) 
vertex_parcel_labels = Adot.parcel.values

def canonicalize_parcels(label_list):
    canonical_labels = []
    for lbl in label_list:
        parts = lbl.split('_')
        # Find hemisphere
        hemi = next((p for p in parts if p in ('LH', 'RH')), None)
        # Remove network prefix and hemisphere
        name_parts = [p for p in parts if not p.startswith('Net') and p not in ('LH', 'RH')]
        # Construct canonical label: parcel_name + hemi
        canonical_lbl = '_'.join(name_parts + [hemi] if hemi else name_parts)
        canonical_labels.append(canonical_lbl)
    return canonical_labels


name_mapping_nirs =  {'VisCent': 'VisCent',
                'VisPeri': 'VisPeri',
                'SomMot1': 'SomMotA',
                'SomMot2': 'SomMotB',
                'DAN1': 'DorsAttnA',
                'DAN2': 'DorsAttnB',
                'VAN1': 'SalVentAttnA',
                'VAN2': 'SalVentAttnB',
                'LimbicB': 'LimbicB',
                'LimbicA': 'LimbicA',
                'Exec1': 'ContA',
                'Exec2': 'ContB',
                'Exec3': 'ContC',
                'DMN1': 'DefaultA',
                'DMN2': 'DefaultB',
                'DMN3': 'DefaultC',
                'TempPar': 'TempPar'
            }   

PARCELS_NOT_SENSITIVE = ['LimbicA', 'LimbicB', 'VisPeri', 'ContC', 'DefaultC']
def _base_network(label: str) -> str:
    """Strip hemisphere suffix: 'Cont_LH' → 'Cont', 'Cont' → 'Cont'."""
    for suffix in ('_LH', '_RH'):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label

#%%
ROOT = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
DATADIR = os.path.join(ROOT, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')

SPLIT_VTC = True
USE_GSR = True
NOISE_MODEL = 'ar_irls'
ADOT_FLAG = 'probe'
spatial_dim = 'vertex'
SPATIAL_DIM = 'parcel'
flag = '_lR-1e-5'
alpha_spatial = 1e-3
alpha_meas = 1e4
direct_name = 'indirect'
save_flag = ''
subj_to_drop = []

hrf_basis = 'cons_gaussians'
if SPLIT_VTC:
    flag += '_VTC_split'
if USE_GSR:
    flag += '_GSR'

filepath = os.path.join(DATADIR, f'image_hrf_ts_{spatial_dim}_as-{alpha_spatial:.0e}_am-{alpha_meas:.0e}_adot-{ADOT_FLAG}_{direct_name}_{NOISE_MODEL}{flag}_{hrf_basis}{save_flag}_v26.pkl.gz')
with gzip.open(filepath, 'rb') as f: 
    nirs_results = pickle.load(f)

fmri_path = '/projectnb/nphfnirs/s/users/lcarlton/DATA/gradCPT_fMRI/derivatives/'
flag_mri = ''
t_win = (10,13)

if SPLIT_VTC:
    flag_mri += '_VTC_split'
if USE_GSR:
    flag_mri += '_GSR'

filepath = os.path.join(fmri_path, f'image_hrf_ts_{NOISE_MODEL}{flag_mri}_{hrf_basis}_v26.pkl.gz')

with gzip.open(filepath, 'rb') as f:
    mri_results = pickle.load(f)

#%%
all_subj_hrf = nirs_results['X_hrf_ts'].sel(parcel=Adot_parcel.parcel.values)
all_subj_var = nirs_results['X_mse_corr'].sel(parcel=Adot_parcel.parcel.values)
all_subj_cov = nirs_results['X_cov_inout'].sel(parcel=Adot_parcel.parcel.values)

all_subj_hrf_mag = all_subj_hrf.sel(time=slice(t_win[0], t_win[1])).mean('time')
all_subj_var_mag = all_subj_var.sel(time=slice(t_win[0], t_win[1])).sum('time') / all_subj_var.sel(time=slice(t_win[0], t_win[1])).sizes['time']**2
all_subj_cov_mag = all_subj_cov.sel(time=slice(t_win[0], t_win[1])).sum('time') / all_subj_cov.sel(time=slice(t_win[0], t_win[1])).sizes['time']**2

diff_mag = all_subj_hrf_mag.sel(trial_type = 'mnt-correct-in') - all_subj_hrf_mag.sel(trial_type = 'mnt-correct-out')
diff_var = all_subj_var_mag.sel(trial_type = 'mnt-correct-in') + all_subj_var_mag.sel(trial_type = 'mnt-correct-out') - 2 * all_subj_cov_mag

nirs_mean, nirs_tval, _, _ = gcpt.get_weighted_group_average(diff_mag, diff_var) 

n_vertices = len(Adot.vertex)
foo_img_v = xr.DataArray(np.full([2, n_vertices], np.nan), 
                                dims=['chromo', 'vertex'],
                                coords = {'parcel': ('vertex', Adot.parcel.values),
                                        'is_brain': ('vertex', Adot.is_brain.values), 
                                        'chromo': ['HbO', 'HbR'],
                                        # 'trial_type': trials, 
                                        # 'time': image_results['X_hrf_ts'].time
                                        }
                                )      
for pp in Adot_parcel.parcel.values:
    if pp == 'scalp': 
        continue
    mask = vertex_parcel_labels == pp
    foo_img_v.loc[:,mask] = nirs_tval.sel(parcel=pp)

image_recon_multi_view(
        X_ts = foo_img_v,
        head = head,
        cmap = 'jet',
        # clim = [-6,6],
        view_type = 'hbo_brain',
        title_str = 'HbO T-stat: in-out',
        filename = None,
        SAVE = False,
        wdw_size = (1300, 768)
    )

if SPLIT_HEMISPHERE:
    network_labels = [p.split('_')[0] + '_' + p.split('_')[-1] for p in diff_mag.parcel.values]
else:
    network_labels = [p.split('_')[0] for p in diff_mag.parcel.values]

all_subj_hrf_mag['parcel'] = network_labels
all_subj_hrf_mag_net = all_subj_hrf_mag.groupby('parcel').mean('parcel')
all_subj_var_mag['parcel'] = network_labels
all_subj_var_mag_net = all_subj_var_mag.groupby('parcel').sum('parcel') / all_subj_var_mag.groupby('parcel').count()**2
all_subj_cov_mag['parcel'] = network_labels
all_subj_cov_mag_net = all_subj_cov_mag.groupby('parcel').sum('parcel') / all_subj_cov_mag.groupby('parcel').count()**2

diff_net_mag = all_subj_hrf_mag_net.sel(trial_type='mnt-correct-in') - all_subj_hrf_mag_net.sel(trial_type='mnt-correct-out')
diff_net_var = all_subj_var_mag_net.sel(trial_type='mnt-correct-in') + all_subj_var_mag_net.sel(trial_type='mnt-correct-out') - 2*all_subj_cov_mag_net

nirs_mean_net, nirs_tval_net, _, _ = gcpt.get_weighted_group_average(diff_net_mag, diff_net_var)

networks = np.array([
    n for n in nirs_tval_net.parcel.values
    if _base_network(n) not in PARCELS_NOT_SENSITIVE
])

nirs_tval_map = nirs_mean_net.reindex(parcel=networks).sel(chromo='HbO').values.squeeze()
# network_intensity = network_intensity.reindex(parcel=networks)


#%%
all_subj_hrf = mri_results['X_hrf_ts']
all_subj_var = mri_results['X_mse']
all_subj_cov = mri_results['X_cov_inout'].pint.quantify('molar**2')

parcel_labels = canonicalize_parcels(all_subj_var.parcel.values)
new_parcel_labels = []
for parcel in parcel_labels:
    parcel = str(parcel)
    tmp = parcel.split('_')
    tmp[0] = name_mapping_nirs[tmp[0]]
    tmp = '_'.join(tmp)
    new_parcel_labels.append( tmp ) 
    
all_subj_hrf['parcel'] = new_parcel_labels
all_subj_var['parcel'] = new_parcel_labels
all_subj_cov['parcel'] = new_parcel_labels

all_subj_var = all_subj_var.sel(parcel=Adot_parcel.parcel.values)
all_subj_hrf = all_subj_hrf.sel(parcel=Adot_parcel.parcel.values)
all_subj_cov = all_subj_cov.sel(parcel=Adot_parcel.parcel.values)

all_subj_hrf_mag = all_subj_hrf.sel(time=slice(t_win[0], t_win[1])).mean('time')
all_subj_var_mag = all_subj_var.sel(time=slice(t_win[0], t_win[1])).sum('time') / all_subj_var.sel(time=slice(t_win[0], t_win[1])).sizes['time']**2
all_subj_cov_mag = all_subj_cov.sel(time=slice(t_win[0], t_win[1])).sum('time') / all_subj_cov.sel(time=slice(t_win[0], t_win[1])).sizes['time']**2

diff_mag = all_subj_hrf_mag.sel(trial_type = 'mnt-correct-in') - all_subj_hrf_mag.sel(trial_type = 'mnt-correct-out')
diff_var = all_subj_var_mag.sel(trial_type = 'mnt-correct-in') + all_subj_var_mag.sel(trial_type = 'mnt-correct-out') - 2 * all_subj_cov_mag

mri_mean, mri_tval, _, _ = gcpt.get_weighted_group_average(diff_mag, diff_var) 

n_vertices = len(Adot.vertex)
foo_img_v = xr.DataArray(np.full([2, n_vertices], np.nan), 
                                dims=['chromo', 'vertex'],
                                coords = {'parcel': ('vertex', Adot.parcel.values),
                                        'is_brain': ('vertex', Adot.is_brain.values), 
                                        'chromo': ['HbO', 'HbR'],
                                        # 'trial_type': trials, 
                                        # 'time': image_results['X_hrf_ts'].time
                                        }
                                )      
for pp in Adot_parcel.parcel.values:
    if pp == 'scalp': 
        continue
    mask = vertex_parcel_labels == pp
    foo_img_v.loc[:,mask] = mri_tval.sel(parcel=pp).squeeze()

image_recon_multi_view(
        X_ts = foo_img_v,
        head = head,
        cmap = 'jet',
        clim = [-6,6],
        view_type = 'hbo_brain',
        title_str = 'HbO T-stat: in-out',
        filename = None,
        SAVE = False,
        wdw_size = (1300, 768)
    )

if SPLIT_HEMISPHERE:
    network_labels = [p.split('_')[0] + '_' + p.split('_')[-1] for p in diff_mag.parcel.values]
else:
    network_labels = [p.split('_')[0] for p in diff_mag.parcel.values]

all_subj_hrf_mag['parcel'] = network_labels
all_subj_hrf_mag_net = all_subj_hrf_mag.groupby('parcel').mean('parcel')
all_subj_var_mag['parcel'] = network_labels
all_subj_var_mag_net = all_subj_var_mag.groupby('parcel').sum('parcel') / all_subj_var_mag.groupby('parcel').count()**2
all_subj_cov_mag['parcel'] = network_labels
all_subj_cov_mag_net = all_subj_cov_mag.groupby('parcel').sum('parcel') / all_subj_cov_mag.groupby('parcel').count()**2

all_subj_hrf_mag_net['trial_type'] = all_subj_hrf_mag.trial_type
all_subj_var_mag_net['trial_type'] = all_subj_hrf_mag.trial_type

diff_net_mag = all_subj_hrf_mag_net.sel(trial_type='mnt-correct-in') - all_subj_hrf_mag_net.sel(trial_type='mnt-correct-out')
diff_net_var = all_subj_var_mag_net.sel(trial_type='mnt-correct-in') + all_subj_var_mag_net.sel(trial_type='mnt-correct-out') - 2*all_subj_cov_mag_net

mri_mean_net, mri_tval_net, _, _ = gcpt.get_weighted_group_average(diff_net_mag, diff_net_var)

networks = np.array([
    n for n in mri_tval_net.parcel.values
    if _base_network(n) not in PARCELS_NOT_SENSITIVE
])

mri_tval_map = mri_mean_net.reindex(parcel=networks).sel(chromo='HbO').values.squeeze()

#%% permutation test correlation between maps 
# B = 5000
# corr_null = []
# CORR_IMG = 'tval'
# spatial_corr = stats.pearsonr(mri_tval.sel(chromo='HbO').real, nirs_tval.sel(chromo='HbO').real)

# all_subj_hrf = nirs_results['X_hrf_ts'].sel(parcel=Adot_parcel.parcel.values)
# all_subj_var = nirs_results['X_mse_corr'].sel(parcel=Adot_parcel.parcel.values)

# all_subj_hrf_mag = all_subj_hrf.sel(time=slice(5,8)).mean('time')
# all_subj_var_mag = all_subj_var.sel(time=slice(5,8)).sum('time') / all_subj_var.sel(time=slice(5,8)).sizes['time']**2

# diff_mag = all_subj_hrf_mag.sel(trial_type = 'mnt-correct-in') - all_subj_hrf_mag.sel(trial_type = 'mnt-correct-out')
# diff_var = all_subj_var_mag.sel(trial_type = 'mnt-correct-in') + all_subj_var_mag.sel(trial_type = 'mnt-correct-out')

# data = diff_mag.sel(chromo='HbO').sortby('parcel')
# mse = diff_var.sel(chromo='HbO').sortby('parcel')

# spatial_dim = 'parcel'
# n_subjects = len(data.subj)

# for b in tqdm(range(B)):

#     # instead shuffle the parcels in the data
#     # shuffled_idx = np.random.permutation(len(x_fnirs.parcel))
#     # shuffled_nirs = x_fnirs.isel(parcel=shuffled_idx)

#     signs = np.random.choice([-1, 1], size=n_subjects).reshape(-1, 1) 
#     data_perm = data * signs # randomly flip signs of data for current permutation
#     mse_perm = mse.copy()

#     subj_mean, tstat, _, _ = gcpt.get_weighted_group_average(data_perm, mse_perm)

#     if CORR_IMG == 'mag':
#         corr = stats.pearsonr(x_fmri.values,  subj_mean.values.real)
#     else:
#         corr = stats.pearsonr(mri_tval.sel(chromo='HbO').real.values,  tstat.values.real)

#     corr_null.append(corr.statistic)

# fig, ax = plt.subplots(1,1, figsize=[5, 5])
# pval = np.mean(abs(np.asarray(corr_null)) >= abs(spatial_corr.statistic))
# ax.set_title(f'Correlating using {CORR_IMG}: true corr = {spatial_corr.statistic:.3f}')
# ax.hist(corr_null, bins=100)
# ax.axvline(spatial_corr.statistic, color='k', ls='--')
# ax.text(0.1, 80, f'p-val =\n {pval:.2e}')

# %%
# ── Per-network colour map ─────────────────────────────────────────────────
# Colour by *base* network so LH and RH share the same hue.
base_nets   = sorted(set(_base_network(n) for n in networks))
base_cmap   = plt.cm.get_cmap('tab20', len(base_nets))
base_colors = {net: base_cmap(i) for i, net in enumerate(base_nets)}

# For hemisphere split: LH = full opacity, RH = slightly lighter
def net_color(label: str, alpha_mod: float = 1.0):
    bc = np.array(base_colors[_base_network(label)])
    if SPLIT_HEMISPHERE and label.endswith('_RH'):
        # lighten RH by blending with white
        bc[:3] = 0.55 * bc[:3] + 0.45
    return tuple(bc)

network_colors = {n: net_color(n) for n in networks}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_group_label(parcel_name: str, split_hemi: bool) -> str:
    """
    Return the grouping label for a parcel name.

    Parcel names are expected to follow the Schaefer/Kong convention:
        <Network>_<Hemi>_<Index>  e.g. "Cont_LH_1", "SomMot_RH_3"

    split_hemi=False  →  "Cont"          (first token only, original behaviour)
    split_hemi=True   →  "Cont_LH"       (first + hemisphere token)

    Falls back gracefully when the name does not contain a hemisphere token.
    """
    tokens = parcel_name.split('_')
    if not split_hemi:
        return tokens[0]

    # Find the hemisphere token (LH / RH) anywhere in the name
    hemi = None
    for tok in tokens[1:]:
        if tok in ('LH', 'RH'):
            hemi = tok
            break

    if hemi is None:
        # No hemisphere information — treat as bilateral
        return tokens[0]
    return f"{tokens[0]}_{hemi}"
# ══════════════════════════════════════════════════════════════════════════
# Scatter plot
# ══════════════════════════════════════════════════════════════════════════
fmri_map = mri_tval.sel(chromo='HbO')
fnirs_map = nirs_tval.sel(chromo='HbO')
fmri_vals = mri_tval_map #.sel(chromo='HbO')
fnirs_vals = nirs_tval_map #.sel(chromo='HbO')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# -- Parcel-level cloud ----------------------------------------------------
for net in networks:
    parcel_mask = np.array([
        _get_group_label(p, SPLIT_HEMISPHERE) == net
        for p in fmri_map.parcel.values
    ])
    if not parcel_mask.any():
        continue
    ax.scatter(fmri_map.values[parcel_mask],
                fnirs_map.values[parcel_mask],
                color=network_colors[net],
                s=30, alpha=0.25, edgecolors='none', zorder=2)

# -- Network-level centroids -----------------------------------------------
for i, net in enumerate(networks):
    ax.scatter(fmri_vals[i], fnirs_vals[i],
                color=network_colors[net],
                s=120, alpha=0.95,
                edgecolors='white', linewidths=0.8, zorder=4)

    # Label: for split hemi, show e.g. "Cont_LH"; otherwise just "Cont"
    # ax.annotate(net,
    #             xy=(fmri_vals[i], fnirs_vals[i]),
    #             xytext=(5, 4), textcoords='offset points',
    #             fontsize=7, color=network_colors[net])

# -- LH↔RH connector arrows (only when hemisphere split is active) ---------
if SPLIT_HEMISPHERE:
    # Pair up LH and RH centroids that share the same base network
    lh_idx = {_base_network(n): i for i, n in enumerate(networks)
                if n.endswith('_LH')}
    rh_idx = {_base_network(n): i for i, n in enumerate(networks)
                if n.endswith('_RH')}
    for base in set(lh_idx) & set(rh_idx):
        li, ri = lh_idx[base], rh_idx[base]
        ax.annotate(
            '', xy=(fmri_vals[ri], fnirs_vals[ri]),
            xytext=(fmri_vals[li], fnirs_vals[li]),
            arrowprops=dict(
                arrowstyle='-',
                color=base_colors[base],
                lw=1.0, alpha=0.45,
                connectionstyle='arc3,rad=0.0'
            ), zorder=3
        )

# -- Reference lines -------------------------------------------------------
ax.axhline(0, color='#b0bec8', lw=0.8, ls=':', zorder=1)
ax.axvline(0, color='#b0bec8', lw=0.8, ls=':', zorder=1)

# -- Regression line (fit to network centroids) ----------------------------
slope, intercept, r, p_val, _ = stats.linregress(fmri_vals, fnirs_vals)
x_line = np.linspace(-4, 6, 100)
ax.plot(x_line, slope * x_line + intercept,
        color='#334155', lw=1.5, ls='--', zorder=2)

p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
ax.text(0.97, 0.05, f'r = {r:.2f}',
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='#e2e8f0',
                    boxstyle='round,pad=0.3'))

# -- Formatting ------------------------------------------------------------
# ax.set_xlim([-4, 8])
# ax.set_ylim([-4, 3])
ax.set_xlabel('fMRI t-stat')
ax.set_ylabel('fNIRS t-stat')
ax.grid(True, lw=0.7, color='#edf0f4')
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

# Legend: one entry per base network (colour only, no LH/RH duplication)
legend_patches = [
    mpatches.Patch(color=base_colors[bn], label=bn)
    for bn in base_nets
    if bn not in PARCELS_NOT_SENSITIVE
]
if SPLIT_HEMISPHERE:
    legend_patches += [
        mpatches.Patch(facecolor='#FFFFFF', label='● LH  ○ RH (lighter)'),
    ]
ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
            ncol=2, frameon=True, framealpha=0.9,
            loc='upper left', borderpad=0.6,
            handlelength=1, handleheight=1)

# plt.tight_layout()
plt.show()
# %%
# ══════════════════════════════════════════════════════════════════════════
# Bar chart
# ══════════════════════════════════════════════════════════════════════════
FMRI_COLOR  = '#2563EB'
FNIRS_COLOR = '#F97316'
PLOT_FNIRS = True
def _draw_bar_panel(ax, nets, fmri_v, fnirs_v, panel_title,
                    bar_colors, show_ylabel=True):
    """Draw a single fMRI/fNIRS bar panel onto *ax*, sorted by fMRI."""
    sort_idx     = np.argsort(fmri_v)[::-1]
    nets_s       = nets #[sort_idx]
    fmri_s       = fmri_v #[sort_idx]
    fnirs_s      = fnirs_v #[sort_idx]

    x     = np.arange(len(nets_s))
    WIDTH = 0.38

    ax.bar(x - WIDTH / 2, fmri_s,  WIDTH,
            color='blue',
            alpha=0.92, zorder=3, linewidth=0.6, edgecolor='white',
            label='fMRI')
    if PLOT_FNIRS:
        ax.bar(x + WIDTH / 2, fnirs_s, WIDTH,
                color='orange',
                alpha=0.55, zorder=3, linewidth=0.6, edgecolor='white',
                label='fNIRS', hatch='')

    # Overlay solid fMRI / hatched fNIRS using the global modality colours
    # as a thin coloured edge so the two modalities remain distinguishable.
    ax.bar(x - WIDTH / 2, fmri_s,  WIDTH,
            color=FMRI_COLOR, alpha=0.50, zorder=3,
            linewidth=0, edgecolor='none')
    if PLOT_FNIRS:
        ax.bar(x + WIDTH / 2, fnirs_s, WIDTH,
                color=FNIRS_COLOR, alpha=0.50, zorder=3,
                linewidth=0, edgecolor='none')

    ax.axhline(0, color='#94A3B8', linewidth=0.9, zorder=2)
    ax.yaxis.grid(True, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_base_network(n) for n in nets_s],
        rotation=45, ha='right'
    )
    if show_ylabel:
        ax.set_ylabel('t-statistic')
    ax.set_title(panel_title,fontweight='600')
    # ax.text(0.99, 0.99, 'sorted by fMRI \u2193',
    #         transform=ax.transAxes, ha='right', va='top',
    #         color='#94a3b8')

fmri_map_net_norm = fmri_vals #/ fmri_vals.std()
fnirs_map_net_norm = fnirs_vals #/ fnirs_vals.std()

if SPLIT_HEMISPHERE:
    # ── Split data into LH and RH subsets ─────────────────────────────────
    lh_mask   = np.array([n.endswith('_LH') for n in networks])
    rh_mask   = np.array([n.endswith('_RH') for n in networks])
    # Also keep bilaterals (no _LH/_RH) in both panels if any
    bil_mask  = ~lh_mask & ~rh_mask

    lh_nets   = networks[lh_mask | bil_mask]
    rh_nets   = networks[rh_mask | bil_mask]
    lh_fmri   = fmri_map_net_norm[lh_mask  | bil_mask]
    lh_fnirs  = fnirs_map_net_norm[lh_mask | bil_mask]
    rh_fmri   = fmri_map_net_norm[rh_mask  | bil_mask]
    rh_fnirs  = fnirs_map_net_norm[rh_mask | bil_mask]

    # Shared y-axis limits so both panels are directly comparable
    all_vals  = np.concatenate([lh_fmri, lh_fnirs, rh_fmri, rh_fnirs])
    y_margin  = 0.12 * (all_vals.max() - all_vals.min())
    ylim      = (all_vals.min() - y_margin, all_vals.max() + y_margin)

    fig, (ax_lh, ax_rh) = plt.subplots(
        1, 2, figsize=(18, 9),
        sharey=True,
        gridspec_kw={'wspace': 0.08}
    )

    _draw_bar_panel(ax_lh, lh_nets, lh_fmri, lh_fnirs,
                    'Left Hemisphere', network_colors, show_ylabel=True)
    _draw_bar_panel(ax_rh, rh_nets, rh_fmri, rh_fnirs,
                    'Right Hemisphere', network_colors, show_ylabel=False)

    ax_lh.set_ylim(ylim)

    # Shared legend on the right panel
    legend_handles = [
        mpatches.Patch(color=FMRI_COLOR,  alpha=0.85, label='fMRI'),
        mpatches.Patch(color=FNIRS_COLOR, alpha=0.85, label='fNIRS'),
    ]
    # ax_rh.legend(handles=legend_handles,
    #                 frameon=True, framealpha=0.95, fontsize=12,
    #                 loc='lower left')

    # if title:
    #     fig.suptitle(title, fontsize=14, fontweight='600', y=1.01)

else:
    # ── Original single-panel bar chart ───────────────────────────────────
    sort_idx     = np.argsort(fmri_vals)[::-1]
    nets_sorted  = networks #[sort_idx]
    fmri_sorted  = fmri_vals#[sort_idx]
    fnirs_sorted = fnirs_vals#[sort_idx]

    fig, ax = plt.subplots(figsize=(14, 10))
    x     = np.arange(len(networks))
    WIDTH = 0.38

    ax.bar(x - WIDTH / 2, fmri_sorted,  WIDTH, label='fMRI',
            color=FMRI_COLOR,  alpha=0.92, zorder=3,
            linewidth=0.6, edgecolor='white')
    if PLOT_FNIRS:
        ax.bar(x + WIDTH / 2, fnirs_sorted, WIDTH, label='fNIRS',
                color=FNIRS_COLOR, alpha=0.92, zorder=3,
                linewidth=0.6, edgecolor='white')

    ax.axhline(0, color='#94A3B8', linewidth=0.9, zorder=2)
    ax.yaxis.grid(True, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticks(x)
    ax.set_xticklabels(nets_sorted, rotation=45, ha='right')
    ax.set_ylabel('t-statistic')
    # ax.set_title(title)
    # ax.text(0.99, 0.99, 'sorted by fMRI \u2193', transform=ax.transAxes,
    #         ha='right', va='top', fontsize=10, color='#94a3b8')
    # ax.legend(
    #     handles=[
    #         mpatches.Patch(color=FMRI_COLOR,  label='fMRI'),
    #         mpatches.Patch(color=FNIRS_COLOR, label='fNIRS'),
    #     ],
    #     frameon=True, framealpha=0.95, loc='lower left'
    # )

plt.tight_layout()
plt.show()


#%% Plot networks
import numpy as np
import xarray as xr
import pyvista as pv

import cedalion
import cedalion.data
import cedalion.dot
import cedalion.dataclasses as cdc
import cedalion.vis.blocks as vbx
import matplotlib.pyplot as plt
# Use 'server' for interactive 3-D rotation in a local Jupyter session.
# Use 'static' for rendered documentation or environments without a display.
pv.set_jupyter_backend('static')

xr.set_options(display_expand_data=False);

savedir =  '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/PAPER_FIGS/FIG1/'

# Build a per-vertex colour list from the parcel labels on the brain surface.
head_ijk  = cedalion.dot.get_standard_headmodel("icbm152")
hmfiles = cedalion.data.get_icbm152_headmodel_files()
# Load the RGB colour map for the Schaefer parcellation.
parcel_colors = hmfiles.load_parcel_colors()

# Each entry is a parcel name -> [R, G, B] list (0-255 range).
# Show a small sample.
dict(list(parcel_colors.items())[:5])

vertex_colors = [
    parcel_colors.get(pc, [180, 180, 180])
    for pc in head_ijk.brain.vertices.parcel.values
]
# --- Build a legend showing each of the 17 networks and its color ---
# Parcel names encode the network as the leading token, e.g.
# "VisCent_Striate_1_LH" -> network "VisCent". Individual parcels within a
# network share (nearly) the same color, so average them for one swatch.
network_colors = {}
for parcel_name, rgb in parcel_colors.items():
    network = parcel_name.split('_')[0]
    network_colors.setdefault(network, []).append(rgb)
network_colors = {
    net: np.mean(rgbs, axis=0) / 255
    for net, rgbs in network_colors.items()
}

# Canonical Yeo/Schaefer 17-network order, with readable labels.
network_order = [
    ('VisCent', 'Visual Central'),
    ('VisPeri', 'Visual Peripheral'),
    ('SomMotA', 'Somatomotor A'),
    ('SomMotB', 'Somatomotor B'),
    ('DorsAttnA', 'Dorsal Attention A'),
    ('DorsAttnB', 'Dorsal Attention B'),
    ('SalVentAttnA', 'Ventral Attention A'),
    ('SalVentAttnB', 'Ventral Attention B'),
    ('LimbicB', 'Limbic B'),
    ('LimbicA', 'Limbic A'),
    ('ContC', 'Control C'),
    ('ContA', 'Control A'),
    ('ContB', 'Control B'),
    ('TempPar', 'Temporal Parietal'),
    ('DefaultC', 'Default C'),
    ('DefaultA', 'Default A'),
    ('DefaultB', 'Default B'),
]

fig_legend, ax_legend = plt.subplots(figsize=(3, 5))
ax_legend.axis('off')
handles = [
    plt.Line2D([0], [0], marker='s', linestyle='', markersize=12,
               markerfacecolor=network_colors[net], markeredgecolor='none')
    for net, _ in network_order
]
labels = [label for _, label in network_order]
ax_legend.legend(handles, labels, loc='center', frameon=False, title='17 Networks')
fig_legend.tight_layout()
fig_legend.savefig(savedir + 'parcellation_legend.png', dpi=300)
# plt.close(fig_legend)

# plt = pv.Plotter()
# vbx.plot_surface(plt, head_ijk.brain, color=vertex_colors)
# plt.show()
views_positions = {
    # 'scale_bar': (1, 1),
    'left': (0, 0),
    'superior': (0, 1),
    'right': (0, 2),
    'anterior': (1, 0),
    'posterior': (1, 2)
    }

positions = {
    'superior': [0, 0, 1],
    'left': [-1, 0, 0],
    'right': [1, 0, 0],
    'anterior': [0, 1, 0],
    'posterior': [0, -1, 0],
    # 'scale_bar': [0, 0, 1]
}


# title_str = 'Sensitivity Profile'
surf = cdc.VTKSurface.from_trimeshsurface(head_ijk.brain)
surf = pv.wrap(surf.mesh)
surf["scalars"] = vertex_colors
kwargs = {}
kwargs["scalars"] = "scalars"
color = None
rgb = True
centroid = np.mean(surf.points, axis=0)

for view in positions.keys():
    p0 = pv.Plotter(
        shape=(1,1),
        window_size=[300,300],
        off_screen=True,
    )
    p0.subplot(0,0)
    
    # p0.add_mesh(surf, color=vertex_colors, rgb=False,
    #                     show_scalar_bar=False, 
    #                     smooth_shading=True, interpolate_before_map=False)
    p0.add_mesh(surf, color=color, rgb=rgb, opacity=1, smooth_shading=True, **kwargs)

    view_up = [0, 1, 0] if view == 'superior' else [0, 0, 1]
    camera_direction = positions.get(view, [0, 0, 1])
    p0.camera_position = [
        centroid + np.array(camera_direction) * 400,
        centroid,
        view_up,
    ]
    # p0.screenshot(savedir + f'parcellation_{view}.png')
    p0.show()
