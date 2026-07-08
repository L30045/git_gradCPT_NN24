"""
Cross-subject EEG ERP and ERSP analysis pipeline.
Results are organised by analysis method in xSubj_results/<analysis>/.
"""
#%% load library
import numpy as np
import datetime
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from tqdm import tqdm
import pickle
import gzip
import glob
import time
import sys
from spectral_connectivity import Multitaper, Connectivity
from spectral_connectivity.transforms import prepare_time_series

#%% output directory — one sub-folder per analysis method
results_root = os.path.join("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg",
                            "xSubj_results")

analysis_dirs = {
    'RT_inzone_vs_outzone':   os.path.join(results_root, 'RT_inzone_vs_outzone'),
    'ERP_city_vs_mnt':        os.path.join(results_root, 'ERP_city_vs_mnt'),
    'ERPImage':               os.path.join(results_root, 'ERPImage'),
    'ERSP_raw':               os.path.join(results_root, 'ERSP_raw'),
    'ERSP_baseline':          os.path.join(results_root, 'ERSP_baseline'),
    'ERSP_mnt_vs_city':       os.path.join(results_root, 'ERSP_mnt_vs_city'),
    'ERP_inzone_vs_outzone':  os.path.join(results_root, 'ERP_inzone_vs_outzone'),
    'ERSP_inzone_outzone':    os.path.join(results_root, 'ERSP_inzone_outzone'),
    'PSD':                    os.path.join(results_root, 'PSD'),
}
for d in analysis_dirs.values():
    os.makedirs(d, exist_ok=True)

is_save_fig = False
if is_save_fig:
    matplotlib.use('Agg')

def save_or_show_fig(fig, save_path):
    """Save fig to save_path if is_save_fig else display it interactively."""
    if is_save_fig:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

#%% preprocessing parameter setting
subj_id_array = sorted([
    int(d.split('-')[1])
    for d in os.listdir(data_save_path)
    if d.startswith('sub-') and os.path.isdir(os.path.join(data_save_path, d))
])

ch_names = ['fz','cz','pz','oz']
split_zone_crit = 'react'
is_bpfilter = True
bp_f_range = [0.1, 45]
is_reref = True
reref_ch = ['tp9h','tp10h']
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(eeg=100e-6)
is_detrend = 1
is_overwrite = False

preproc_params = dict(
    is_bpfilter=is_bpfilter,
    bp_f_range=bp_f_range,
    is_reref=is_reref,
    reref_ch=reref_ch,
    is_ica_rmEye=is_ica_rmEye,
    select_event=select_event,
    baseline_length=baseline_length,
    epoch_reject_crit=epoch_reject_crit,
    is_detrend=is_detrend,
    ch_names=ch_names,
    is_overwrite=is_overwrite
)

#%% load preprocessed EEG for all subjects
excluded_subjects = {}  # key: "sub-xxx", value: reason string
subj_EEG_dict = dict()
rm_ch_dict = dict()
for subj_id in tqdm(subj_id_array):
    gz_path = os.path.join(data_save_path, f"sub-{subj_id}", f"sub-{subj_id}_preprocessed_dict.pkl.gz")
    if not os.path.exists(gz_path):
        print(f"sub-{subj_id}: preprocessed dict not found at {gz_path}, skipping.")
        excluded_subjects[f"sub-{subj_id}"] = "Preprocessed data file (.pkl.gz) not found"
        continue
    with gzip.open(gz_path, 'rb') as f:
        _payload = pickle.load(f)
    subj_EEG_dict[f"sub-{subj_id}"] = _payload["EEG"]
    rm_ch_dict[f"sub-{subj_id}"]    = _payload["rm_ch"]
    print(f"sub-{subj_id}: loaded runs {list(_payload['EEG'].keys())}")

#%% epoch all subjects
subj_epoch_dict = dict()
subj_vtc_dict = dict()
subj_react_dict = dict()
for key_name in tqdm(subj_EEG_dict.keys()):
    print(f"Epoching {key_name}")
    single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = \
        eeg_epoch_subj_level(key_name, subj_EEG_dict[key_name], preproc_params)
    subj_epoch_dict[key_name] = single_subj_epoch_dict
    subj_vtc_dict[key_name]   = single_subj_vtc_dict
    subj_react_dict[key_name] = single_subj_react_dict

#%% Combined runs — compute per-subject median threshold
combine_epoch_dict = dict()
combine_vtc_dict   = dict()
combine_react_dict = dict()
in_out_zone_dict   = dict()

# per-subject median split threshold
subj_thres_zone = {}
for key_name in subj_epoch_dict.keys():
    avail_runs = list(subj_epoch_dict[key_name].keys())
    match split_zone_crit:
        case 'vtc':
            vals_list = [subj_vtc_dict[key_name][run][ev]
                         for run in avail_runs
                         for ev in event_labels_lookup.keys()
                         if not ev.endswith("_response")
                         and len(subj_vtc_dict[key_name][run][ev]) > 0]
        case 'react':
            vals_list = [subj_react_dict[key_name][run][ev]
                         for run in avail_runs
                         for ev in event_labels_lookup.keys()
                         if not ev.endswith("_response")
                         and len(subj_react_dict[key_name][run][ev]) > 0]
    if len(vals_list) == 0:
        print(f"  No valid trials found for {key_name}, excluding from cross-subject analysis.")
        excluded_subjects[key_name] = (
            f"No valid non-response trials found after epoching "
            f"(available runs: {avail_runs}; split criterion: {split_zone_crit})"
        )
        continue
    subj_thres_zone[key_name] = np.median(np.concatenate(vals_list))

# build combined epoch / vtc / react / zone dicts
for select_event in event_labels_lookup.keys():
    epoch_dict         = {ch: [] for ch in ch_names}
    vtc_dict           = {ch: [] for ch in ch_names}
    react_dict         = {ch: [] for ch in ch_names}
    ch_in_out_zone_dict = {ch: [] for ch in ch_names}

    for key_name in subj_epoch_dict.keys():
        if key_name not in subj_thres_zone:
            continue  # already excluded above
        avail_runs = list(subj_epoch_dict[key_name].keys())
        tmp_epoch_list  = []
        tmp_vtc_list    = []
        tmp_react_list  = []
        tmp_zone_list   = []
        for run in avail_runs:
            loc_e = subj_epoch_dict[key_name][run][select_event]
            loc_v = subj_vtc_dict[key_name][run][select_event]
            loc_r = subj_react_dict[key_name][run][select_event]
            if len(loc_e) > 0:
                tmp_epoch_list.append(loc_e)
                tmp_vtc_list.append(loc_v)
                tmp_react_list.append(loc_r)
                tmp_zone_list.append(loc_v < subj_thres_zone[key_name])

        for ch in ch_names:
            if len(tmp_epoch_list) == 0:
                epoch_dict[ch].append([])
                vtc_dict[ch].append([])
                react_dict[ch].append([])
                ch_in_out_zone_dict[ch].append([])
                continue
            ch_picked = [x.copy().pick(ch) for x in tmp_epoch_list if ch in x.ch_names]
            if len(ch_picked) == 0:
                epoch_dict[ch].append([])
                vtc_dict[ch].append([])
                react_dict[ch].append([])
                ch_in_out_zone_dict[ch].append([])
                continue
            epoch_dict[ch].append(mne.concatenate_epochs(ch_picked, verbose=False))
            vtc_dict[ch].append(np.concatenate([v for v,e in zip(tmp_vtc_list,   tmp_epoch_list) if ch in e.ch_names]))
            react_dict[ch].append(np.concatenate([r for r,e in zip(tmp_react_list, tmp_epoch_list) if ch in e.ch_names]))
            ch_in_out_zone_dict[ch].append(np.concatenate([z for z,e in zip(tmp_zone_list,  tmp_epoch_list) if ch in e.ch_names]))

    combine_epoch_dict[select_event] = epoch_dict
    combine_vtc_dict[select_event]   = vtc_dict
    combine_react_dict[select_event] = react_dict
    in_out_zone_dict[select_event]   = ch_in_out_zone_dict

#%% Remove subjects with too few epochs preserved
# track which subjects remain after epoch QC
included_subj_keys = [k for k in subj_epoch_dict.keys() if k in subj_thres_zone]
n_subj_before = len(included_subj_keys)

check_ch = 'cz'
epoch_count = []
for ev in combine_epoch_dict.keys():
    tmp_count = [len(x) for x in combine_epoch_dict[ev][check_ch]]
    if len(tmp_count) > 0:
        epoch_count.append(tmp_count)
epoch_count = np.sum(np.vstack(epoch_count), axis=0)

print("Target number of epoch = 2700 (450 epochs * 3 runs * 2 time-lock)")
print(f"Epoch counts per subject: {epoch_count}")
rm_subj_mask = epoch_count < 0.5 * 2700
keep_subj_mask = ~rm_subj_mask

for i, key_name in enumerate(included_subj_keys):
    if rm_subj_mask[i]:
        excluded_subjects[key_name] = (
            f"Too few epochs preserved after artifact rejection "
            f"({epoch_count[i]:.0f} < 50% of target 2700)"
        )
        print(f"  Removing {key_name}: only {epoch_count[i]:.0f} epochs")

# apply mask
for ev in combine_epoch_dict.keys():
    for ch in combine_epoch_dict[ev].keys():
        combine_epoch_dict[ev][ch] = [x for i,x in enumerate(combine_epoch_dict[ev][ch]) if keep_subj_mask[i]]
for ev in combine_vtc_dict.keys():
    for ch in combine_vtc_dict[ev].keys():
        combine_vtc_dict[ev][ch] = [x for i,x in enumerate(combine_vtc_dict[ev][ch]) if keep_subj_mask[i]]
for ev in combine_react_dict.keys():
    for ch in combine_react_dict[ev].keys():
        combine_react_dict[ev][ch] = [x for i,x in enumerate(combine_react_dict[ev][ch]) if keep_subj_mask[i]]
for ev in in_out_zone_dict.keys():
    for ch in in_out_zone_dict[ev].keys():
        in_out_zone_dict[ev][ch] = [x for i,x in enumerate(in_out_zone_dict[ev][ch]) if keep_subj_mask[i]]

n_subjects = int(keep_subj_mask.sum())
print(f"Subjects remaining for cross-subject analysis: {n_subjects}/{n_subj_before}")

#%% Analysis 1 — RT in-zone vs out-of-zone (city_correct, CZ)
check_ch = 'cz'
ev_city = 'city_correct'
in_zone_RT  = [x[y]  for x,y in zip(combine_react_dict[ev_city][check_ch], in_out_zone_dict[ev_city][check_ch]) if len(x) > 0]
out_zone_RT = [x[~y] for x,y in zip(combine_react_dict[ev_city][check_ch], in_out_zone_dict[ev_city][check_ch]) if len(x) > 0]
rt_diff_dist = [1000*(np.mean(x)-np.mean(y)) for x,y in zip(in_zone_RT, out_zone_RT)]
print(f'RT diff (in/out zone) = {np.mean(rt_diff_dist):.2f} ms')

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(rt_diff_dist, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(rt_diff_dist), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(rt_diff_dist):.2f} ms')
ax.set_xlabel('RT Difference (in-zone - out-of-zone) [ms]')
ax.set_ylabel('Frequency')
ax.set_title(f'Cross-subject RT Differences ({check_ch.upper()}, n={len(rt_diff_dist)})')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_or_show_fig(fig, os.path.join(analysis_dirs['RT_inzone_vs_outzone'], f'xSubj_RT_inzone_vs_outzone_{check_ch}.png'))

#%% Analysis 2 — ERP: city vs mountain (all channels)
vis_events = ['city_correct', 'mnt_correct']
colors = ['b', 'r']
vis_ch = ch_names

condition_data = {}
for sel_ev in vis_events:
    condition_data[sel_ev] = {}
    for ch in vis_ch:
        xSubj_erps = [ep.average().data for ep in combine_epoch_dict[sel_ev][ch] if len(ep) > 0]
        if len(xSubj_erps) == 0:
            continue
        xSubj_erps = np.vstack(xSubj_erps)
        condition_data[sel_ev][ch] = {'erps': xSubj_erps, 'n_subjects': xSubj_erps.shape[0]}

for ch in vis_ch:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, sel_ev in enumerate(vis_events):
        if ch not in condition_data.get(sel_ev, {}):
            continue
        plt_erps = condition_data[sel_ev][ch]['erps']
        n = condition_data[sel_ev][ch]['n_subjects']
        mean_erp = np.mean(plt_erps, axis=0)
        sem_erp  = np.std(plt_erps, axis=0) / np.sqrt(n)
        t_ms = combine_epoch_dict[sel_ev][ch][0].times * 1000
        label = 'City' if sel_ev.startswith('city') else 'Mountain'
        ax.plot(t_ms, mean_erp, color=colors[idx], linewidth=2, label=f'{label} (n={n})')
        ax.fill_between(t_ms, mean_erp - 2*sem_erp, mean_erp + 2*sem_erp, alpha=0.3, color=colors[idx])
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(f'Cross-subject ERP City vs Mountain — {ch.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERP_city_vs_mnt'], f'xSubj_ERP_city_vs_mnt_{ch}.png'))

#%% Analysis 3 — ERP Image sorted by VTC (mnt_correct, CZ)
erp_image_event = 'mnt_correct'
erp_image_ch    = 'cz'
plt_epoch = mne.concatenate_epochs([x for x in combine_epoch_dict[erp_image_event][erp_image_ch] if len(x) > 0])
time_vector = plt_epoch.times
plt_epoch_data = np.squeeze(plt_epoch.get_data())
window_size = np.max([4, np.floor(plt_epoch_data.shape[0] * 0.01).astype(int)])
plt_vtc   = np.concatenate(combine_vtc_dict[erp_image_event][erp_image_ch])
plt_react = np.concatenate(combine_react_dict[erp_image_event][erp_image_ch])
title_txt = f'Cross-subject — {erp_image_event} — {erp_image_ch.upper()}'

fig = plt_ERPImage(time_vector, plt_epoch_data,
                   sort_idx=plt_vtc,
                   smooth_window_size=window_size,
                   clim=[-10e-6, 10e-6],
                   title_txt=title_txt,
                   ref_onset=plt_react)
save_or_show_fig(fig, os.path.join(analysis_dirs['ERPImage'], f'xSubj_ERPImage_{erp_image_event}_{erp_image_ch}.png'))

#%% Analysis 4 — ERSP raw power (mnt_correct, CZ)
ersp_event = 'mnt_correct'
ersp_ch    = 'cz'
time_halfbandwidth_product = 1
time_window_duration_ersp  = 0.2
time_window_step_ersp      = 0.05

start_time = time.time()
plt_epoch = mne.concatenate_epochs(combine_epoch_dict[ersp_event][ersp_ch])
(_, multitaper, _) = plt_multitaper(plt_epoch,
                                    time_halfbandwidth_product=time_halfbandwidth_product,
                                    time_window_duration=time_window_duration_ersp,
                                    time_window_step=time_window_step_ersp)
fig = plt.gcf()
fig.suptitle(f'Cross-subject ERSP (raw) — {ersp_event} — {ersp_ch.upper()}')
save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_raw'], f'xSubj_ERSP_raw_{ersp_event}_{ersp_ch}.png'))
print(f"ERSP (raw): freq res = {multitaper.frequency_resolution:.2f} Hz  ({time.time()-start_time:.1f}s)")

#%% Analysis 5 — ERSP ratio to baseline (mnt_correct, CZ)
(_, multitaper, _) = plt_multitaper(plt_epoch,
                                    time_halfbandwidth_product=time_halfbandwidth_product,
                                    time_window_duration=time_window_duration_ersp,
                                    time_window_step=time_window_step_ersp,
                                    ratio_to="baseline")
fig = plt.gcf()
fig.suptitle(f'Cross-subject ERSP (baseline ratio) — {ersp_event} — {ersp_ch.upper()}')
save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_baseline'], f'xSubj_ERSP_baseline_{ersp_event}_{ersp_ch}.png'))

#%% Analysis 6 — ERSP mnt vs city ratio (CZ)
ref_event = 'city_correct'
plt_epoch_ref = mne.concatenate_epochs(combine_epoch_dict[ref_event][ersp_ch])
(_, multitaper, _) = plt_multitaper(plt_epoch,
                                    time_halfbandwidth_product=time_halfbandwidth_product,
                                    time_window_duration=time_window_duration_ersp,
                                    time_window_step=time_window_step_ersp,
                                    ratio_to=plt_epoch_ref)
fig = plt.gcf()
fig.suptitle(f'Cross-subject ERSP (mnt/city ratio) — {ersp_ch.upper()}')
save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_mnt_vs_city'], f'xSubj_ERSP_mnt_vs_city_{ersp_ch}.png'))

#%% Analysis 7 — ERP in-zone vs out-of-zone (mnt_correct, all channels)
zone_erp_event = 'mnt_correct'
in_zone_erp  = {}
out_zone_erp = {}

for ch in vis_ch:
    subj_in_erp  = []
    subj_out_erp = []
    for subj_i, epoch in enumerate(combine_epoch_dict[zone_erp_event][ch]):
        if len(epoch) == 0:
            continue
        ch_erp   = np.squeeze(epoch.get_data())
        in_mask  = in_out_zone_dict[zone_erp_event][ch][subj_i]
        out_mask = ~in_mask
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue
        subj_in_erp.append(np.mean(ch_erp[in_mask],  axis=0))
        subj_out_erp.append(np.mean(ch_erp[out_mask], axis=0))
    in_zone_erp[ch]  = np.vstack(subj_in_erp)  if subj_in_erp  else None
    out_zone_erp[ch] = np.vstack(subj_out_erp) if subj_out_erp else None

for ch in vis_ch:
    if in_zone_erp[ch] is None or out_zone_erp[ch] is None:
        continue
    plt_in  = in_zone_erp[ch]
    plt_out = out_zone_erp[ch]
    n = plt_in.shape[0]
    mean_in  = np.mean(plt_in,  axis=0)
    sem_in   = np.std(plt_in,   axis=0) / np.sqrt(n)
    mean_out = np.mean(plt_out, axis=0)
    sem_out  = np.std(plt_out,  axis=0) / np.sqrt(n)
    t_ms = combine_epoch_dict[zone_erp_event][ch][0].times * 1000

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms, mean_in,  color='b', linewidth=2, label=f'In-zone (n={n})')
    ax.fill_between(t_ms, mean_in  - 2*sem_in,  mean_in  + 2*sem_in,  alpha=0.3, color='b')
    ax.plot(t_ms, mean_out, color='r', linewidth=2, label=f'Out-of-zone (n={n})')
    ax.fill_between(t_ms, mean_out - 2*sem_out, mean_out + 2*sem_out, alpha=0.3, color='r')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(f'Cross-subject ERP in-zone vs out-of-zone — {zone_erp_event} — {ch.upper()} (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERP_inzone_vs_outzone'], f'xSubj_ERP_inzone_vs_outzone_{zone_erp_event}_{ch}.png'))

#%% Analysis 7b — ERP difference (in-zone - out-of-zone) with paired t-test (mnt_correct, all channels)
alpha_diff = 0.05

for ch in vis_ch:
    if in_zone_erp[ch] is None or out_zone_erp[ch] is None:
        continue
    plt_in  = in_zone_erp[ch]
    plt_out = out_zone_erp[ch]
    n = plt_in.shape[0]
    diff = plt_in - plt_out  # subjects x time
    mean_diff = np.mean(diff, axis=0)
    sem_diff  = np.std(diff, axis=0) / np.sqrt(n)
    t_ms = combine_epoch_dict[zone_erp_event][ch][0].times * 1000

    tvals, pvals = stats.ttest_rel(plt_in, plt_out, axis=0)
    sig_mask, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha_diff, method='fdr_bh')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms, mean_diff, color='purple', linewidth=2, label=f'In-zone - Out-of-zone (n={n})')
    ax.fill_between(t_ms, mean_diff - 2*sem_diff, mean_diff + 2*sem_diff, alpha=0.3, color='purple')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    if sig_mask.any():
        ylim = ax.get_ylim()
        y_bar = ylim[0] + 0.05 * (ylim[1] - ylim[0])
        ax.scatter(t_ms[sig_mask], np.full(sig_mask.sum(), y_bar),
                   color='black', marker='s', s=10,
                   label=f'FDR-corrected p < {alpha_diff} (paired t-test)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude Difference (V)')
    ax.set_title(f'Cross-subject ERP Difference (in-zone - out-of-zone) — {zone_erp_event} — {ch.upper()} (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERP_inzone_vs_outzone'], f'xSubj_ERP_diff_inzone_outzone_{zone_erp_event}_{ch}.png'))

#%% Analysis 8 — ERSP in-zone vs out-of-zone (city_correct_response, CZ)
inout_event = 'city_correct_response'
inout_ch    = 'cz'
time_window_duration_inout = 0.25
time_window_step_inout     = 0.1

start_time = time.time()
in_zone_epochs_list  = []
out_zone_epochs_list = []
for subj_i, epoch in enumerate(combine_epoch_dict[inout_event][inout_ch]):
    if len(epoch) == 0:
        continue
    in_mask  = in_out_zone_dict[inout_event][inout_ch][subj_i]
    out_mask = ~in_mask
    if in_mask.sum() > 0:
        in_zone_epochs_list.append(epoch[in_mask])
    if out_mask.sum() > 0:
        out_zone_epochs_list.append(epoch[out_mask])

if in_zone_epochs_list:
    ep_in  = mne.concatenate_epochs(in_zone_epochs_list)
    (_, _, _) = plt_multitaper(ep_in,
                               time_halfbandwidth_product=time_halfbandwidth_product,
                               time_window_duration=time_window_duration_inout,
                               time_window_step=time_window_step_inout,
                               ratio_to=None,
                               vis_f_range=[1/time_window_duration_inout, 40])
    fig = plt.gcf()
    fig.suptitle(f'Cross-subject ERSP in-zone — {inout_event} — {inout_ch.upper()}')
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'xSubj_ERSP_inzone_{inout_event}_{inout_ch}.png'))

if out_zone_epochs_list:
    ep_out = mne.concatenate_epochs(out_zone_epochs_list)
    (_, _, _) = plt_multitaper(ep_out,
                               time_halfbandwidth_product=time_halfbandwidth_product,
                               time_window_duration=time_window_duration_inout,
                               time_window_step=time_window_step_inout,
                               ratio_to=None,
                               vis_f_range=[1/time_window_duration_inout, 40])
    fig = plt.gcf()
    fig.suptitle(f'Cross-subject ERSP out-of-zone — {inout_event} — {inout_ch.upper()}')
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'xSubj_ERSP_outzone_{inout_event}_{inout_ch}.png'))

if in_zone_epochs_list and out_zone_epochs_list:
    (_, multitaper, _) = plt_multitaper(ep_in,
                                        time_halfbandwidth_product=time_halfbandwidth_product,
                                        time_window_duration=time_window_duration_inout,
                                        time_window_step=time_window_step_inout,
                                        ratio_to=ep_out,
                                        vis_f_range=[1/time_window_duration_inout, 40])
    fig = plt.gcf()
    fig.suptitle(f'Cross-subject ERSP in/out ratio — {inout_event} — {inout_ch.upper()}')
    save_or_show_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'xSubj_ERSP_inout_ratio_{inout_event}_{inout_ch}.png'))
    print(f"ERSP in/out: freq res = {multitaper.frequency_resolution:.2f} Hz  ({time.time()-start_time:.1f}s)")

#%% Analysis 9 — PSD comparison (city & mnt, in/out zone, CZ)
psd_ch = 'cz'
psd_results = {}
for psd_event in ['city_correct', 'mnt_correct']:
    in_ep_list  = []
    out_ep_list = []
    for subj_i, epoch in enumerate(combine_epoch_dict[psd_event][psd_ch]):
        if len(epoch) == 0:
            continue
        in_mask  = in_out_zone_dict[psd_event][psd_ch][subj_i]
        out_mask = ~in_mask
        if in_mask.sum() > 0:
            in_ep_list.append(epoch[in_mask])
        if out_mask.sum() > 0:
            out_ep_list.append(epoch[out_mask])
    if not in_ep_list or not out_ep_list:
        continue
    ep_in  = mne.concatenate_epochs(in_ep_list)
    ep_out = mne.concatenate_epochs(out_ep_list)
    (log_pow_in,  _, _)            = plt_multitaper(ep_in,  time_halfbandwidth_product=time_halfbandwidth_product, time_window_duration=None, is_plot=False)
    (log_pow_out, _, connectivity) = plt_multitaper(ep_out, time_halfbandwidth_product=time_halfbandwidth_product, time_window_duration=None, is_plot=False)
    psd_results[psd_event] = {'in': log_pow_in, 'out': log_pow_out, 'freqs': connectivity.frequencies}

if len(psd_results) == 2:
    vis_f_range = [0, 50]
    freqs    = psd_results['city_correct']['freqs']
    vis_mask = (freqs >= vis_f_range[0]) & (freqs <= vis_f_range[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freqs[vis_mask], psd_results['city_correct']['in'][vis_mask],  'b-',  label='City (in-zone)')
    ax.plot(freqs[vis_mask], psd_results['city_correct']['out'][vis_mask], 'b--', label='City (out-of-zone)')
    ax.plot(freqs[vis_mask], psd_results['mnt_correct']['in'][vis_mask],   'r-',  label='Mountain (in-zone)')
    ax.plot(freqs[vis_mask], psd_results['mnt_correct']['out'][vis_mask],  'r--', label='Mountain (out-of-zone)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Power ($\log\,V^2$)')
    ax.set_title(f'Cross-subject PSD comparison — {psd_ch.upper()} (n={n_subjects})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show_fig(fig, os.path.join(analysis_dirs['PSD'], f'xSubj_PSD_city_mnt_inout_{psd_ch}.png'))

print("\nAll analyses complete. Results saved in:", results_root)

#%% Write exclusion log
log_path = os.path.join(results_root, "excluded_subjects.txt")
with open(log_path, 'w') as f:
    f.write(f"Exclusion log — generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Split criterion: {split_zone_crit}\n")
    f.write(f"Subjects included in cross-subject analysis: {n_subjects}\n")
    f.write("=" * 60 + "\n\n")
    if excluded_subjects:
        for subj, reason in sorted(excluded_subjects.items()):
            f.write(f"{subj}\n  Reason: {reason}\n\n")
    else:
        f.write("No subjects excluded.\n")
print(f"Exclusion log saved to: {log_path}")
