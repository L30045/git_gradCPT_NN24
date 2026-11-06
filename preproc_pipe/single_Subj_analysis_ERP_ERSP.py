"""
Single-subject EEG ERP and ERSP analysis pipeline.
Results are organised by analysis method in single_Subj_results/<analysis>/.
"""
#%% load library
import numpy as np
from scipy import stats
import matplotlib

# is_plot=True : show every figure inline in the interactive window (and save it)
# is_plot=False: batch run, save figures only (non-interactive Agg backend)
is_plot = False

if is_plot:
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')   # undo an earlier Agg in this session
    except NameError:
        pass
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
from utils import *
from tqdm import tqdm
import pickle
import gzip
import time


def save_fig(fig, path):
    """Save a figure; display it inline when is_plot, then close it."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if is_plot:
        from IPython.display import display
        display(fig)
    plt.close(fig)

#%% output directory — one sub-folder per analysis method
script_dir = os.path.dirname(os.path.abspath(__file__))
results_root = os.path.join("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg",
                            "single_Subj_results")

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
    'BandPowerImage':         os.path.join(results_root, 'BandPowerImage'),
    'BandPower_inzone_outzone': os.path.join(results_root, 'BandPower_inzone_outzone'),
}
for d in analysis_dirs.values():
    os.makedirs(d, exist_ok=True)

#%% preprocessing parameter setting
subj_id_array = sorted([
    int(d.split('-')[1])
    for d in os.listdir(data_save_path)
    if d.startswith('sub-') and os.path.isdir(os.path.join(data_save_path, d))
])

ch_names = ['fz', 'cz', 'pz', 'oz']
split_zone_crit = 'vtc'
is_bpfilter = True
bp_f_range = [0.1, 45]
is_reref = True
reref_ch = ['tp9h', 'tp10h']
is_ica_rmEye = True
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
    select_event="mnt_correct",
    baseline_length=baseline_length,
    epoch_reject_crit=epoch_reject_crit,
    is_detrend=is_detrend,
    ch_names=ch_names,
    is_overwrite=is_overwrite
)

# multitaper parameters
time_halfbandwidth_product = 1
time_window_duration_ersp = 0.2
time_window_step_ersp = 0.05
time_window_duration_inout = 0.25
time_window_step_inout = 0.1

# band-power image parameters (same bands as extract_EEG_trial_features.py)
bp_bands = {'delta': (1.0, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0)}

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
    rm_ch_dict[f"sub-{subj_id}"] = _payload["rm_ch"]
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
    subj_vtc_dict[key_name] = single_subj_vtc_dict
    subj_react_dict[key_name] = single_subj_react_dict

#%% per-subject analysis loop
for key_name in tqdm(subj_epoch_dict.keys()):
    subj_id = int(key_name.split('-')[-1])
    print(f"\n{'='*40}\nProcessing {key_name}\n{'='*40}")

    # ------------------------------------------------------------------ #
    # 1.  Combine epochs across runs and build zone labels
    # ------------------------------------------------------------------ #
    # compute per-subject threshold (median of VTC or RT across all runs/events)
    avail_runs = list(subj_epoch_dict[key_name].keys())
    match split_zone_crit:
        case 'vtc':
            all_vals_list = [subj_vtc_dict[key_name][run][ev]
                             for run in avail_runs
                             for ev in event_labels_lookup.keys()
                             if not ev.endswith("_response")
                             and len(subj_vtc_dict[key_name][run][ev]) > 0]
        case 'react':
            all_vals_list = [subj_react_dict[key_name][run][ev]
                             for run in avail_runs
                             for ev in event_labels_lookup.keys()
                             if not ev.endswith("_response")
                             and len(subj_react_dict[key_name][run][ev]) > 0]
    if len(all_vals_list) == 0:
        print(f"  No valid trials found for {key_name}, skipping.")
        excluded_subjects[key_name] = (
            f"No valid non-response trials found after epoching "
            f"(available runs: {avail_runs}; split criterion: {split_zone_crit})"
        )
        continue
    subj_thres = np.median(np.concatenate(all_vals_list))

    # combine runs -> one epoch object per (event, channel)
    epoch_dict = {ev: {ch: [] for ch in ch_names} for ev in event_labels_lookup.keys()}
    vtc_dict   = {ev: {ch: [] for ch in ch_names} for ev in event_labels_lookup.keys()}
    react_dict = {ev: {ch: [] for ch in ch_names} for ev in event_labels_lookup.keys()}
    zone_dict  = {ev: {ch: [] for ch in ch_names} for ev in event_labels_lookup.keys()}

    for select_event in event_labels_lookup.keys():
        tmp_epoch_list = []
        tmp_vtc_list   = []
        tmp_react_list = []
        tmp_zone_list  = []
        for run in avail_runs:
            loc_e = subj_epoch_dict[key_name][run][select_event]
            loc_v = subj_vtc_dict[key_name][run][select_event]
            loc_r = subj_react_dict[key_name][run][select_event]
            if len(loc_e) > 0:
                tmp_epoch_list.append(loc_e)
                tmp_vtc_list.append(loc_v)
                tmp_react_list.append(loc_r)
                tmp_zone_list.append(loc_v < subj_thres)

        for ch in ch_names:
            if len(tmp_epoch_list) == 0:
                continue
            ch_epochs = [x.copy().pick(ch) for x in tmp_epoch_list if ch in x.ch_names]
            if len(ch_epochs) == 0:
                continue
            epoch_dict[select_event][ch] = mne.concatenate_epochs(ch_epochs, verbose=False)
            vtc_dict[select_event][ch]   = np.concatenate([v for v, e in zip(tmp_vtc_list,   tmp_epoch_list) if ch in e.ch_names])
            react_dict[select_event][ch] = np.concatenate([r for r, e in zip(tmp_react_list, tmp_epoch_list) if ch in e.ch_names])
            zone_dict[select_event][ch]  = np.concatenate([z for z, e in zip(tmp_zone_list,  tmp_epoch_list) if ch in e.ch_names])

    # ------------------------------------------------------------------ #
    # 2.  RT in-zone vs out-of-zone comparison
    # ------------------------------------------------------------------ #
    check_ch = 'cz'
    ev_city = 'city_correct'
    if len(epoch_dict[ev_city][check_ch]) > 0:
        in_mask  = zone_dict[ev_city][check_ch]
        out_mask = ~in_mask
        rt_in  = react_dict[ev_city][check_ch][in_mask]
        rt_out = react_dict[ev_city][check_ch][out_mask]
        rt_diff_ms = (np.mean(rt_in) - np.mean(rt_out)) * 1000
        t_stat, p_val = stats.ttest_ind(rt_in, rt_out)
        print(f"  RT diff (in-zone - out-of-zone): {rt_diff_ms:.2f} ms  t={t_stat:.3f}  p={p_val:.4f}")

        sig_label = f't={t_stat:.2f}, p={p_val:.4f}'
        if p_val < 0.001:
            sig_label += ' ***'
        elif p_val < 0.01:
            sig_label += ' **'
        elif p_val < 0.05:
            sig_label += ' *'

        # plot each zone's mean RT relative to the overall mean RT (all trials of both zones)
        rt_all_mean = np.mean(np.concatenate([rt_in, rt_out]))
        rel_means = np.array([np.mean(rt_in), np.mean(rt_out)]) * 1000 - rt_all_mean * 1000
        sems = np.array([np.std(rt_in) / np.sqrt(len(rt_in)),
                         np.std(rt_out) / np.sqrt(len(rt_out))]) * 1000

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar([f'In-zone (n={len(rt_in)})', f'Out-of-zone (n={len(rt_out)})'],
                      rel_means, yerr=sems, capsize=5, color=['steelblue', 'tomato'])
        ax.axhline(0, color='k', linewidth=1)
        y_lim = np.max(np.abs(rel_means) + sems) * 1.4
        ax.set_ylim(-y_lim, y_lim)
        ax.annotate(f'{sig_label}\nin - out = {rt_diff_ms:.1f} ms',
                    xy=(0.5, 0.97), xycoords='axes fraction',
                    ha='center', va='top', fontsize=10)
        ax.set_ylabel(f'Mean RT - overall mean (ms)\noverall mean = {rt_all_mean * 1000:.0f} ms')
        ax.set_title(f'{key_name} — RT in-zone vs out-of-zone ({ev_city}, {check_ch.upper()})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(analysis_dirs['RT_inzone_vs_outzone'], f'{key_name}_RT_inzone_vs_outzone.png'))

    # ------------------------------------------------------------------ #
    # 3.  ERP: city vs mountain
    # ------------------------------------------------------------------ #
    vis_events = ['city_correct', 'mnt_correct']
    colors = ['b', 'r']
    vis_ch = ch_names

    for ch in vis_ch:
        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, select_event in enumerate(vis_events):
            ep = epoch_dict[select_event][ch]
            if len(ep) == 0:
                continue
            ep_data = np.squeeze(ep.get_data())
            mean_erp = np.mean(ep_data, axis=0)
            sem_erp  = np.std(ep_data, axis=0) / np.sqrt(ep_data.shape[0])
            t_ms = ep.times * 1000
            label = 'City' if select_event.startswith('city') else 'Mountain'
            ax.plot(t_ms, mean_erp, color=colors[idx], linewidth=2, label=f'{label} (n={ep_data.shape[0]})')
            ax.fill_between(t_ms, mean_erp - 2 * sem_erp, mean_erp + 2 * sem_erp, alpha=0.3, color=colors[idx])
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        # overall mean RT of city_correct trials (mnt_correct trials have no response)
        rt_city = react_dict['city_correct'][ch]
        if len(rt_city) > 0:
            rt_city_mean_ms = np.mean(rt_city) * 1000
            ax.axvline(rt_city_mean_ms, color='b', linestyle=':', linewidth=1.5,
                       label=f'City mean RT ({rt_city_mean_ms:.0f} ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(f'{key_name} — ERP City vs Mountain — {ch.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(analysis_dirs['ERP_city_vs_mnt'], f'{key_name}_ERP_city_vs_mnt_{ch}.png'))

    # ------------------------------------------------------------------ #
    # 4.  ERP Image sorted by VTC
    # ------------------------------------------------------------------ #
    erp_image_event = 'mnt_correct'
    erp_image_ch    = 'cz'
    ep = epoch_dict[erp_image_event][erp_image_ch]
    if len(ep) > 0:
        time_vector = ep.times
        ep_data = np.squeeze(ep.get_data())
        window_size = np.max([4, np.floor(ep_data.shape[0] * 0.01).astype(int)])
        plt_vtc   = vtc_dict[erp_image_event][erp_image_ch]
        plt_react = react_dict[erp_image_event][erp_image_ch]
        title_txt = f'{key_name} — {erp_image_event} — {erp_image_ch.upper()}'

        fig = plt_ERPImage(time_vector, ep_data,
                           sort_idx=plt_vtc,
                           smooth_window_size=window_size,
                           clim=[-10e-6, 10e-6],
                           title_txt=title_txt,
                           ref_onset=plt_react,
                           show=False)
        save_fig(fig, os.path.join(analysis_dirs['ERPImage'], f'{key_name}_ERPImage_{erp_image_event}_{erp_image_ch}.png'))

    # ------------------------------------------------------------------ #
    # 5.  ERSP — mnt_correct (raw power)
    # ------------------------------------------------------------------ #
    ersp_event = 'mnt_correct'
    ersp_ch    = 'cz'
    ep = epoch_dict[ersp_event][ersp_ch]
    if len(ep) > 0:
        start_time = time.time()
        (_, multitaper, _) = plt_multitaper(ep,
                                            time_halfbandwidth_product=time_halfbandwidth_product,
                                            time_window_duration=time_window_duration_ersp,
                                            time_window_step=time_window_step_ersp,
                                       show=False)
        fig = plt.gcf()
        fig.suptitle(f'{key_name} — ERSP (raw) — {ersp_event} — {ersp_ch.upper()}')
        save_fig(fig, os.path.join(analysis_dirs['ERSP_raw'], f'{key_name}_ERSP_raw_{ersp_event}_{ersp_ch}.png'))
        print(f"  ERSP (raw): freq res = {multitaper.frequency_resolution:.2f} Hz  ({time.time()-start_time:.1f}s)")

    # ------------------------------------------------------------------ #
    # 6.  ERSP — mnt_correct ratio to baseline
    # ------------------------------------------------------------------ #
    if len(ep) > 0:
        (_, multitaper, _) = plt_multitaper(ep,
                                            time_halfbandwidth_product=time_halfbandwidth_product,
                                            time_window_duration=time_window_duration_ersp,
                                            time_window_step=time_window_step_ersp,
                                            ratio_to="baseline",
                                       show=False)
        fig = plt.gcf()
        fig.suptitle(f'{key_name} — ERSP (baseline ratio) — {ersp_event} — {ersp_ch.upper()}')
        save_fig(fig, os.path.join(analysis_dirs['ERSP_baseline'], f'{key_name}_ERSP_baseline_{ersp_event}_{ersp_ch}.png'))

    # ------------------------------------------------------------------ #
    # 7.  ERSP — mnt_correct vs city_correct
    # ------------------------------------------------------------------ #
    ref_event = 'city_correct'
    ep_ref = epoch_dict[ref_event][ersp_ch]
    if len(ep) > 0 and len(ep_ref) > 0:
        (_, multitaper, _) = plt_multitaper(ep,
                                            time_halfbandwidth_product=time_halfbandwidth_product,
                                            time_window_duration=time_window_duration_ersp,
                                            time_window_step=time_window_step_ersp,
                                            ratio_to=ep_ref,
                                       show=False)
        fig = plt.gcf()
        fig.suptitle(f'{key_name} — ERSP (mnt/city ratio) — {ersp_ch.upper()}')
        save_fig(fig, os.path.join(analysis_dirs['ERSP_mnt_vs_city'], f'{key_name}_ERSP_mnt_vs_city_{ersp_ch}.png'))

    # ------------------------------------------------------------------ #
    # 8.  ERP — in-zone vs out-of-zone (all channels, mnt_correct)
    # ------------------------------------------------------------------ #
    zone_erp_event = 'mnt_correct'
    for ch in vis_ch:
        ep = epoch_dict[zone_erp_event][ch]
        if len(ep) == 0:
            continue
        ep_data   = np.squeeze(ep.get_data())
        in_mask   = zone_dict[zone_erp_event][ch]
        out_mask  = ~in_mask
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue

        mean_in  = np.mean(ep_data[in_mask],  axis=0)
        sem_in   = np.std(ep_data[in_mask],   axis=0) / np.sqrt(in_mask.sum())
        mean_out = np.mean(ep_data[out_mask], axis=0)
        sem_out  = np.std(ep_data[out_mask],  axis=0) / np.sqrt(out_mask.sum())
        t_ms = ep.times * 1000

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_ms, mean_in,  color='b', linewidth=2, label=f'In-zone (n={in_mask.sum()})')
        ax.fill_between(t_ms, mean_in  - 2*sem_in,  mean_in  + 2*sem_in,  alpha=0.3, color='b')
        ax.plot(t_ms, mean_out, color='r', linewidth=2, label=f'Out-of-zone (n={out_mask.sum()})')
        ax.fill_between(t_ms, mean_out - 2*sem_out, mean_out + 2*sem_out, alpha=0.3, color='r')
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        # mean RT lines from city_correct trials (mnt_correct trials have no response),
        # split by the same per-subject VTC threshold
        rt_city   = react_dict['city_correct'][ch]
        zone_city = zone_dict['city_correct'][ch]
        if len(rt_city) > 0:
            ax.axvline(np.mean(rt_city) * 1000, color='k', linestyle=':', linewidth=1.5,
                       label=f'City mean RT, all ({np.mean(rt_city) * 1000:.0f} ms)')
            for z_mask, z_color, z_name in [(zone_city, 'b', 'in-zone'), (~zone_city, 'r', 'out-of-zone')]:
                if z_mask.sum() > 0:
                    z_rt_ms = np.mean(rt_city[z_mask]) * 1000
                    ax.axvline(z_rt_ms, color=z_color, linestyle=':', linewidth=1.5,
                               label=f'City mean RT, {z_name} ({z_rt_ms:.0f} ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(f'{key_name} — ERP in-zone vs out-of-zone — {zone_erp_event} — {ch.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(analysis_dirs['ERP_inzone_vs_outzone'], f'{key_name}_ERP_inzone_vs_outzone_{zone_erp_event}_{ch}.png'))

    # ------------------------------------------------------------------ #
    # 9.  ERSP — in-zone vs out-of-zone (city_correct_response)
    # ------------------------------------------------------------------ #
    inout_event = 'city_correct_response'
    inout_ch    = 'cz'
    ep = epoch_dict[inout_event][inout_ch]
    if len(ep) > 0:
        start_time = time.time()
        in_mask  = zone_dict[inout_event][inout_ch]
        out_mask = ~in_mask
        ratio_to_val = None  # response-locked: no baseline

        if in_mask.sum() > 0:
            ep_in  = ep[in_mask]
            (_, _, _) = plt_multitaper(ep_in,
                                       time_halfbandwidth_product=time_halfbandwidth_product,
                                       time_window_duration=time_window_duration_inout,
                                       time_window_step=time_window_step_inout,
                                       ratio_to=ratio_to_val,
                                       vis_f_range=[1/time_window_duration_inout, 40],
                                       show=False)
            fig = plt.gcf()
            fig.suptitle(f'{key_name} — ERSP in-zone — {inout_event} — {inout_ch.upper()}')
            save_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'{key_name}_ERSP_inzone_{inout_event}_{inout_ch}.png'))

        if out_mask.sum() > 0:
            ep_out = ep[out_mask]
            (_, _, _) = plt_multitaper(ep_out,
                                       time_halfbandwidth_product=time_halfbandwidth_product,
                                       time_window_duration=time_window_duration_inout,
                                       time_window_step=time_window_step_inout,
                                       ratio_to=ratio_to_val,
                                       vis_f_range=[1/time_window_duration_inout, 40],
                                       show=False)
            fig = plt.gcf()
            fig.suptitle(f'{key_name} — ERSP out-of-zone — {inout_event} — {inout_ch.upper()}')
            save_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'{key_name}_ERSP_outzone_{inout_event}_{inout_ch}.png'))

        if in_mask.sum() > 0 and out_mask.sum() > 0:
            (_, multitaper, _) = plt_multitaper(ep_in,
                                                time_halfbandwidth_product=time_halfbandwidth_product,
                                                time_window_duration=time_window_duration_inout,
                                                time_window_step=time_window_step_inout,
                                                ratio_to=ep_out,
                                                vis_f_range=[1/time_window_duration_inout, 40],
                                       show=False)
            fig = plt.gcf()
            fig.suptitle(f'{key_name} — ERSP in/out ratio — {inout_event} — {inout_ch.upper()}')
            save_fig(fig, os.path.join(analysis_dirs['ERSP_inzone_outzone'], f'{key_name}_ERSP_inout_ratio_{inout_event}_{inout_ch}.png'))
            print(f"  ERSP in/out: freq res = {multitaper.frequency_resolution:.2f} Hz  ({time.time()-start_time:.1f}s)")

    # ------------------------------------------------------------------ #
    # 10. PSD comparison — city & mnt, in-zone vs out-of-zone
    # ------------------------------------------------------------------ #
    psd_ch = 'cz'
    psd_results = {}
    for psd_event in ['city_correct', 'mnt_correct']:
        ep = epoch_dict[psd_event][psd_ch]
        if len(ep) == 0:
            continue
        in_mask  = zone_dict[psd_event][psd_ch]
        out_mask = ~in_mask
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue
        (log_pow_in,  _, _)            = plt_multitaper(ep[in_mask],  time_halfbandwidth_product=time_halfbandwidth_product, time_window_duration=None, is_plot=False)
        (log_pow_out, _, connectivity) = plt_multitaper(ep[out_mask], time_halfbandwidth_product=time_halfbandwidth_product, time_window_duration=None, is_plot=False)
        psd_results[psd_event] = {'in': log_pow_in, 'out': log_pow_out, 'freqs': connectivity.frequencies}

    if len(psd_results) == 2:
        vis_f_range = [0, 50]
        freqs = psd_results['city_correct']['freqs']
        vis_mask = (freqs >= vis_f_range[0]) & (freqs <= vis_f_range[1])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(freqs[vis_mask], psd_results['city_correct']['in'][vis_mask],  'b-',  label='City (in-zone)')
        ax.plot(freqs[vis_mask], psd_results['city_correct']['out'][vis_mask], 'b--', label='City (out-of-zone)')
        ax.plot(freqs[vis_mask], psd_results['mnt_correct']['in'][vis_mask],   'r-',  label='Mountain (in-zone)')
        ax.plot(freqs[vis_mask], psd_results['mnt_correct']['out'][vis_mask],  'r--', label='Mountain (out-of-zone)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'Power ($\log\,V^2$)')
        ax.set_title(f'{key_name} — PSD comparison — {psd_ch.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(analysis_dirs['PSD'], f'{key_name}_PSD_city_mnt_inout_{psd_ch}.png'))

    # ------------------------------------------------------------------ #
    # 11. Band-power image — mnt_correct, sorted by VTC (no baseline correction)
    # ------------------------------------------------------------------ #
    # Instantaneous band power = squared Hilbert envelope of the band-passed CONTINUOUS run
    # (avoids the edge effects of a time-frequency transform on the short epochs), then cut
    # into epochs at the same events kept in the amplitude epochs, so trials and VTC match
    # sections 1-4. Absolute power (uV^2): baseline=None, no normalisation to pre-stimulus.
    bp_event = 'mnt_correct'
    run_name_lookup = {f"run{int(rn.split('cpt')[-1]):02d}": rn
                       for rn in subj_EEG_dict[key_name] if 'gradcpt' in rn}
    bp_trials = {b: {ch: [] for ch in vis_ch} for b in bp_bands}   # band -> ch -> list of (trial, time)
    bp_vtc = {ch: [] for ch in vis_ch}
    bp_times = None
    for run in avail_runs:
        ep_run = subj_epoch_dict[key_name][run][bp_event]
        if len(ep_run) == 0:
            continue
        raw = subj_EEG_dict[key_name][run_name_lookup[run]]
        run_chs = [ch for ch in vis_ch if ch in ep_run.ch_names]
        for bname, (lo, hi) in bp_bands.items():
            raw_bp = raw.copy().pick(run_chs).filter(lo, hi, verbose=False)
            raw_bp.apply_hilbert(envelope=True, verbose=False)
            raw_bp.apply_function(lambda x: (x * 1e6) ** 2, verbose=False)   # uV^2
            ep_bp = mne.Epochs(raw_bp, events=ep_run.events, event_id=ep_run.event_id,
                               tmin=ep_run.tmin, tmax=ep_run.tmax, baseline=None,
                               reject=None, detrend=None, preload=True, verbose=False)
            assert len(ep_bp) == len(ep_run)
            bp_times = ep_bp.times
            for ch in run_chs:
                bp_trials[bname][ch].append(ep_bp.get_data(picks=ch)[:, 0, :])
        for ch in run_chs:
            bp_vtc[ch].append(subj_vtc_dict[key_name][run][bp_event])

    for ch in vis_ch:
        if len(bp_vtc[ch]) == 0:
            continue
        plt_vtc = np.concatenate(bp_vtc[ch])
        window_size = np.max([4, np.floor(len(plt_vtc) * 0.01).astype(int)])
        for bname, (lo, hi) in bp_bands.items():
            bp = np.concatenate(bp_trials[bname][ch], axis=0)          # (trial, time), uV^2
            clim = np.percentile(bp, [2, 98])
            title_txt = f'{key_name} — {bp_event} — {ch.upper()} — {bname} ({lo:g}-{hi:g} Hz)'
            fig = plt_ERPImage(bp_times, bp,
                               sort_idx=plt_vtc,
                               smooth_window_size=window_size,
                               clim=clim,
                               title_txt=title_txt,
                               cmap='viridis',
                               cbar_label=r'Power ($\mu V^2$)',
                               avg_ylabel=r'Power ($\mu V^2$)',
                               img_title='Band-power image',
                               avg_title='Average band power',
                               show=False)
            save_fig(fig, os.path.join(analysis_dirs['BandPowerImage'],
                                     f'{key_name}_BandPowerImage_{bp_event}_{ch}_{bname}.png'))

    # ------------------------------------------------------------------ #
    # 12. Band power — in-zone vs out-of-zone (mnt_correct, no baseline correction)
    # ------------------------------------------------------------------ #
    # Uses the band-power epochs from section 11. Zones use the same rule as section 1
    # (VTC < subject threshold = in-zone). Top: mean power time course +-2 SEM.
    # The test compares each trial's power averaged over the whole epoch; power is skewed,
    # so the t-test is done on log10 power.
    for ch in vis_ch:
        if len(bp_vtc[ch]) == 0:
            continue
        in_mask  = np.concatenate(bp_vtc[ch]) < subj_thres
        out_mask = ~in_mask
        if in_mask.sum() < 2 or out_mask.sum() < 2:
            continue
        fig, axes = plt.subplots(2, len(bp_bands), figsize=(4 * len(bp_bands), 7),
                                 height_ratios=[2, 1], constrained_layout=True)
        for i, (bname, (lo, hi)) in enumerate(bp_bands.items()):
            bp = np.concatenate(bp_trials[bname][ch], axis=0)          # (trial, time), uV^2
            ax = axes[0, i]
            for z_mask, z_color, z_name in [(in_mask, 'b', 'In-zone'), (out_mask, 'r', 'Out-of-zone')]:
                m = bp[z_mask].mean(axis=0)
                se = bp[z_mask].std(axis=0) / np.sqrt(z_mask.sum())
                ax.plot(bp_times * 1000, m, color=z_color, linewidth=2, label=f'{z_name} (n={z_mask.sum()})')
                ax.fill_between(bp_times * 1000, m - 2 * se, m + 2 * se, color=z_color, alpha=0.3)
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.set_title(f'{bname} ({lo:g}-{hi:g} Hz)')
            ax.set_xlabel('Time (ms)')
            if i == 0:
                ax.set_ylabel(r'Power ($\mu V^2$)')
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # trial-wise epoch-mean power: in vs out
            log_tr = np.log10(bp.mean(axis=1))
            t_stat, p_val = stats.ttest_ind(log_tr[in_mask], log_tr[out_mask])
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            ax = axes[1, i]
            ax.boxplot([log_tr[in_mask], log_tr[out_mask]], widths=0.5, showfliers=False)
            for k, (z_mask, z_color) in enumerate([(in_mask, 'b'), (out_mask, 'r')]):
                jitter = np.random.default_rng(0).uniform(-0.12, 0.12, z_mask.sum())
                ax.scatter(k + 1 + jitter, log_tr[z_mask], s=6, color=z_color, alpha=0.5)
            ax.set_xticks([1, 2], ['In-zone', 'Out-of-zone'])
            ax.set_title(f't={t_stat:.2f}, p={p_val:.4f} {stars}', fontsize=10)
            if i == 0:
                ax.set_ylabel(r'log$_{10}$ epoch-mean power')
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'{key_name} — band power in-zone vs out-of-zone — {bp_event} — {ch.upper()}')
        save_fig(fig, os.path.join(analysis_dirs['BandPower_inzone_outzone'],
                                   f'{key_name}_BandPower_inzone_vs_outzone_{bp_event}_{ch}.png'))

    print(f"  Saved all figures for {key_name}")

print("\nAll subjects processed. Results saved in:", results_root)

#%% write exclusion log
import datetime
log_path = os.path.join(results_root, "excluded_subjects.txt")
with open(log_path, 'w') as f:
    f.write(f"Exclusion log — generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Split criterion: {split_zone_crit}\n")
    f.write("=" * 60 + "\n\n")
    if excluded_subjects:
        for subj, reason in sorted(excluded_subjects.items()):
            f.write(f"{subj}\n  Reason: {reason}\n\n")
    else:
        f.write("No subjects excluded.\n")
print(f"Exclusion log saved to: {log_path}")
