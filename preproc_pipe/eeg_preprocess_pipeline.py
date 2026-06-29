"""
General EEG preprocessing pipeline
"""
#%% load library
import numpy as np
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

#%% preprocessing parameter setting
# subj_id_array = [670, 671, 673, 695]
# subj_id_array = [670, 671, 673, 695, 719, 721, 723, 726, 727, 730, 733]
# subj_id_array = [670, 695, 719, 721, 723, 726, 727, 730]
# subj_id_array = [746, 750, 751]

# Discover all subjects that have an eeg/ folder
all_subj_dirs = sorted(glob.glob(os.path.join(data_path, "sub-*")))
all_subj_ids = [int(os.path.basename(d).split('-')[1])
                for d in all_subj_dirs
                if os.path.isdir(os.path.join(d, 'eeg'))]
ch_names = ['fz','cz','pz','oz']
split_zone_crit = 'react'
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
# reref_ch = None # reref to average
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None
is_overwrite = False # Force to re run preprocessing if it is True

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    select_event = select_event,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names,
    is_overwrite = is_overwrite
)

#%% visualize EEG after each preprocessing step — all subjects, all gradCPT runs

def _savefig(fig, save_dir, fname):
    """Save figure and close it."""
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

def _make_psd_fig(EEG, ch_names_pick, title, fmax=60):
    picks = [EEG.ch_names.index(c) for c in ch_names_pick if c in EEG.ch_names]
    fig, ax = plt.subplots(figsize=(8, 4))
    EEG.compute_psd(picks=picks, fmax=fmax).plot(axes=ax, show=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def _make_timeseries_fig(EEG, ch_names_pick, title, t_sec=(0, 10)):
    picks = [EEG.ch_names.index(c) for c in ch_names_pick if c in EEG.ch_names]
    tmin_idx = int(t_sec[0] * EEG.info["sfreq"])
    tmax_idx = int(t_sec[1] * EEG.info["sfreq"])
    data = EEG.get_data(picks=picks)[:, tmin_idx:tmax_idx] * 1e6  # V -> µV
    times = EEG.times[tmin_idx:tmax_idx]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, i, ch in zip(axes, range(len(picks)), [EEG.ch_names[p] for p in picks]):
        ax.plot(times, data[i], lw=0.8)
        ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig

#%% main
for vis_subj_id in tqdm(all_subj_ids, desc="subjects"):
    raw_EEG_path = os.path.join(data_path, f'sub-{vis_subj_id}', 'eeg')
    vhdr_files = sorted(f for f in
                        [os.path.basename(x) for x in glob.glob(os.path.join(raw_EEG_path, "*.vhdr"))]
                        if "cpt" in f.lower())
    if not vhdr_files:
        print(f"sub-{vis_subj_id}: no gradCPT vhdr files found, skipping.")
        continue

    # accumulators for eeg_preproc_subj_level-compatible output
    single_subj_EEG_dict = dict()
    single_subj_rm_ch_dict = dict()

    # delete existing figures before replotting
    if is_overwrite:
        _vis_dir_batch = os.path.join(data_save_path, f"sub-{vis_subj_id}", "preprocess_visualization")
        if os.path.isdir(_vis_dir_batch):
            for _f in glob.glob(os.path.join(_vis_dir_batch, "*.png")):
                os.remove(_f)
            print(f"sub-{vis_subj_id}: cleared existing figures in {_vis_dir_batch}")

    for vhdr_file in vhdr_files:
        # derive BIDS-style run label from filename (e.g. run-01, run-02, run-03)
        run_match = re.search(r'run-(\d+)', vhdr_file, re.IGNORECASE)
        run_label = f"run-{int(run_match.group(1)):02d}" if run_match else "run-00"
        prefix = f"sub-{vis_subj_id}_{run_label}"

        # output folder
        save_dir = os.path.join(data_save_path, f"sub-{vis_subj_id}", "preprocess_visualization")
        os.makedirs(save_dir, exist_ok=True)

        # skip if all figures already exist
        expected_figs = [
            f"{prefix}_step0_raw_psd.png", f"{prefix}_step0_raw_timeseries.png",
            f"{prefix}_step1_band_pass_psd.png", f"{prefix}_step1_band_pass_timeseries.png",
            f"{prefix}_step2_bad_channels.png",
            f"{prefix}_step3_pre_reref_mastoid_timeseries.png",
            f"{prefix}_step3_reref_psd.png", f"{prefix}_step3_reref_timeseries.png",
            f"{prefix}_step4_ICA_scores.png",
            f"{prefix}_step4_ICA_psd.png", f"{prefix}_step4_ICA_timeseries.png",
        ]
        if not is_overwrite and all(os.path.exists(os.path.join(save_dir, f)) for f in expected_figs):
            print(f"{prefix}: figures exist, skipping.")
            continue

        print(f"Visualizing {prefix} ...")
        try:
            EEG_raw = fix_and_load_brainvision(os.path.join(raw_EEG_path, vhdr_file))
        except Exception as e:
            print(f"  Could not load {vhdr_file}: {e}")
            continue

        # Step 0 – raw ────────────────────────────────────────────────────────
        eeg_trigger = EEG_raw.get_data()[4]
        thres_trigger = (np.max(eeg_trigger) - np.min(eeg_trigger)) / 2 + np.min(eeg_trigger)
        # check if the trigger goes to 0V or -0.9V when being pressed
        press_period = eeg_trigger < thres_trigger
        if np.sum(press_period) > np.sum(~press_period):
            press_period = ~press_period
        eeg_duration = np.max(np.diff(np.where(press_period)[0])) / EEG_raw.info["sfreq"] / 60
        print(f"  Valid recording length: {eeg_duration:.1f} min")

        _savefig(_make_psd_fig(EEG_raw, ch_names, f"{prefix} – Raw PSD"),
                 save_dir, f"{prefix}_step0_raw_psd.png")
        _savefig(_make_timeseries_fig(EEG_raw, ch_names, f"{prefix} – Raw time series"),
                 save_dir, f"{prefix}_step0_raw_timeseries.png")

        # Step 1 – band-pass filter ───────────────────────────────────────────
        EEG_step1 = EEG_raw.copy()
        if is_bpfilter:
            EEG_step1.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1], picks='all', verbose=False)

        _savefig(_make_psd_fig(EEG_step1, ch_names, f"{prefix} – Band-pass {bp_f_range} Hz PSD"),
                 save_dir, f"{prefix}_step1_band_pass_psd.png")
        _savefig(_make_timeseries_fig(EEG_step1, ch_names, f"{prefix} – Band-pass time series"),
                 save_dir, f"{prefix}_step1_band_pass_timeseries.png")

        # Step 2 – bad channel detection and removal ──────────────────────────
        rm_ch_list = list(set(check_flat_channels(EEG_step1) +
                              list(check_abnormal_var_channels(EEG_step1)) +
                              check_large_amp_channels(EEG_step1)))
        print(f"  Removed channels: {rm_ch_list}")
        # recompute for variance bar chart
        eeg_data = EEG_step1.get_data(picks='eeg')
        eeg_chs  = np.array([x["ch_name"] for x in EEG_step1.info["chs"] if x["kind"] == 2])
        eeg_var  = np.var(eeg_data, axis=1)

        fig_var, ax_var = plt.subplots(figsize=(12, 4))
        ax_var.bar(eeg_chs, eeg_var * 1e12,
                   color=["red" if c in rm_ch_list else "steelblue" for c in eeg_chs])
        ax_var.set_xlabel("Channel"); ax_var.set_ylabel("Variance (µV²)")
        ax_var.set_title(f"{prefix} – Channel variance (red = removed: {rm_ch_list})")
        plt.xticks(rotation=90, fontsize=6); plt.tight_layout()
        _savefig(fig_var, save_dir, f"{prefix}_step2_bad_channels.png")

        EEG_step2 = EEG_step1.copy()
        if rm_ch_list:
            EEG_step2.drop_channels(rm_ch_list)

        # time series of mastoid reference channels before re-reference
        mastoid_chs = [c for c in ['tp9h', 'tp10h'] if c in EEG_step2.ch_names]
        if mastoid_chs:
            _savefig(_make_timeseries_fig(EEG_step2, mastoid_chs,
                                          f"{prefix} – Mastoid (tp9h/tp10h) before re-reference"),
                     save_dir, f"{prefix}_step2_pre_reref_mastoid_timeseries.png")

        # Step 3 – re-reference ───────────────────────────────────────────────
        EEG_step3 = EEG_step2.copy()
        if is_reref:
            ref = reref_ch if reref_ch else 'average'
            EEG_step3.set_eeg_reference(ref_channels=ref, ch_type='eeg', verbose=False)

        _savefig(_make_psd_fig(EEG_step3, ch_names, f"{prefix} – Re-reference PSD"),
                 save_dir, f"{prefix}_step3_reref_psd.png")
        _savefig(_make_timeseries_fig(EEG_step3, ch_names, f"{prefix} – Re-reference time series"),
                 save_dir, f"{prefix}_step3_reref_timeseries.png")

        # Step 4 – ICA eye-artifact removal ───────────────────────────────────
        EEG_step4 = EEG_step3.copy()
        if is_ica_rmEye:
            try:
                n_eeg = EEG_step4.info.get_channel_types().count('eeg')
                ica = mne.preprocessing.ICA(n_components=n_eeg, method='infomax',
                                            random_state=42, verbose=False)
                ica.fit(EEG_step4, picks=['eeg'], verbose=False)
                eog_inds, eog_scores = ica.find_bads_eog(EEG_step4, ch_name=['hEOG', 'vEOG'],
                                                          measure='correlation', threshold=0.9,
                                                          verbose=False)
                print(f"  ICA eye components: {eog_inds}")

                # ICA scores
                fig_scores = ica.plot_scores(eog_scores, exclude=eog_inds, show=False)
                fig_scores.suptitle(f"{prefix} – ICA EOG scores"); plt.tight_layout()
                _savefig(fig_scores, save_dir, f"{prefix}_step4_ICA_scores.png")

                # ICA topomaps for excluded components
                if eog_inds:
                    try:
                        fig_comp = ica.plot_components(picks=eog_inds, show=False)
                        if not isinstance(fig_comp, list):
                            fig_comp = [fig_comp]
                        for fi, fc in enumerate(fig_comp):
                            fc.suptitle(f"{prefix} – ICA excluded components"); plt.tight_layout()
                            _savefig(fc, save_dir, f"{prefix}_step4_ICA_components_{fi}.png")
                    except Exception as e:
                        # QhullError: too few channels remaining for topomap triangulation
                        print(f"  Could not plot ICA components ({type(e).__name__}): {e}")

                ica.exclude = eog_inds
                EEG_step4 = ica.apply(EEG_step4, verbose=False)
                EEG_step4._data[4] = eeg_trigger
            except Exception as e:
                print(f"  ICA failed: {e}")
                # save a placeholder scores figure showing the error
                fig_err, ax_err = plt.subplots()
                ax_err.text(0.5, 0.5, f"ICA failed:\n{e}", ha='center', va='center', wrap=True)
                ax_err.axis('off')
                _savefig(fig_err, save_dir, f"{prefix}_step4_ICA_scores.png")

        _savefig(_make_psd_fig(EEG_step4, ch_names, f"{prefix} – Post-ICA PSD"),
                 save_dir, f"{prefix}_step4_ICA_psd.png")
        _savefig(_make_timeseries_fig(EEG_step4, ch_names, f"{prefix} – Post-ICA time series"),
                 save_dir, f"{prefix}_step4_ICA_timeseries.png")

        # save preprocessed EEG as .fif
        fif_dir = os.path.join(data_save_path, f"sub-{vis_subj_id}")
        os.makedirs(fif_dir, exist_ok=True)
        fif_path = os.path.join(fif_dir, f"sub-{vis_subj_id}_task-gradCPT_{run_label}_preproc_eeg.fif")
        if is_overwrite or not os.path.exists(fif_path):
            EEG_step4.save(fif_path, overwrite=True, verbose=False)
            print(f"  Saved preprocessed EEG → {fif_path}")
        else:
            print(f"  Preprocessed EEG already exists, skipping: {fif_path}")

        print(f"  Saved to {save_dir}")

        # accumulate per-run results (key format matches eeg_preproc_subj_level output)
        run_digit = vhdr_file.split('.')[0][-1]
        run_key = "gradcpt" + run_digit
        single_subj_EEG_dict[run_key] = EEG_step4
        single_subj_rm_ch_dict[run_key] = rm_ch_list

    # ── save subject-level preprocessed dicts ────────────────────────────────
    subj_save_dir = os.path.join(data_save_path, f"sub-{vis_subj_id}")
    os.makedirs(subj_save_dir, exist_ok=True)
    gz_path = os.path.join(subj_save_dir, f"sub-{vis_subj_id}_preprocessed_dict.pkl.gz")

    if not is_overwrite and os.path.exists(gz_path):
        print(f"sub-{vis_subj_id}: preprocessed dict already exists, skipping save.")
    else:
        payload = dict(EEG=single_subj_EEG_dict, rm_ch=single_subj_rm_ch_dict)
        with gzip.open(gz_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"sub-{vis_subj_id}: saved preprocessed dict → {gz_path}")

print("Preprocessing visualization complete for all subjects.")

#%% Check how many EEG available
_eeg_summary = []
for _d in sorted(glob.glob(os.path.join(data_path, "sub-*"))):
    _sid = os.path.basename(_d)
    _eeg_dir = os.path.join(_d, "eeg")
    if not os.path.isdir(_eeg_dir):
        _eeg_summary.append((_sid, 0, []))
        continue
    _runs = sorted(os.path.basename(x) for x in glob.glob(os.path.join(_eeg_dir, "*.vhdr"))
                   if "cpt" in os.path.basename(x).lower())
    _eeg_summary.append((_sid, len(_runs), _runs))

_with_eeg    = [s for s in _eeg_summary if s[1] > 0]
_without_eeg = [s[0] for s in _eeg_summary if s[1] == 0]

print(f"Subjects with gradCPT EEG: {len(_with_eeg)} / {len(_eeg_summary)}")
for _sid, _n, _runs in _with_eeg:
    print(f"  {_sid}: {_n} run(s) — {[r.split('.')[0] for r in _runs]}")
print(f"\nSubjects without EEG folder: {_without_eeg}")

#%% preprocess single subject — all gradCPT runs
subj_id = 765

_raw_eeg_path = os.path.join(data_path, f"sub-{subj_id}", "eeg")
_save_dir     = os.path.join(data_save_path, f"sub-{subj_id}")
_gz_path      = os.path.join(_save_dir, f"sub-{subj_id}_preprocessed_dict.pkl.gz")
os.makedirs(_save_dir, exist_ok=True)

_vhdr_files = sorted(f for f in
                     [os.path.basename(x) for x in glob.glob(os.path.join(_raw_eeg_path, "*.vhdr"))]
                     if "cpt" in f.lower())

# load existing dict so runs already saved are preserved
if os.path.exists(_gz_path):
    with gzip.open(_gz_path, 'rb') as f:
        _existing = pickle.load(f)
    _single_subj_EEG_dict   = _existing["EEG"]
    _single_subj_rm_ch_dict = _existing["rm_ch"]
    print(f"Loaded existing dict with runs: {list(_single_subj_EEG_dict.keys())}")
else:
    _single_subj_EEG_dict   = dict()
    _single_subj_rm_ch_dict = dict()

# delete existing figures before replotting
_vis_dir = os.path.join(_save_dir, "preprocess_visualization")
if is_overwrite and os.path.isdir(_vis_dir):
    for _f in glob.glob(os.path.join(_vis_dir, "*.png")):
        os.remove(_f)
    print(f"sub-{subj_id}: cleared existing figures in {_vis_dir}")

for _vhdr_file in _vhdr_files:
    _run_match = re.search(r'run-(\d+)', _vhdr_file, re.IGNORECASE)
    _run_digit = str(int(_run_match.group(1))) if _run_match else '0'
    _run_label = f"run-{int(_run_digit):02d}"
    _run_key   = "gradcpt" + _run_digit
    _prefix    = f"sub-{subj_id}_{_run_label}"
    _vis_dir   = os.path.join(_save_dir, "preprocess_visualization")
    os.makedirs(_vis_dir, exist_ok=True)

    if not is_overwrite and _run_key in _single_subj_EEG_dict:
        print(f"  {_run_key}: already in dict, skipping.")
        continue
    print(f"Preprocessing sub-{subj_id} {_run_key} ...")

    # ── Step 0: load raw ──────────────────────────────────────────────────────
    EEG_raw = fix_and_load_brainvision(os.path.join(_raw_eeg_path, _vhdr_file))
    eeg_trigger = EEG_raw.get_data()[4]
    thres_trigger = (np.max(eeg_trigger) - np.min(eeg_trigger)) / 2 + np.min(eeg_trigger)
    # check if the trigger goes to 0V or -0.9V when being pressed
    press_period = eeg_trigger < thres_trigger
    if np.sum(press_period) > np.sum(~press_period):
        press_period = ~press_period
    eeg_duration = np.max(np.diff(np.where(press_period)[0])) / EEG_raw.info["sfreq"] / 60
    print(f"  Valid recording length: {eeg_duration:.1f} min")

    _savefig(_make_psd_fig(EEG_raw, ch_names, f"{_prefix} – Raw PSD"),
             _vis_dir, f"{_prefix}_step0_raw_psd.png")
    _savefig(_make_timeseries_fig(EEG_raw, ch_names, f"{_prefix} – Raw time series"),
             _vis_dir, f"{_prefix}_step0_raw_timeseries.png")

    # ── Step 1: band-pass filter ──────────────────────────────────────────────
    EEG_step1 = EEG_raw.copy()
    if is_bpfilter:
        EEG_step1.filter(l_freq=bp_f_range[0], h_freq=bp_f_range[1], picks='all', verbose=False)

    _savefig(_make_psd_fig(EEG_step1, ch_names, f"{_prefix} – Band-pass {bp_f_range} Hz PSD"),
             _vis_dir, f"{_prefix}_step1_band_pass_psd.png")
    _savefig(_make_timeseries_fig(EEG_step1, ch_names, f"{_prefix} – Band-pass time series"),
             _vis_dir, f"{_prefix}_step1_band_pass_timeseries.png")

    # ── Step 2: detect and drop bad channels ──────────────────────────────────
    rm_ch_list = list(set(check_flat_channels(EEG_step1) +
                          list(check_abnormal_var_channels(EEG_step1)) +
                          check_large_amp_channels(EEG_step1)))
    print(f"  Removed channels: {rm_ch_list}")
    # recompute for variance bar chart
    eeg_data = EEG_step1.get_data(picks='eeg')
    eeg_chs  = np.array([x["ch_name"] for x in EEG_step1.info["chs"] if x["kind"] == 2])
    eeg_var  = np.var(eeg_data, axis=1)

    fig_var, ax_var = plt.subplots(figsize=(12, 4))
    ax_var.bar(eeg_chs, eeg_var * 1e12,
               color=["red" if c in rm_ch_list else "steelblue" for c in eeg_chs])
    ax_var.set_xlabel("Channel"); ax_var.set_ylabel("Variance (µV²)")
    ax_var.set_title(f"{_prefix} – Channel variance (red = removed: {rm_ch_list})")
    plt.xticks(rotation=90, fontsize=6); plt.tight_layout()
    _savefig(fig_var, _vis_dir, f"{_prefix}_step2_bad_channels.png")

    EEG_step2 = EEG_step1.copy()
    if rm_ch_list:
        EEG_step2.drop_channels(rm_ch_list)

    # mastoid channels before re-reference
    mastoid_chs = [c for c in ['tp9h', 'tp10h'] if c in EEG_step2.ch_names]
    if mastoid_chs:
        _savefig(_make_timeseries_fig(EEG_step2, mastoid_chs,
                                      f"{_prefix} – Mastoid (tp9h/tp10h) before re-reference"),
                 _vis_dir, f"{_prefix}_step2_pre_reref_mastoid_timeseries.png")

    # ── Step 3: re-reference ──────────────────────────────────────────────────
    EEG_step3 = EEG_step2.copy()
    if is_reref:
        ref = reref_ch if reref_ch else 'average'
        EEG_step3.set_eeg_reference(ref_channels=ref, ch_type='eeg', verbose=False)

    _savefig(_make_psd_fig(EEG_step3, ch_names, f"{_prefix} – Re-reference PSD"),
             _vis_dir, f"{_prefix}_step3_reref_psd.png")
    _savefig(_make_timeseries_fig(EEG_step3, ch_names, f"{_prefix} – Re-reference time series"),
             _vis_dir, f"{_prefix}_step3_reref_timeseries.png")

    # ── Step 4: ICA eye-artifact removal ─────────────────────────────────────
    EEG_step4 = EEG_step3.copy()
    if is_ica_rmEye:
        n_eeg = EEG_step4.info.get_channel_types().count('eeg')
        ica = mne.preprocessing.ICA(n_components=n_eeg, method='infomax', random_state=42, verbose=False)
        ica.fit(EEG_step4, picks=['eeg'], verbose=False)
        eog_inds, eog_scores = ica.find_bads_eog(EEG_step4, ch_name=['hEOG', 'vEOG'],
                                                  measure='correlation', threshold=0.9,
                                                  verbose=False)
        print(f"  ICA eye components removed: {eog_inds}")

        fig_scores = ica.plot_scores(eog_scores, exclude=eog_inds, show=False)
        fig_scores.suptitle(f"{_prefix} – ICA EOG scores"); plt.tight_layout()
        _savefig(fig_scores, _vis_dir, f"{_prefix}_step4_ICA_scores.png")

        if eog_inds:
            try:
                fig_comp = ica.plot_components(picks=eog_inds, show=False)
                if not isinstance(fig_comp, list):
                    fig_comp = [fig_comp]
                for fi, fc in enumerate(fig_comp):
                    fc.suptitle(f"{_prefix} – ICA excluded components"); plt.tight_layout()
                    _savefig(fc, _vis_dir, f"{_prefix}_step4_ICA_components_{fi}.png")
            except Exception as e:
                # QhullError: too few channels remaining for topomap triangulation
                print(f"  Could not plot ICA components ({type(e).__name__}): {e}")

        ica.exclude = eog_inds
        EEG_step4 = ica.apply(EEG_step4, verbose=False)
        EEG_step4._data[4] = eeg_trigger

    _savefig(_make_psd_fig(EEG_step4, ch_names, f"{_prefix} – Post-ICA PSD"),
             _vis_dir, f"{_prefix}_step4_ICA_psd.png")
    _savefig(_make_timeseries_fig(EEG_step4, ch_names, f"{_prefix} – Post-ICA time series"),
             _vis_dir, f"{_prefix}_step4_ICA_timeseries.png")

    print(f"  Figures saved to {_vis_dir}")

    # save preprocessed EEG as .fif
    _fif_path = os.path.join(_save_dir, f"sub-{subj_id}_task-gradCPT_{_run_label}_preproc_eeg.fif")
    if is_overwrite or not os.path.exists(_fif_path):
        EEG_step4.save(_fif_path, overwrite=True, verbose=False)
        print(f"  Saved preprocessed EEG → {_fif_path}")
    else:
        print(f"  Preprocessed EEG already exists, skipping: {_fif_path}")

    _single_subj_EEG_dict[_run_key]   = EEG_step4
    _single_subj_rm_ch_dict[_run_key] = rm_ch_list

# ── save ──────────────────────────────────────────────────────────────────────
print(f"Runs in dict: {sorted(_single_subj_EEG_dict.keys())}")
_payload = dict(EEG=_single_subj_EEG_dict, rm_ch=_single_subj_rm_ch_dict)
with gzip.open(_gz_path, 'wb') as f:
    pickle.dump(_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved → {_gz_path}")
