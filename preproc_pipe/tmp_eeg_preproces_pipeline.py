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

ch_names = ['Fz','Cz','Pz','Oz']
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_check_flat = True # check if there are flat channels in EEG
is_check_ch_var = True # check if there are channels with abnormal var
is_check_amp = False # check if EEG amplitude exceed 1000 muV
is_reref = False
reref_ch = ['tp9h','tp10h']
# reref_ch = None # reref to average
is_ica_rmEye = True
select_event = "mnt_correct"
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None
is_overwrite = True # Force to re run preprocessing if it is True

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

#%% preprocess single subject — all gradCPT runs
is_overwrite = True
ch_names = ['Fz','Cz','Pz','Oz']
subj_id = 'Easycap'
data_path = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/sourcedata/raw"
data_save_path = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/derivatives"
_raw_eeg_path = os.path.join(data_path, f"sub-{subj_id}", "eeg")
_save_dir     = os.path.join(data_path, f"sub-{subj_id}","eeg","preprocessed")
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

    # generate events.tsv if missing
    _events_tsv_path = os.path.join(
        _save_dir,
        f"sub-{subj_id}_task-gradCPT_{_run_label}_events.tsv"
    )
    if not os.path.exists(_events_tsv_path):
        print(f"  Generating events.tsv for sub-{subj_id} {_run_label} ...")
        try:
            gen_EEG_event_tsv(subj_id,savepath=_save_dir,gradcpt_path=os.path.join(data_path, f"sub-{subj_id}", "gradCPT"))
        except Exception as e:
            print(f"  Could not generate events.tsv: {e}")

    print(f"Preprocessing sub-{subj_id} {_run_key} ...")

    # ── Step 0: load raw ──────────────────────────────────────────────────────
    EEG_raw = fix_and_load_brainvision(os.path.join(_raw_eeg_path, _vhdr_file))
    EEG_raw.set_channel_types({'Trigger': 'misc'})
    eeg_trigger = EEG_raw.get_data(picks='Trigger')
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
    rm_ch_list = []
    if is_check_flat:
        rm_ch_list.extend(check_flat_channels(EEG_step1))
    if is_check_ch_var:
        rm_ch_list.extend(list(check_abnormal_var_channels(EEG_step1)))
    if is_check_amp:
        rm_ch_list.extend(check_large_amp_channels(EEG_step1))
    rm_ch_list = set(rm_ch_list)
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

# %%
