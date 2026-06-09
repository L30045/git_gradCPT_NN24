#%%
import os
import glob
import re
import numpy as np
import pandas as pd
import mne

project_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24'
eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')

# epoch parameters matching preproc pipeline
EPOCH_TMIN = -0.2
EPOCH_TMAX = 1.0
EPOCH_REJECT = dict(eeg=100e-6)
MIN_EPOCHS = 500  # threshold for "enough epochs"

# ── 0. fNIRS ─────────────────────────────────────────────────────────────────
fnirs_files = glob.glob(
    os.path.join(project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf')
)
fnirs_subjects = sorted({
    re.search(r'sub-(\d+)', f).group(1)
    for f in fnirs_files
    if re.search(r'sub-(\d+)', f)
})
print("=" * 60)
print(f"Subjects with fNIRS (N={len(fnirs_subjects)}):")
print("  " + ", ".join(f"sub-{s}" for s in fnirs_subjects))

# ── 1. Pupil data ────────────────────────────────────────────────────────────
physio_files = glob.glob(
    os.path.join(project_path, 'sub-*', 'nirs',
                 '*task-gradCPT*eyetracking_physio.tsv')
)
pupil_subjects = sorted({
    re.search(r'sub-(\d+)', f).group(1)
    for f in physio_files
    if re.search(r'sub-(\d+)', f)
})
print("=" * 60)
print(f"Subjects with pupil data (N={len(pupil_subjects)}):")
print("  " + ", ".join(f"sub-{s}" for s in pupil_subjects))

# ── 2. Resting EEG ───────────────────────────────────────────────────────────
rest_files = glob.glob(
    os.path.join(eeg_deriv, 'sub-*', '*task-Rest*preproc_eeg.fif')
)
rest_subjects = sorted({
    re.search(r'sub-(\d+)', f).group(1)
    for f in rest_files
    if re.search(r'sub-(\d+)', f)
})
print()
print(f"Subjects with resting EEG (N={len(rest_subjects)}):")
print("  " + ", ".join(f"sub-{s}" for s in rest_subjects))

# ── 2b. Resting-state physio (pupil) ─────────────────────────────────────────
rs_physio_subjects = []
for sid in rest_subjects:
    nirs_dir = os.path.join(project_path, f'sub-{sid}', 'nirs')
    found = any(
        os.path.isfile(os.path.join(nirs_dir, fname))
        for fname in [
            f'sub-{sid}_task-RS_run-01_recording-eyetracking_physio_20260423.tsv',
            f'sub-{sid}_task-RS_run-01_recording-eyetracking_physio_20260311_correct_idx.tsv',
            f'sub-{sid}_task-RS_run-01_recording-eyetracking_physio.tsv',
        ]
    )
    if found:
        rs_physio_subjects.append(sid)
print()
print(f"Subjects with resting EEG + RS physio (N={len(rs_physio_subjects)}):")
print("  " + ", ".join(f"sub-{s}" for s in rs_physio_subjects))
no_rs_physio = [s for s in rest_subjects if s not in rs_physio_subjects]
if no_rs_physio:
    print(f"  (missing RS physio: {', '.join(f'sub-{s}' for s in no_rs_physio)})")

# ── 3. GradCPT EEG with enough epochs ────────────────────────────────────────
# Events are stored in companion *_events.tsv files (not in the fif)
gradcpt_fif_files = sorted(glob.glob(
    os.path.join(eeg_deriv, 'sub-*', '*task-gradCPT*preproc_eeg.fif')
))
subj_to_fifs = {}
for f in gradcpt_fif_files:
    m = re.search(r'sub-(\d+)', f)
    if m:
        subj_to_fifs.setdefault(m.group(1), []).append(f)

print()
print(f"Counting GradCPT epochs (reject threshold: {EPOCH_REJECT['eeg']*1e6:.0f} µV, "
      f"min epochs per subject: {MIN_EPOCHS}) ...")

subj_epoch_counts = {}
for sid in sorted(subj_to_fifs):
    total = 0
    for fif in sorted(subj_to_fifs[sid]):
        # companion events file
        events_tsv = fif.replace('_preproc_eeg.fif', '_events.tsv')
        if not os.path.exists(events_tsv):
            continue
        ev_df = pd.read_csv(events_tsv, sep='\t')
        # keep all trials (both city and mountain)
        onsets_sec = ev_df['onset'].values
        if len(onsets_sec) == 0:
            continue
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        events_arr = np.column_stack([
            (onsets_sec * sfreq).astype(int),
            np.zeros(len(onsets_sec), dtype=int),
            np.ones(len(onsets_sec), dtype=int),
        ])
        # clip events to valid range
        n_times = raw.n_times
        valid = (events_arr[:, 0] >= 0) & (events_arr[:, 0] < n_times)
        events_arr = events_arr[valid]
        if len(events_arr) == 0:
            continue
        epochs = mne.Epochs(
            raw, events_arr, event_id=1,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None, preload=True, verbose=False,
        )
        epochs.drop_bad(reject=EPOCH_REJECT, verbose=False)
        total += len(epochs)
    subj_epoch_counts[sid] = total

enough = {sid: n for sid, n in subj_epoch_counts.items() if n >= MIN_EPOCHS}
print()
print(f"Subjects with GradCPT EEG + ≥{MIN_EPOCHS} good epochs (N={len(enough)}):")
for sid, n in sorted(enough.items()):
    print(f"  sub-{sid}: {n} epochs")

not_enough = {sid: n for sid, n in subj_epoch_counts.items() if n < MIN_EPOCHS}
if not_enough:
    print()
    print(f"Subjects with GradCPT EEG but <{MIN_EPOCHS} good epochs (N={len(not_enough)}):")
    for sid, n in sorted(not_enough.items()):
        print(f"  sub-{sid}: {n} epochs")

#%% ── Summary ───────────────────────────────────────────────────────────────────
pupil_gradcpt = sorted(
    set(pupil_subjects) & set(enough.keys())
)
all_three = sorted(
    set(pupil_subjects) & set(rest_subjects) & set(enough.keys())
)
fnirs_gradcpt = sorted(
    set(fnirs_subjects) & set(enough.keys())
)
fnirs_pupil_gradcpt = sorted(
    set(fnirs_subjects) & set(pupil_subjects) & set(enough.keys())
)
all_four = sorted(
    set(fnirs_subjects) & set(pupil_subjects) & set(rest_subjects) & set(enough.keys())
)
print()
print("=" * 60)
print(f"Subjects with fNIRS + GradCPT EEG ≥{MIN_EPOCHS} epochs (N={len(fnirs_gradcpt)}):")
print("  " + ", ".join(f"sub-{s}" for s in fnirs_gradcpt))
print()
print(f"Subjects with pupil + GradCPT EEG ≥{MIN_EPOCHS} epochs (N={len(pupil_gradcpt)}):")
print("  " + ", ".join(f"sub-{s}" for s in pupil_gradcpt))
print()
print(f"Subjects with ALL THREE (pupil + rest EEG + GradCPT EEG ≥{MIN_EPOCHS} epochs) "
      f"(N={len(all_three)}):")
print("  " + ", ".join(f"sub-{s}" for s in all_three))
print()
print(f"Subjects with fNIRS + pupil + GradCPT EEG ≥{MIN_EPOCHS} epochs (N={len(fnirs_pupil_gradcpt)}):")
print("  " + ", ".join(f"sub-{s}" for s in fnirs_pupil_gradcpt))
print()
print(f"Subjects with ALL FOUR (fNIRS + pupil + rest EEG + GradCPT EEG ≥{MIN_EPOCHS} epochs) "
      f"(N={len(all_four)}):")
print("  " + ", ".join(f"sub-{s}" for s in all_four))
