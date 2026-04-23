#%% load library
import numpy as np
import scipy as sp
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
# import utils
from params_setting import *
from utils import smoothing_VTC_gaussian_array
sys.path.append("/projectnb/nphfnirs/s/datasets/gradCPT_NN24/code/eyetracking")
from pupil_labs import neon_recording as nr
import re
from utils_eyetracking import preprocess_pupil, _build_gaussian_basis, _convolve_onsets
import warnings


#%%
f_lowpass=6
f_downsample=60
detrend_order=1
win_trials=20
step_trials=5
include_tot=False

subj = 'sub-746'
subj_id       = subj.replace('sub-', '')
subj_nirs_dir = os.path.join(project_path, subj, 'nirs')
subj_neon_dir = os.path.join(project_path, 'sourcedata', 'raw', subj, 'eye_tracking')
neon_dirs_subj = sorted([d for d in os.listdir(subj_neon_dir) if re.match(r'\d{4}-', d)])

run_id = 1
dirs = os.listdir(project_path)
subject_list = sorted([d for d in dirs if 'sub' in d])
physio_file = os.path.join(subj_nirs_dir,
                    f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio_20260423.tsv")
if not os.path.isfile(physio_file):
    physio_file = os.path.join(subj_nirs_dir,
        f"{subj}_task-gradCPT_run-{run_id:02d}_recording-eyetracking_physio.tsv")
event_file = os.path.join(subj_nirs_dir,
    f"{subj}_task-gradCPT_run-{run_id:02d}_events.tsv")

neon_data  = pd.read_csv(physio_file, sep='\t')

# Neon recording for blink removal
rec = None
if neon_dirs_subj and run_id - 1 < len(neon_dirs_subj):
    try:
        rec = nr.open(os.path.join(subj_neon_dir, neon_dirs_subj[run_id - 1]))
    except Exception:
        pass

#%% Replay each preprocessing step and visualize
def build_snapshots(neon_data, rec, order, f_lowpass, f_downsample):
    t_neon = neon_data['timestamps']
    pupil_raw = (neon_data['eyeleft_pupilDiameter'] + neon_data['eyeright_pupilDiameter']) / 2
    pupil_d = pupil_raw.values.copy().astype(float)

    snapshots = [('Raw (avg L+R)', t_neon.values, pupil_d.copy())]

    t_neon_arr = neon_data['timestamps_neon']
    t_blink_start = (rec.blinks["start_time"] - rec.start_time) / 1e9
    t_blink_stop  = (rec.blinks["stop_time"]  - rec.start_time) / 1e9
    for t_start, t_stop in zip(t_blink_start, t_blink_stop):
        mask = (t_neon_arr >= t_start) & (t_neon_arr <= t_stop)
        pupil_d[mask] = np.nan
    valid = ~np.isnan(pupil_d)
    pupil_d = np.interp(t_neon, t_neon[valid], pupil_d[valid])
    snapshots.append(('After linear interpolation', t_neon.values, pupil_d.copy()))

    _t_idx = np.arange(len(pupil_d), dtype=float)
    _valid = np.where(~np.isnan(pupil_d))[0]
    _coef  = np.polyfit(_t_idx[_valid], pupil_d[_valid], order)
    _trend = np.polyval(_coef, _t_idx)
    snapshots.append((f'Before poly detrend (order={order})', t_neon.values, pupil_d.copy(), _trend))
    pupil_d = pupil_d - _trend
    snapshots.append((f'After poly detrend (order={order})', t_neon.values, pupil_d.copy(), None))

    t_neon_arr = t_neon.values
    fs = 1.0 / np.median(np.diff(t_neon_arr))
    sos = sp.signal.butter(4, f_lowpass, btype='low', fs=fs, output='sos')
    pupil_d = sp.signal.sosfiltfilt(sos, pupil_d)
    snapshots.append((f'After lowpass filter ({f_lowpass} Hz)', t_neon_arr, pupil_d.copy()))

    t_new = np.arange(t_neon_arr[0], t_neon_arr[-1], 1.0 / f_downsample)
    pupil_d = np.interp(t_new, t_neon_arr, pupil_d)
    snapshots.append((f'After downsample ({f_downsample} Hz)', t_new, pupil_d.copy()))

    return snapshots

snapshots_col1 = build_snapshots(neon_data, rec, order=1,          f_lowpass=f_lowpass, f_downsample=f_downsample)
snapshots_col2 = build_snapshots(neon_data, rec, order=2,          f_lowpass=f_lowpass, f_downsample=f_downsample)

#%% Plot
n_rows = len(snapshots_col1)
fig, axes = plt.subplots(n_rows, 2, figsize=(18, 3 * n_rows), sharex=False)
fig.suptitle(f'{subj} run-{run_id:02d} — pupil preprocessing steps', fontsize=13)

for col_idx, (snapshots, order_label) in enumerate([(snapshots_col1, 'order=1'), (snapshots_col2, 'order=2')]):
    for row_idx, snap in enumerate(snapshots):
        ax = axes[row_idx, col_idx]
        label, t, sig = snap[0], snap[1], snap[2]
        trend = snap[3] if len(snap) > 3 else None
        ax.plot(t, sig, lw=0.6)
        if trend is not None:
            ax.plot(t, trend, lw=1.5, color='red', linestyle='--', label='poly fit')
            ax.legend(fontsize=8)
        ax.set_title(f'[{order_label}] {label}', fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Diameter')
        ax.autoscale(axis='x', tight=True)

plt.tight_layout()
plt.show()

# %%
t_pupil,pupil_tonic = preprocess_pupil(
                    neon_data, rec=rec,
                    f_lowpass=f_lowpass,
                    f_downsample=f_downsample,
                    detrend_order=detrend_order,
                    is_rm_phasic=False,
                    events_df=events_df,
                )