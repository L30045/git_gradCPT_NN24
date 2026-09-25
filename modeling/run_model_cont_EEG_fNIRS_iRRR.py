#%% load library
import numpy as np
import pickle
import copy
import gzip
import glob
import time
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.io
import cedalion.models.glm as glm
from cedalion.sigproc import frequency
from statsmodels.gam.smooth_basis import BSplines
from scipy.signal import butter, sosfiltfilt, filtfilt, windows
from scipy.linalg import svdvals
from numpy.linalg import matrix_rank
sys.path.append('/projectnb/stephenlab/cchang1/iRRR_python')
from iRRR.iRRR_normal import irrr_normal

#%% mask out low-sensitivity parcels using the forward-model sensitivity matrix
Adot_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/fw/probe/'
Adot = cedalion.io.load_Adot(Adot_path + 'Adot_v26.nc')

Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
mask_medial = Adot_brain.parcel.isin([
    'Background+FreeSurfer_Defined_Medial_Wall_LH',
    'Background+FreeSurfer_Defined_Medial_Wall_RH'
])
Adot_brain = Adot_brain.sel(vertex=~mask_medial)

intensity = np.log10(Adot_brain[:, :, 1].sum('channel'))
sensitivity_mask = (intensity > -2).drop_vars('wavelength')

Adot_brain_sens = Adot_brain.sel(vertex=sensitivity_mask.values)
Adot_parcel = Adot_brain_sens.groupby('parcel').sum('vertex')
Adot_parcel = Adot_parcel.assign_coords(
    {'is_brain': ('parcel', np.ones(len(Adot_parcel.parcel), dtype=bool))}
)
sensitive_parcels = Adot_parcel.parcel.values  # parcels surviving the sensitivity mask (429 of 601)

#%% find subjects with fNIRS and enough EEG epochs
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')
_MIN_EPOCHS = 500

_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf'))
    if re.search(r'sub-(\d+)', f)
}

_gradcpt_fifs = sorted(glob.glob(os.path.join(_eeg_deriv, 'sub-*', '*task-gradCPT*preproc_eeg.fif')))
_subj_to_fifs = {}
for _f in _gradcpt_fifs:
    _m = re.search(r'sub-(\d+)', _f)
    if _m:
        _subj_to_fifs.setdefault(_m.group(1), []).append(_f)

_subj_epoch_counts = {}
for _sid in sorted(_subj_to_fifs):
    _total = 0
    for _fif in sorted(_subj_to_fifs[_sid]):
        _events_tsv = _fif.replace('_preproc_eeg.fif', '_events.tsv')
        if not os.path.exists(_events_tsv):
            continue
        _ev_df = pd.read_csv(_events_tsv, sep='\t')
        _onsets = _ev_df['onset'].values
        if len(_onsets) == 0:
            continue
        _raw = mne.io.read_raw_fif(_fif, preload=True, verbose=False)
        _events_arr = np.column_stack([
            (_onsets * _raw.info['sfreq']).astype(int),
            np.zeros(len(_onsets), dtype=int),
            np.ones(len(_onsets), dtype=int),
        ])
        _valid = (_events_arr[:, 0] >= 0) & (_events_arr[:, 0] < _raw.n_times)
        _events_arr = _events_arr[_valid]
        if len(_events_arr) == 0:
            continue
        _epochs = mne.Epochs(_raw, _events_arr, event_id=1,
                             tmin=-0.2, tmax=1.0,
                             baseline=None, preload=True, verbose=False)
        _epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)
        _total += len(_epochs)
    _subj_epoch_counts[_sid] = _total

_enough_sids = {sid for sid, n in _subj_epoch_counts.items() if n >= _MIN_EPOCHS}
subj_id_array = [int(s) for s in sorted(_fnirs_subjects & _enough_sids)]

# check if any of subject in subj_id_array is in the excluded_subj
subj_id_array = [x for x in subj_id_array if f'sub-{x}' not in excluded_subj]

#%% select model type
# eeg_reg_type = 'cont_EEG_allBandPower-bandpass'
eeg_reg_type = 'cont_EEG_cz_3-stage'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save DM and GLM results
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
is_plot = False # If True, generate visualization plots
select_chromo='HbO'
# select_parcel='DorsAttnA_ParOcc_1_RH'
select_parcel='DefaultA_PFCd_1_LH'
USE_GSR=True
DO_3STAGE_REGRESSION = True # If True: (1) OLS-regress out per-run drift, (2) OLS-regress out GSR
                              # (computed from the drift-residualized signal), (3) AR-IRLS-fit only the
                              # EEG regressors on the twice-residualized signal. Overrides is_GSR_then_Others
                              # and skips adding drift/GSR back into dm_all later, since they're already removed.
cfg_GLM['do_GSR']=USE_GSR
len_delay = 15 # Delay time in HRF (sec)
bspline_degree = 3
n_bspline_basis = len_delay # low-rank df for the B-spline basis spanning the delay axis (< n_regressor)
irrr_lam1 = 1.0 # iRRR nuclear-norm penalty (Y is scaled to unit std before fitting, so this is scale-free)
# AR-IRLS run (run_model_cont_EEG_fNIRS.py) whose Y_all / dm_all / B-spline basis are reused here;
# iRRR shares the same preprocessed Y and design matrix, so only the fit differs
ar_irls_reg_type = 'cont_EEG_cz_3-stage_bspline-test'

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
    betas_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}_betas.pkl')
    stats_save_path = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_iRRR_{hp_flag}_stats.pkl')
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% load Y_all, dm_all and the B-spline basis saved by the AR-IRLS run
    ar_irls_prefix = os.path.join(data_save_path, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    Y_all_load_path = f'{ar_irls_prefix}_Y_all.pkl.gz'
    dm_all_load_path = f'{ar_irls_prefix}_dm_all.pkl.gz'
    ar_irls_betas_path = f'{ar_irls_prefix}_betas.pkl'
    if not all(os.path.exists(f) for f in [Y_all_load_path, dm_all_load_path, ar_irls_betas_path]):
        print(f"{subject}: AR-IRLS Y_all/dm_all/betas not found ({ar_irls_prefix}_*), skipping.")
        continue

    with gzip.open(Y_all_load_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(dm_all_load_path, 'rb') as f:
        dm_all = pickle.load(f)
    with open(ar_irls_betas_path, 'rb') as f:
        basis_da = pickle.load(f)['basis_da']
    n_regressor = len(basis_da.regressor)

    #%% get GLM fitting results for each subject from shank Jun 02 2025
    print(f"Start cont_EEG GLM fitting ({subject})")
    # iRRR: fit all parcels jointly, Y (time x parcel) ~ X (time x bspline), with a
    # nuclear-norm penalty on the (bspline x parcel) coefficient matrix so the HRFs
    # across parcels share a low-rank structure
    Y_np = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel').values
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    X_np = X_da.values
    Y_scale = np.nanstd(Y_np)
    n_time, n_parcel = Y_np.shape
    X_c = X_np - X_np.mean(0, keepdims=True)
    irrr_weight = [np.max(svdvals(X_c)) * (np.sqrt(n_parcel) + np.sqrt(matrix_rank(X_c))) / n_time]
    C, mu, _, _, _, irrr_details = irrr_normal(Y_np / Y_scale, [X_np], irrr_lam1,
                                               {'varyrho': True, 'Tol': 0.01, 'fig': False,
                                                'weight': irrr_weight},
                                               return_details=True)
    C = C * Y_scale  # back to original Y units
    mu = mu * Y_scale
    betas_all = xr.DataArray(
        C.T[:, None, :],
        dims=('parcel', 'chromo', 'regressor'),
        coords={'parcel': Y_all.parcel.values, 'chromo': [select_chromo],
                'regressor': X_da.regressor.values},
    )
    stats_dict = {'irrr_lam1': irrr_lam1, 'irrr_weight': irrr_weight, 'Y_scale': Y_scale,
                  'intercept': mu, 'rank': matrix_rank(C), 'singular_values': svdvals(C),
                  'details': irrr_details}
    print(f"iRRR fit: rank(B) = {stats_dict['rank']} (of {min(C.shape)})")
    # extract HRF (delay-regressor betas) per parcel, then expand the low-rank
    # bspline coefficients back to full per-delay resolution via the same basis
    eeg_reg = [p for p in betas_all.regressor.values if 'bspline' in p]
    betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
    betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])

    stats_dict['Y_all_path'] = Y_all_load_path
    stats_dict['dm_all_path'] = dm_all_load_path

    #%% save betas (Y_all / dm_all are not re-saved; stats_dict points to the AR-IRLS copies)
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg'] = betas_eeg
        betas_dict['betas_bspline'] = betas_bspline
        betas_dict['basis_da'] = basis_da
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)

        with open(stats_save_path, 'wb') as f:
            pickle.dump(stats_dict, f)
