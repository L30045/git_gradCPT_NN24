#%% load library
import numpy as np
import pickle
import gzip
import glob
import sys
import os
import re
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from joblib import Parallel, delayed
from tqdm import tqdm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import model
from params_setting import *

#%% find subjects with a shared Y_all (only saved for subjects with fNIRS and enough EEG epochs)
_eeg_deriv = os.path.join(project_path, 'derivatives', 'eeg')

#%% select model type
eeg_reg_type = 'cont_EEG_cz_3-stage'
is_overwrite = True # If True, force re-training GLM.
is_save = True # If True, save betas and stats
is_hp_fNIRS = True # If True, highpass fNIRS by 0.02 Hz
select_chromo = 'HbO'
max_jobs = -1 # parallel jobs across parcels
irls_norm = sm.robust.norms.TukeyBiweight(c=4.685) # same robust norm as model.my_ar_irls_GLM
# AR-IRLS run (run_model_cont_EEG_fNIRS.py) whose dm_all / B-spline basis are reused here;
# Y_all is the shared parcel_Y_all_truncated_to_trials file, so only the fit differs
ar_irls_reg_type = 'cont_EEG_cz_3-stage_bspline-test'

hp_flag = 'Hp' if is_hp_fNIRS else 'noHp'
subj_id_array = sorted(
    int(re.search(r'sub-(\d+)', f).group(1))
    for f in glob.glob(get_shared_Y_all_path(os.path.join(_eeg_deriv, 'sub-*'), 'sub-*', hp_flag))
)
subj_id_array = [x for x in subj_id_array if f'sub-{x}' not in excluded_subj]

#%% main
for subj_id in subj_id_array:
    subject = f"sub-{subj_id}"
    print(f"Start processing {subject}")
    data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)

    # check if betas.pkl exist already. If yes, skip this subject.
    save_prefix = os.path.join(data_save_path, f'{subject}_{eeg_reg_type}_fixAR-IRLS_{hp_flag}')
    betas_save_path = get_betas_path(save_prefix)
    stats_save_path = get_stats_path(save_prefix)
    if not is_overwrite and os.path.exists(betas_save_path):
        print(f"{subject}: betas already exist, skipping.")
        continue

    #%% load Y_all, dm_all and the B-spline basis saved by the AR-IRLS run
    ar_irls_prefix = os.path.join(data_save_path, f'{subject}_{ar_irls_reg_type}_{NOISE_MODEL}_{hp_flag}')
    Y_all_load_path = get_shared_Y_all_path(data_save_path, subject, hp_flag)
    dm_all_load_path = get_dm_all_path(ar_irls_prefix)
    ar_irls_betas_path = get_betas_path(ar_irls_prefix)
    if not all(os.path.exists(f) for f in [Y_all_load_path, dm_all_load_path, ar_irls_betas_path]):
        print(f"{subject}: Y_all or AR-IRLS dm_all/betas not found ({ar_irls_prefix}_*), skipping.")
        continue

    with gzip.open(Y_all_load_path, 'rb') as f:
        Y_all = pickle.load(f)
    with gzip.open(dm_all_load_path, 'rb') as f:
        dm_all = pickle.load(f)
    with open(ar_irls_betas_path, 'rb') as f:
        ar_irls_betas_dict = pickle.load(f)
    basis_da = ar_irls_betas_dict['basis_da']
    n_regressor = len(basis_da.regressor)

    #%% spectral whitening of Y and X
    # the residual spectrum is estimated from the OLS residuals of Y_partial ~ 1 + X (Y_all is already
    # drift- and GSR-residualized), averaged across parcels, and its 1/sqrt(power) filter is applied
    # to both Y and X. Runs are padded and filtered separately.
    Y_da = Y_all.sel(chromo=select_chromo).pint.dequantify().transpose('time', 'parcel')
    X_da = dm_all.common.sel(chromo=select_chromo).transpose('time', 'regressor')
    X_ols = np.column_stack([np.ones(len(X_da.time)), X_da.values])
    B_ols = np.linalg.lstsq(X_ols, Y_da.values, rcond=None)[0]
    samples = Y_da.samples.values  # per-run sample index; resets mark run starts in Y_all
    run_bounds = np.where(np.diff(samples) <= 0)[0] + 1
    noise_model = model.NoiseModel(method='spectral')
    noise_model.fit(model.split_runs(Y_da.values - X_ols @ B_ols, samples))
    Y_white = np.vstack(noise_model.whiten(model.split_runs(Y_da.values, samples)))
    X_white = np.vstack(noise_model.whiten(model.split_runs(X_da.values, samples)))

    print(f"spectral whitening: {len(run_bounds) + 1} runs, w_pad = {noise_model.w_pad}")

    #%% IRLS fit per parcel on the whitened data (no further AR whitening)
    print(f"Start cont_EEG GLM fitting ({subject})")
    # the whitened series are zero-mean (DC bin of the kernel is 0), so no intercept is added
    parcels = Y_da.parcel.values
    x_df = pd.DataFrame(X_white, columns=X_da.regressor.values)
    rlm_results = Parallel(n_jobs=max_jobs, backend='threading')(
        delayed(model.irls_fit)(pd.Series(Y_white[:, p_i]), x_df, M=irls_norm)
        for p_i in tqdm(range(len(parcels)))
    )

    betas_all = xr.DataArray(
        np.stack([res.params.values for res in rlm_results])[:, None, :],
        dims=('parcel', 'chromo', 'regressor'),
        coords={'parcel': parcels, 'chromo': [select_chromo],
                'regressor': X_da.regressor.values},
    )
    # extract HRF (delay-regressor betas) per parcel, then expand the low-rank
    # bspline coefficients back to full per-delay resolution via the same basis
    eeg_reg = [p for p in betas_all.regressor.values if 'bspline' in p]
    betas_bspline = betas_all.sel(regressor=eeg_reg).rename({"regressor": "component"})
    betas_eeg = xr.dot(betas_bspline, basis_da, dims="component")
    betas_eeg = betas_eeg.assign_coords(regressor=[f"delay{d_i}" for d_i in range(n_regressor)])

    #%% f test / contrast t test per parcel
    stats_dict = {'run_bounds': run_bounds,
                  'whiten_kernel_fft': noise_model.W_fft, 'acf_kernel': noise_model.acf_kernel,
                  'Y_all_path': Y_all_load_path, 'dm_all_path': dm_all_load_path}
    # test if EEG can explain more variance
    f_hypotheses = [f'{name} = 0' for name in eeg_reg]
    stats_dict['f_test_full_noEEG'] = {p: res.f_test(f_hypotheses) for p, res in zip(parcels, rlm_results)}
    # test if EEG betas sums to 0
    t_hypotheses = '+'.join(eeg_reg) + ' = 0'
    stats_dict['t_test_0_eeg'] = {p: res.t_test(t_hypotheses) for p, res in zip(parcels, rlm_results)}

    #%% save betas (Y_all / dm_all are not re-saved; stats_dict points to the loaded copies)
    if is_save:
        betas_dict = dict()
        betas_dict['betas'] = betas_all
        betas_dict['betas_eeg'] = betas_eeg
        betas_dict['betas_bspline'] = betas_bspline
        betas_dict['basis_da'] = basis_da
        os.makedirs(os.path.dirname(betas_save_path), exist_ok=True)
        with open(betas_save_path, 'wb') as f:
            pickle.dump(betas_dict, f)

        os.makedirs(os.path.dirname(stats_save_path), exist_ok=True)
        with open(stats_save_path, 'wb') as f:
            pickle.dump(stats_dict, f)
