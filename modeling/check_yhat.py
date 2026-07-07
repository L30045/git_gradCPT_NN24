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
import statsmodels.api as sm
import scipy
import cedalion
import statsmodels.api as sm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
import utils
import model
from params_setting import *
from tqdm import tqdm
import re
import xarray as xr
import cedalion.xrutils as xrutils

#%% settings
subj_id = 695
debug_channel = 'S10D127'
debug_chromo = 'HbO'
save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")

model_types = ['full_noEEG_rejected_ttest', 'reduced', 'onlyEEG']
model_to_dmkey = {
    'full_noEEG_rejected_ttest': 'full',
    'reduced': 'onlyStim',
    'onlyEEG': 'onlyEEG',
}

#%% load design matrices and dependent variable (Y_all, as in model_EEG_inform_single_subject.py)
with open(os.path.join(save_file_path, 'dm_dict.pkl'), 'rb') as f:
    dm_dict = pickle.load(f)

Y_all = dm_dict['Y_all']  # dims: chromo, channel, time
y_true = Y_all.sel(channel=[debug_channel])

#%% retrain stim and EEG only
ar_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_full_noEEG_rejected_ttest_20260706.pkl')
with open(ar_path, 'rb') as f:
    old_full_result = pickle.load(f)
    autoReg_dict = old_full_result['autoReg_dict']

beta_dict = dict()

#%% def original ar_irls.py in cedalion
def ar_irls_GLM(y, x, pmax=40, M=sm.robust.norms.TukeyBiweight(c=4.685)):
    """This function implements the AR-IRLS GLM model.

    The autoregressive iteratively reweighted least squares GLM model is described in
    :cite:t:`Barker2013`. By estimating prewhitening filters it addresses serial
    correlations and confounding noise components in the signal and avoids the inflated
    false positive rates observed when fitting the GLM with ordinary least squares.

    Inputs:
        y - pandas Serial
        x - pandas DataFrame
        pmax- max AR model order (default 40)
        M- statsmodel.robust.norms type (default Huber)

    Outputs:
        stats- statsmodel.RLM results model

    Initial Contributors:
        Ted Huppert | huppert1@pitt.edu | 2024

      d is matrix containing the data; each column is a channel of data

      X is the regression/design matrix

      Pmax is the maximum AR model order that you want to consider. A
      purely speculative guess is that the average model order is
      approximatley equal to the 2-3 times the sampling rate, so setting Pmax
      to 4 or 5 times the sampling rate should work fine.  The code does not
      suffer a hugeperformance hit by using a higher Pmax; however, the number
      of time points used to estimate the AR model will be
      "# of time points - Pmax", so don't set Pmax too high.

      "tune" is the tuning constant used for Tukey's bisquare function during
      iterative reweighted least squares. The default value is 4.685.
      Decreasing "tune" will make the regression less sensitive to outliers,
      but at the expense of performance (statistical efficiency) when data
      does not have outliers. For reference, the values of tune for 85, 90,
      and 95  statistical efficiency are

      tune = 4.685 --> 95%
      tune = 4.00  --> ~90%
      tune = 3.55  --> ~85%

      I have not tested these to find an optimal value for the "average" NIRS
      dataset; however, 4.685 was used in the published simulations and worked
      quite well even with a high degree of motion artifacts from children.
      If you really want to adjust it, you could use the above values as a
      guideline.

      DO NOT preprocess your data with a low pass filter.
      The algorithm is trying to transform the residual to create a
      white spectrum.  If part of the spectrum is missing due to low pass
      filtering, the AR coefficients will be unstable.  High pass filtering
      may be ok, but I suggest putting orthogonal polynomials (e.g. Legendre)
      or low frequency discrete cosine terms directly into the design matrix
      (e.g. from spm_dctmtx.m from SPM).  Don't use regular polynomials
      (e.g. 1 t t^2 t^3 t^4 ...) as this can result in a poorly conditioned
      design matrix.

      If you choose to resample your data to a lower sampling frequency,
      makes sure to choose an appropriate cutoff frequency so that that the
      resulting time series is not missing part of the frequency spectrum
      (up to the Nyquist bandwidth).  The code should work fine on 10-30 Hz
      data.
    """

    mask = np.isfinite(y.values)

    yorg : pd.Series = pd.Series(y.values[mask].copy())
    xorg : pd.DataFrame = x[mask].reset_index(drop=True)

    y = yorg.copy()
    x = xorg.copy()

    rlm_model = sm.RLM(y, x, M=M)
    params = rlm_model.fit()

    resid = pd.Series(y - x @ params.params)
    for _ in range(4):  # TODO - check convergence
        y = yorg.copy()
        x = xorg.copy()

        # Update the AR whitening filter
        arcoef = cedalion.math.ar_model.bic_arfit(resid, pmax=pmax)
        wf = np.hstack([1, -arcoef.params[1:]])
        p = len(wf) - 1

        # Apply the AR filter to the lhs and rhs of the model
        yf = pd.Series(scipy.signal.lfilter(wf, 1, y))

        xf = np.zeros(x.shape)
        xx = x.to_numpy()
        for i in range(xx.shape[1]):
            xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i])

        xf = pd.DataFrame(xf)
        xf.columns = x.columns

        # fit the model ignoring the first p samples, for which the AR filter is not
        # yet fully initialized.
        rlm_model = sm.RLM(yf[p:], xf.iloc[p:], M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params, arcoef.params

#%% open my_fit
ts = y_true
design_matrix = dm_dict[model_to_dmkey['reduced']]
autoReg = None
ar_order = 30
verbose = False

ts = ts.pint.dequantify()

dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


reg_results = xr.DataArray(
    np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
    dims=("channel", dim3_name),
    coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
)
autoReg_dict = dict()

for (
    dim3,
    group_channels,
    group_design_matrix,
) in design_matrix.iter_computational_groups(ts):
    group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
        "time", "channel"
    )

    # pass x as a DataFrame to statsmodel to make it aware of regressor names
    x = pd.DataFrame(
        group_design_matrix.values, columns=group_design_matrix.regressor.values
    )

    for chan in tqdm(group_y.channel.values, disable=not verbose):
        result = ar_irls_GLM(group_y.loc[:, chan], x, pmax=ar_order)
        reg_results.loc[chan, dim3] = result[0]
        autoReg_dict[chan] = result[1]

description='AR_IRLS' # FIXME
reg_results.attrs["description"] = description
betas = reg_results.sm.params

 # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
y_hat_ori_stim = (design_matrix.common * betas).sum('regressor')
y_hat_ori_stim = y_hat_ori_stim.transpose('chromo', 'channel', 'time')

#%% retrain stim-only model using original cedalion ar_irls_GLM instead of my_ar_irls_GLM
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)
y_pred = y_hat_ori_stim.sel(channel=debug_channel, chromo=debug_chromo).values
plt.plot(Y_all.time.values, y_pred, label=model_type, alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#%% train models
# full
full_result, full_ar = model.my_fit(y_true, dm_dict['full'], autoReg=None)
beta_dict['full_noEEG_rejected_ttest'] = full_result.sm.params
# stim only model
stim_results, stim_ar = model.my_fit(y_true, dm_dict[model_to_dmkey['reduced']], autoReg=None)
beta_dict['reduced'] = stim_results.sm.params
# EEG only model
eeg_results, eeg_ar = model.my_fit(y_true, dm_dict[model_to_dmkey['onlyEEG']], autoReg=None)
beta_dict['onlyEEG'] = eeg_results.sm.params

#%% calculate y_hat
y_hat_dict = dict()
for model_type in model_types:
    betas = beta_dict[model_type]  # dims: channel, chromo, regressor
    dm = dm_dict[model_to_dmkey[model_type]]  # .common dims: time, regressor, chromo

    # Y_hat = X @ beta, summed over regressors -> dims: time, channel, chromo
    y_hat = (dm.common * betas).sum('regressor')
    y_hat_dict[model_type] = y_hat.transpose('chromo', 'channel', 'time')

#%% compare stim-only, EEG-only, and full model y_hat for a single channel
plt.figure(figsize=(14, 4))
plt.plot(Y_all.time.values, y_true.sel(chromo='HbO').values.flatten(), label='Y (true)', color='k', linewidth=1)
for model_type in model_types:
    y_pred = y_hat_dict[model_type].sel(channel=debug_channel, chromo=debug_chromo).values
    plt.plot(Y_all.time.values, y_pred, label=model_type, alpha=0.7)
plt.xlabel('time (s)')
plt.ylabel(f'{debug_chromo} concentration')
plt.title(f'sub-{subj_id} channel={debug_channel}')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
