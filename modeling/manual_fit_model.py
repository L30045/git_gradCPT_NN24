#%% load library
import numpy as np
import pickle
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
from utils import *
import model
from params_setting import *
import cedalion.xrutils as xrutils
import scipy


#%%
subj_id = 723
print(f"Start processing sub-{subj_id}")
# load HbO
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']
cfg_GLM['geo3d'] = geo3d

# get epoched concentration
run_dict = dict()
# Find all event files in project_path
event_files = glob.glob(os.path.join(project_path, f"sub-{subj_id}", 'nirs', f"sub-{subj_id}_task-gradCPT_run-*_events.tsv"))
event_files = sorted(event_files)  # Sort to ensure consistent ordering

# Load each event file into run_dict
for event_file in event_files:
    # Extract run number from filename (e.g., run-01 -> 1)
    run_num = event_file.split('run-')[1].split('_')[0]
    run_key = f'run{run_num}'

    # Initialize run dict if not exists
    if run_key not in run_dict:
        run_dict[run_key] = dict()

    # Load event dataframe
    run_dict[run_key]['ev_df'] = pd.read_csv(event_file, sep='\t')

# find corresponding runs in all_runs and assign to run_dict
for r_i, run in enumerate(all_runs):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        ev_df = run_dict[run_key]['ev_df']
        if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
            run_dict[run_key]['run'] = run[0]
            run_dict[run_key]['conc_ts'] = run['conc_o']
            run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
            break

# epoch HbO
len_epoch = 12 # seconds
t_conc_ts = run['conc_o'].time
sfreq_conc = 1/np.diff(t_conc_ts)[0]
len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

# get epoched EEG
# load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)


# get mnt_correct trials 
mnt_correct_idx_dict = model.get_valid_event_idx('mnt_correct',single_subj_epoch_dict)
mnt_correct_area_dict = model.get_ERP_area('mnt_correct', single_subj_epoch_dict)

# get mnt_incorrect trials
mnt_incorrect_idx_dict = model.get_valid_event_idx('mnt_incorrect_response',single_subj_epoch_dict)
mnt_incorrect_area_dict = model.get_ERP_area('mnt_incorrect_response', single_subj_epoch_dict)

# combine mnt_correct_idx_dict, mnt_correct_area_dict, mnt_incorrect_idx_dict, mnt_incorrect_area_dict into a dict
ev_dict = dict()
for run_key in mnt_correct_idx_dict.keys():
    ev_dict[run_key] = {
        'mnt_correct': {
            'idx': mnt_correct_idx_dict[run_key],
            'area': mnt_correct_area_dict[run_key]
        },
        'mnt_incorrect': {
            'idx': mnt_incorrect_idx_dict[run_key],
            'area': mnt_incorrect_area_dict[run_key]
        }
    }

# Get reduced model DM
select_ch = 'S10D127'
run_list = []
pruned_chans_list = []
stim_list = []
for run_key in run_dict.keys():
    run_list.append(run_dict[run_key]['run'])
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
reduced_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)

# get drift and ss
basis_dm = model.create_no_info_dm(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
dm_all = basis_dm
model.vis_dm(dm_all)

# get HbO and select channel
Y_all = Y_all.sel(chromo=['HbO'],channel=[select_ch])
dm_all.common = dm_all.common.sel(chromo=['HbO'])

#%% Define my AR-IRLS
def my_ar_irls_GLM(y, x, pmax=30, M=sm.robust.norms.HuberT()):
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

        # Apply the AR filter to the lhs and rhs of the model
        a = y[0]
        yf = pd.Series(scipy.signal.lfilter(wf, 1, y - a)) + a
        xf = np.zeros(x.shape)
        xx = x.to_numpy()
        for i in range(xx.shape[1]):
            b = xx[0, i]
            xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i] - b) + b

        xf = pd.DataFrame(xf)
        xf.columns = x.columns

        rlm_model = sm.RLM(yf, xf, M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

    return params, wf

def fit(
    ts: cdt.NDTimeSeries,
    design_matrix: DesignMatrix,
    noise_model: str = "ols",
    ar_order: int = 30,
    max_jobs: int = -1,
    verbose: bool = False,
):
    # FIXME: unit handling?
    # shoud the design matrix be dimensionless? -> thetas will have units
    ts = ts.pint.dequantify()

    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


    reg_results = xr.DataArray(
        np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
        dims=("channel", dim3_name),
        coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
    )

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

        if(max_jobs==1):
            for chan in tqdm(group_y.channel.values, disable=not verbose):
                result = my_ar_irls_GLM(group_y.loc[:, chan], x, noise_model, ar_order)
                reg_results.loc[chan, dim3] = result
        else:
            args_list=[]
            for chan in group_y.channel.values:
                args_list.append([group_y.loc[:, chan], x, noise_model, ar_order])

            with parallel_config(backend='threading', n_jobs=max_jobs):
                batch_results = tqdm(
                    Parallel(return_as="generator")(
                        delayed(my_ar_irls_GLM)(*args) for args in args_list
                    ),
                    total=len(args_list)
                )

            for chan, result in zip(group_y.channel.values, batch_results):
                reg_results.loc[chan, dim3] = result

    description='AR_IRLS' # FIXME
    reg_results.attrs["description"] = description

    return reg_results

#%% get GLM fitting results for each subject from shank Jun 02 2025
# 3. get betas and covariance
result_dict = dict()
print(f"Start EEG-informed GLM fitting (sub-{subj_id})")
glm_results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'], max_jobs=1)
result_dict['resid'] = glm_results.sm.resid
betas = glm_results.sm.params
cov_params = glm_results.sm.cov_params()
result_dict['betas']=betas
result_dict['cov_params']=cov_params

#%% from solve.fit

# Timing for rlm.fit() calls
nb_test = 4 # default 4
Y_all = Y_all.pint.dequantify()

dim3_name = xrutils.other_dim(dm_all.common, "time", "regressor")

reg_results = xr.DataArray(
    np.empty((Y_all.sizes["channel"], Y_all.sizes[dim3_name]), dtype=object),
    dims=("channel", dim3_name),
    coords=xrutils.coords_from_other(Y_all.isel(time=0), dims=("channel", dim3_name))
)

for (
    dim3,
    group_channels,
    group_design_matrix,
) in dm_all.iter_computational_groups(Y_all):
    group_y = Y_all.sel({"channel": group_channels, dim3_name: dim3}).transpose(
        "time", "channel"
    )

    # pass x as a DataFrame to statsmodel to make it aware of regressor names
    x = pd.DataFrame(
        group_design_matrix.values, columns=group_design_matrix.regressor.values
    )
    y = group_y.loc[:,select_ch]
    M=sm.robust.norms.HuberT()
    mask = np.isfinite(y.values)

    yorg : pd.Series = pd.Series(y.values[mask].copy())
    xorg : pd.DataFrame = x[mask].reset_index(drop=True)

    y = yorg.copy()
    x = xorg.copy()

    rlm_model = sm.RLM(y, x, M=M)
    params = rlm_model.fit()

    resid = pd.Series(y - x @ params.params)
    for iter_i in range(nb_test):  # TODO - check convergence
        y = yorg.copy()
        x = xorg.copy()

        # Update the AR whitening filter
        arcoef = cedalion.math.ar_model.bic_arfit(resid, pmax=30)
        wf = np.hstack([1, -arcoef.params[1:]])

        # Apply the AR filter to the lhs and rhs of the model
        a = y[0]
        yf = pd.Series(scipy.signal.lfilter(wf, 1, y - a)) + a
        xf = np.zeros(x.shape)
        xx = x.to_numpy()
        for i in range(xx.shape[1]):
            b = xx[0, i]
            xf[:, i] = scipy.signal.lfilter(wf, 1, xx[:, i] - b) + b

        xf = pd.DataFrame(xf)
        xf.columns = x.columns

        rlm_model = sm.RLM(yf, xf, M=M)
        params = rlm_model.fit()

        resid = pd.Series(yorg - xorg @ params.params)

#%% Residuals
my_x = np.squeeze(dm_all.common.values)
my_y = np.squeeze(Y_all.values)
whiten_fit = (xf@params.params.values).values
whiten_y = yf
my_fit = my_x@params.params.values
my_resid = my_y-my_fit
my_y_fitted_val = whiten_y-params.fittedvalues
print(f"Y_Diff={np.sum(my_y-yorg.values)}")
print(f"X_Diff={np.sum(my_x-xorg.values)}")
print(f"Betas Diff = {np.sum(betas.values-params.params.values)}")
print(f"Resid Diff in original space = {np.sum(resid.values-my_resid)}")
print(f"Resid Diff in whitten space = {np.sum(params.resid-my_y_fitted_val)}")

fig, ax = plt.subplots(3,1)
ax[0].plot(xorg@params.params,label='Xorg @ beta')
ax[0].plot(yorg, label='Y')
ax[0].legend()
ax[1].plot(whiten_fit,label='whitten Fit')
ax[1].plot(whiten_y, label='whitten y')
ax[1].plot(params.fittedvalues,label='fittedvalues')
ax[1].legend()
ax[2].plot(resid,label='resid in original space')
ax[2].plot(params.resid.values, label='resid in whitten space')
ax[2].legend()
