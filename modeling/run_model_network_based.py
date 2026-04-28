#%% load library
import numpy as np
import pickle
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

#%% Test subject
subj_id = 721
filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/pipeline_reorder/processed_data/sub-{subj_id}"
filename = f"sub-{subj_id}_task-gradCPT_adot-probe_spatialdim-vertex_IR_ts.pkl"

# load HbO
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']
cfg_GLM['geo3d'] = geo3d

# load parcel
with open(os.path.join(filepath, filename),'rb') as f:
    results = pickle.load(f)
parcel_ts_post = results['parcel_ts_post']
vertex_mse = results['parcel_mse']

# sort

#%% for each run
roi_parcel_ts_list = []
roi_var_list = []
for run_id in range(len(parcel_ts_post)):
    # select functional network (default mode network, dorsal lateral attention network)
    """
    parcel name before '_' is the network it belongs to.
    """
    # get unique network
    network_labels = np.array([x.split('_')[0] for x in parcel_ts_post[run_id]['parcel'].values])
    unique_networks = np.unique(network_labels)
    # get data
    data = parcel_ts_post[run_id].copy()
    data = data.assign_coords({'channel': ('parcel', network_labels)})
    # get parcel mse
    mse = vertex_mse[run_id].copy()
    mse = mse.groupby('parcel').mean('vertex')
    mse = mse.assign_coords({'channel': ('parcel', network_labels)})

    # Then I do our weighted averaging across the network dimension:
    roi_parcel_ts = (data/mse).groupby('channel').sum('parcel') / (1/mse).groupby('channel').sum('parcel')
    roi_var =  1 / (1/mse).groupby('channel').sum('parcel')

    # store the values
    roi_parcel_ts_list.append(roi_parcel_ts)
    roi_var_list.append(roi_var)

#%%
model_type='full_noEEG_rejected_ttest'

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

#%% get epoched concentration
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
sorted_run_time = []
for r_i, run in enumerate(all_runs):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        ev_df = run_dict[run_key]['ev_df']
        if len(ev_df) > 0 and len(run.stim) > 0 and np.all(run.stim.iloc[0] == ev_df.iloc[0]):
            sorted_run_time.append(run[0].time.values)
            break

# find corresponding roi_parcel_ts in roi_parcel_ts_list and assign to run_dict
for r_i, roi in enumerate(roi_parcel_ts_list):
    # Match this run to the correct run_dict entry by comparing first event
    for run_key in run_dict.keys():
        run_id = int(run_key.split('run')[-1])-1
        if len(sorted_run_time[run_id]) == len(roi.time.values) and np.all(roi.time.values==sorted_run_time[run_id]):
            run_dict[run_key]['network'] = roi
            run_dict[run_key]['network_mse'] = roi_var_list[r_i]
            break

# epoch HbO
len_epoch = 12 # seconds
t_conc_ts = run_dict['run01']['network'].time
sfreq_conc = np.median(1/np.diff(t_conc_ts))
len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

#%% get epoched EEG
# load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = utils.eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)


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

#%% Get reduced model DM
run_list = []
stim_list = []
for run_key in run_dict.keys():
    run_list.append(run_dict[run_key]['network'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
    ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
    stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
reduced_dm = model.get_GLM_copy_from_pf_DM_network(run_list, cfg_GLM, stim_list)
# Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)
# Y_all = Y_all.assign_coords(samples=('time', np.arange(Y_all.sizes['time'])))

# get drift and ss
basis_dm = model.create_no_info_dm_network(run_list, cfg_GLM, stim_list)

# Get EEG DM
eeg_dm_dict = model.create_eeg_dm_network(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

# combine EEG DMs from all runs into one big DM
Y_all, eeg_dm, runs_updated = model.concatenate_runs_dms_network(run_dict, eeg_dm_dict)

# save DMs
save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
save_dm_name = os.path.join(save_file_path, 'dm_dict_network.pkl')
if not os.path.exists(save_dm_name):
    dm_dict = dict()
    dm_dict['basis']=basis_dm
    dm_dict['onlyEEG']=model.combine_dm(eeg_dm, basis_dm)
    dm_dict['onlyStim']=reduced_dm
    dm_dict['full']=model.combine_dm(eeg_dm, reduced_dm)
    dm_dict['Y_all']=Y_all
    with open(save_dm_name,'wb') as f:
        pickle.dump(dm_dict,f)

#%% assign DM
if model_type.startswith('full'):
    # Combine EEG DM with Reduced DM to get full model
    dm_all = model.combine_dm(eeg_dm, reduced_dm)
elif model_type=='reduced':
    dm_all = reduced_dm
elif model_type=='onlyEEG':
    dm_all = model.combine_dm(eeg_dm, basis_dm)
else:
    dm_all = basis_dm

#%% get GLM fitting results for each subject from shank Jun 02 2025
print(f"Start EEG-informed GLM fitting (sub-{subj_id})")
if model_type.startswith('full'):
    glm_results, autoReg_dict = model.my_fit(Y_all, dm_all)
else:
    file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    with open(os.path.join(file_path,f'sub-{subj_id}_glm_mnt_full_noEEG_rejected.pkl'),'rb') as f:
        full_result = pickle.load(f)
        autoReg_dict = full_result['autoReg_dict']
    glm_results, autoReg_dict = model.my_fit(Y_all, dm_all, autoReg=autoReg_dict)

# 3. get betas and covariance
result_dict = dict()
result_dict['resid'] = glm_results.sm.resid
betas = glm_results.sm.params
cov_params = glm_results.sm.cov_params()
result_dict['betas']=betas
result_dict['cov_params']=cov_params
result_dict['autoReg_dict']=autoReg_dict

#%% f test
if model_type.startswith('full'):
    # full vs stim
    param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
    # Create hypothesis strings
    hypotheses = [f'{name} = 0' for name in param_names]
    # Run F-test
    f_test_result = glm_results.sm.f_test(hypotheses)
    result_dict['f_test_full_stim'] = f_test_result
    # full vs basis
    param_names = [name for name in glm_results.sm.params.regressor.values if ('eeg' in name) or ('stim' in name)]
    # Create hypothesis strings
    hypotheses = [f'{name} = 0' for name in param_names]
    # Run F-test
    f_test_result = glm_results.sm.f_test(hypotheses)
    result_dict['f_test_full_basis'] = f_test_result
    # full vs eeg
    param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
    # Create hypothesis strings
    hypotheses = [f'{name} = 0' for name in param_names]
    # Run F-test
    f_test_result = glm_results.sm.f_test(hypotheses)
    result_dict['f_test_full_eeg'] = f_test_result
elif model_type=='reduced':
    param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
    # Create hypothesis strings
    hypotheses = [f'{name} = 0' for name in param_names]
    # Run F-test
    f_test_result = glm_results.sm.f_test(hypotheses)
    result_dict['f_test_stim_basis'] = f_test_result
elif model_type=='onlyEEG':
    param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
    # Create hypothesis strings
    hypotheses = [f'{name} = 0' for name in param_names]
    # Run F-test
    f_test_result = glm_results.sm.f_test(hypotheses)
    result_dict['f_test_eeg_basis'] = f_test_result

#%% diagnose redundant regressors
# param_names_diag = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
# cov_full = glm_results.sm.cov_params().sel(
#     regressor_r=param_names_diag,
#     regressor_c=param_names_diag
# )
# print("cov_params submatrix dims:", cov_full.dims, "shape:", cov_full.shape)

# # collapse all non-regressor dims by taking first index — collinearity is in
# # regressor space so one spatial/chromo slice is sufficient for diagnosis
# extra_dims = [d for d in cov_full.dims if d not in ('regressor_r', 'regressor_c')]
# cov_sub = cov_full.isel({d: 0 for d in extra_dims}).values  # shape: (n_params, n_params)
# print("2-D slice used for SVD, shape:", cov_sub.shape)

# U, sv, Vt = np.linalg.svd(cov_sub)
# print("\nSingular values:")
# for i in range(len(sv)):
#     print(f"  sv[{i:2d}] = {sv[i]:.3e}")

# # direction corresponding to smallest singular value = redundant constraint
# redundant_vec = Vt[-1]
# print("\nRedundant direction (|loading| > 0.1):")
# for name, load in zip(param_names_diag, redundant_vec):
#     if abs(load) > 0.1:
#         print(f"  {name:50s}  loading = {load:.3f}")

# # SE from diagonal of the slice; inflated SE flags near-collinear regressor
# betas_diag = glm_results.sm.params.sel(regressor=param_names_diag).isel(
#     {d: 0 for d in extra_dims}
# )
# se_diag = np.sqrt(np.diag(cov_sub))
# print("\nBeta and SE per EEG regressor:")
# for name, beta, se in zip(param_names_diag, np.array(betas_diag).flatten(), se_diag):
#     print(f"  {name:50s}  beta={beta:+.3e}  SE={se:.3e}")

#%% contrast t test
if model_type.startswith('full'):
    # full vs stim
    param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
    # Create hypothesis strings
    hypotheses = '+'.join(param_names)+' = 0'
    # Run F-test
    t_test_result = glm_results.sm.t_test(hypotheses)
    result_dict['t_test_0_eeg'] = t_test_result
    # full vs basis
    param_names = [name for name in glm_results.sm.params.regressor.values if ('eeg' in name) or ('stim' in name)]
    # Create hypothesis strings
    hypotheses = '+'.join(param_names)+' = 0'
    # Run F-test
    t_test_result = glm_results.sm.t_test(hypotheses)
    result_dict['t_test_0_eeg_stim'] = t_test_result
    # full vs eeg
    param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
    # Create hypothesis strings
    hypotheses = '+'.join(param_names)+' = 0'
    # Run F-test
    t_test_result = glm_results.sm.t_test(hypotheses)
    result_dict['t_test_0_stim'] = t_test_result
elif model_type=='reduced':
    param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
    # Create hypothesis strings
    hypotheses = '+'.join(param_names)+' = 0'
    # Run F-test
    t_test_result = glm_results.sm.t_test(hypotheses)
    result_dict['t_test_0_stim'] = t_test_result
elif model_type=='onlyEEG':
    param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
    # Create hypothesis strings
    hypotheses = '+'.join(param_names)+' = 0'
    # Run F-test
    t_test_result = glm_results.sm.t_test(hypotheses)
    result_dict['f_test_0_eeg'] = t_test_result

#%% get HRF and MSE for each event type
if model_type!='basis':
    # 4. estimate HRF and MSE
    trial_type_list = ['mnt-correct','mnt-incorrect']

    betas = glm_results.sm.params
    cov_params = glm_results.sm.cov_params()
    run_unit = Y_all.pint.units
    # check if it is a full model
    if model_type.startswith('full'):
        # TODO: find an elegant way to check if _stim regressor is presented
        """
        NOTE: The number of regressors is fixed.
        """
        sample_run = run_dict[run_key]['network'].copy()
        sample_run = sample_run.assign_coords(samples=('time', np.arange(sample_run.sizes['time'])))
        sample_run.time.attrs['units'] = units.s
        basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(sample_run)
        basis_hrf = model.xr.concat([basis_hrf,basis_hrf],dim='component')
    else:
        basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(sample_run)


    hrf_mse_list = []
    hrf_estimate_list = []

    for trial_type in trial_type_list:
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = model.estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = model.estimate_HRF_cov(cov_hrf, basis_hrf)

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)

    hrf_estimate = model.xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(run_unit)

    hrf_mse = model.xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(run_unit**2)

    # set universal time so that all hrfs have the same time base 
    fs = model.frequency.sampling_rate(sample_run).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    result_dict['hrf_estimate'] = hrf_estimate
    result_dict['hrf_mse'] = hrf_mse

#%%
save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
with open(os.path.join(save_file_path,f'sub-{subj_id}_glm_mnt_{model_type}.pkl'),'wb') as f:
    pickle.dump(result_dict,f)