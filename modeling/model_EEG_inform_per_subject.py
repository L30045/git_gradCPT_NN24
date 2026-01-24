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
mne.viz.set_browser_backend("matplotlib")
import os
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from utils import *
import model
from model import extract_n2_p3_features
from params_setting import *

#%% load HbO
subj_id = 695
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

#%% get epoched EEG
# load eeg to match the time
single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)

# #%% get event trials
# def get_valid_event_idx(ev_name, single_subj_epoch_dict):
#     ev_idx_dict = dict()
#     for run_key in single_subj_epoch_dict.keys():
#         ev_idx_dict[run_key] = dict()
#         if len(single_subj_epoch_dict[run_key][ev_name])>0:
#             # get preserved trial index
#             ev_preserved_idx = np.where([len(log) == 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
#             # get rejected trial index
#             ev_rejected_idx = np.where([len(log) != 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
#             # add dict
#             ev_idx_dict[run_key]['preserved'] = ev_preserved_idx
#             ev_idx_dict[run_key]['rejected'] = ev_rejected_idx
#         else:
#             ev_idx_dict[run_key]['preserved'] = []
#             ev_idx_dict[run_key]['rejected'] = []
#     return ev_idx_dict

# #%% get ERP area
# def get_ERP_area(ev_name, single_subj_epoch_dict, is_norm=True):
#     # define output dict
#     erp_area_dict = dict()
#     # define ERP period of interest
#     if ev_name.endswith('response'):
#         n2_window=(0,0.2)
#         p3_window=(0,0.2)
#     else:
#         n2_window=(0.4, 0.7)
#         p3_window=(0.6, 1.1)
#     # for each run
#     for run_key in single_subj_epoch_dict.keys():
#         erp_area_dict[run_key]=dict()
#         if len(single_subj_epoch_dict[run_key][ev_name])>0:
#             t_vector = single_subj_epoch_dict[run_key][ev_name].times
#             # for each channel, extract ERP area
#             ev_eeg = single_subj_epoch_dict[run_key][ev_name].pick(picks='eeg')
#             for ch_name in ev_eeg.ch_names:
#                 ev_ch_eeg = ev_eeg.get_data()[:,ev_eeg.ch_names.index(ch_name),:]            
#                 # Extract N2 and P3 features
#                 area_list = []
#                 for eeg_i in range(len(ev_ch_eeg)):
#                     n2_p3_features = extract_n2_p3_features(ev_ch_eeg[eeg_i], t_vector,
#                                                             n2_window=n2_window,
#                                                             p3_window=p3_window)
#                     n2_area = np.abs(n2_p3_features['n2_area'])
#                     p3_area = np.abs(n2_p3_features['p3_area'])
#                     if ev_name.endswith('respons'):
#                         area_list.append(n2_area)
#                     else:
#                         area_list.append(n2_area+p3_area)
#                 # rescale area to range 0 to 1. (0 as 0, 1 as max(area))
#                 area_list = np.array(area_list)
#                 if is_norm:
#                     area_list = area_list/np.max(area_list)
#                 # store results
#                 erp_area_dict[run_key][ch_name] = area_list
#         else:
#             erp_area_dict[run_key] = []
#     return erp_area_dict

#%% get mnt_correct trials 
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

#%% add events to design matrix
# # add events per run.
# def add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=None, select_chs=['cz']):
#     """
#     select_chs: select channels to add to design matrix
#     """
#     dm_dict = dict()
#     for run_key in run_dict.keys():
#         dm_dict[run_key] = dict()
#         target_run = run_dict[run_key]['run']
#         conc_o = run_dict[run_key]['conc_ts']
#         chs_pruned = run_dict[run_key]['chs_pruned']
#         ev_df = run_dict[run_key]['ev_df']
#         # for each run, get drift and short-separation regressors (if any)
#         if cfg_GLM['do_drift']:
#             drift_regressors = model.get_drift_regressors([conc_o], cfg_GLM)
#         elif cfg_GLM['do_drift_legendre']:
#             drift_regressors = model.get_drift_legendre_regressors([conc_o], cfg_GLM)
#         else:
#             drift_regressors = None
#         if cfg_GLM['do_short_sep']:
#             ss_regressors = model.get_short_regressors([conc_o], [chs_pruned], cfg_GLM['geo3d'], cfg_GLM)
#         else:
#             ss_regressors = None
#         # for each event, create a dm list
#         if not select_event:
#             select_event = ev_dict[run_key].keys()
#         for ev_name in select_event:
#             match ev_name:
#                 case 'mnt_correct':
#                     target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
#                     # rename trial_type
#                     target_ev_df.loc[:,'trial_type'] = 'mnt-correct'
#                 case 'mnt_incorrect':
#                     target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
#                     # rename trial_type
#                     target_ev_df.loc[:,'trial_type'] = 'mnt-incorrect'
#             # check if event exist
#             if len(target_ev_df)==0:
#                 # store in dm_dict
#                 dm_dict[run_key][ev_name] = []
#                 continue
#             # create design matrix
#             dm_list = []
#             for ev_i, event_id in enumerate(ev_dict[run_key][ev_name]['idx']['preserved']):
#                 dm = glm.design_matrix.hrf_regressors(
#                                             target_run,
#                                             target_ev_df.iloc[[event_id]],
#                                             glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
#                                         )
#                 # rescale by Cz area
#                 #TODO: allow multiple channels in DM
#                 dm.common = dm.common*ev_dict[run_key][ev_name]['area']['cz'][ev_i]
#                 # append
#                 dm_list.append(dm)
#             # Create a new design matrix object with the concatenated common regressors
#             dms = dm_list.pop()
#             dms_common = dms.common
#             # merge all dms along time axis
#             while len(dm_list)>0:
#                 dms_common += dm_list.pop().common
#             # assign merged common back to dms
#             dms.common = dms_common
#             # add drift and short-separation regressors if any
#             if drift_regressors:
#                 dms &= reduce(operator.and_, drift_regressors)
#                 dms.common = dms.common.fillna(0)
#             if ss_regressors:
#                 dms &= reduce(operator.and_, ss_regressors)
#                 dms.common = dms.common.fillna(0)
#             # store in dm_dict
#             dm_dict[run_key][ev_name] = dms
     
#     return dm_dict

#%% Get DM
dm_dict = model.add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

#%% combine DMs from all runs into one big DM
Y_all, dm_all, runs_updated = model.concatenate_runs_dms(run_dict, dm_dict)

#%% check dm
# plt_dm = dm_dict['run01']
def vis_dm(plt_dm):
    # using xr.DataArray.plot
    f, ax = plt.subplots(1,1,figsize=(12,10))
    # plt_dm.common.sel(chromo="HbO", time=plt_dm.common.time<600).T.plot(vmin=-2,vmax=2)
    plt_dm.common.sel(chromo="HbO").T.plot(vmin=-2,vmax=2)
    plt.title("Shared Regressors")
    #p.xticks(rotation=90)
    plt.show()

#%% get GLM fitting results for each subject from shank Jun 02 2025
# 3. get betas and covariance
result_dict = dict()
glm_results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])
result_dict['resid'] = glm_results.sm.resid

#%% get HRF and MSE for each run
# 4. estimate HRF and MSE
trial_type_list = ['mnt_correct','mnt_incorrect']

betas = glm_results.sm.params
cov_params = glm_results.sm.cov_params()
run_unit = Y_all.pint.units
basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])

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

hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
hrf_estimate = hrf_estimate.pint.quantify(run_unit)

hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
hrf_mse = hrf_mse.pint.quantify(run_unit**2)

# set universal time so that all hrfs have the same time base 
fs = model.frequency.sampling_rate(run_dict[run_key]['run']).to('Hz')
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

#%% GLM model from pf.GLM()
# Unknown pf.GLM() loaded. Required only 4 inputs (no geo3d).
# def GLM_copy_from_pf(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
#     # 1. need to concatenate runs 
#     if len(runs) > 1:
#         Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
#     else:
#         Y_all = runs[0]
#         stim_df = stim_list[0]
#         runs_updated = runs
        
#     run_unit = Y_all.pint.units
#     # 2. define design matrix
#     dms = glm.design_matrix.hrf_regressors(
#                                     Y_all,
#                                     stim_df,
#                                     glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
#                                 )


#     # Combine drift and short-separation regressors (if any)
#     if cfg_GLM['do_drift']:
#         drift_regressors = model.get_drift_regressors(runs_updated, cfg_GLM)
#         dms &= reduce(operator.and_, drift_regressors)

#     if cfg_GLM['do_drift_legendre']:
#         drift_regressors = model.get_drift_legendre_regressors(runs_updated, cfg_GLM)
#         dms &= reduce(operator.and_, drift_regressors)

#     if cfg_GLM['do_short_sep']:
#         ss_regressors = model.get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
#         dms &= reduce(operator.and_, ss_regressors)

#     dms.common = dms.common.fillna(0)

#     # 3. get betas and covariance
#     results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
#     betas = results.sm.params
#     cov_params = results.sm.cov_params()

#     # 4. estimate HRF and MSE
#     basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

#     trial_type_list = stim_df['trial_type'].unique()

#     hrf_mse_list = []
#     hrf_estimate_list = []

#     for trial_type in trial_type_list:
        
#         betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
#         hrf_estimate = model.estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
#         cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
#                             regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
#                                     )
#         hrf_mse = model.estimate_HRF_cov(cov_hrf, basis_hrf)

#         hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
#         hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

#         hrf_estimate_list.append(hrf_estimate)
#         hrf_mse_list.append(hrf_mse)

#     hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
#     hrf_estimate = hrf_estimate.pint.quantify(run_unit)

#     hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
#     hrf_mse = hrf_mse.pint.quantify(run_unit**2)

#     # set universal time so that all hrfs have the same time base 
#     fs = frequency.sampling_rate(runs[0]).to('Hz')
#     before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
#     after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

#     dT = np.round(1 / fs, 3)  # millisecond precision
#     n_timepoints = len(hrf_estimate.time)
#     reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

#     hrf_mse = hrf_mse.assign_coords({'time': reltime})
#     hrf_mse.time.attrs['units'] = 'second'

#     hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
#     hrf_estimate.time.attrs['units'] = 'second'

#     return results, hrf_estimate, hrf_mse

#%% get Laura's HRF estimate, MSE, and model residual
result_dict_stim = dict()
run_list = []
pruned_chans_list = []
stim_list = []
for run_key in run_dict.keys():
    run_list.append(run_dict[run_key]['run'])
    pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
    ev_df = run_dict[run_key]['ev_df'].copy()
    # rename trial_type
    ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)].loc[:,'trial_type'] = 'mnt-correct'
    ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)].loc[:,'trial_type'] = 'mnt-incorrect'
    stim_list.append(ev_df[ev_df['trial_type']=='mnt'])

results, hrf_estimate, hrf_mse = model.GLM_copy_from_pf(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
result_dict_stim['resid'] = results.sm.resid
result_dict_stim['hrf_estimate'] = hrf_estimate
result_dict_stim['hrf_mse'] = hrf_mse

#%% save dict
save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
save_dict = dict(
    eeg_informed=result_dict,
    stim_only=result_dict_stim,
)
with open(os.path.join(save_file_path,f'sub-{subj_id}_glm_mnt_results.pkl'),'wb') as f:
    pickle.dump(save_dict,f)

#%% Visualizing MSE
summarized_method = lambda x, y: np.sum(x[:,:,:,y].values,axis=x.dims.index('time')).reshape(-1)
mse_check = 'HbO'
run_key = 'run01'
if mse_check == 'HbT':
    vis_mse = summarized_method(hrf_dict[run_key]['hrf_mse'],0)+summarized_method(hrf_dict[run_key]['hrf_mse'],1)
    vis_mse_laura = summarized_method(hrf_dict_laura[run_key]['hrf_mse'],0)+summarized_method(hrf_dict_laura[run_key]['hrf_mse'],1)
else:
    vis_mse = summarized_method(hrf_dict[run_key]['hrf_mse'],0 if mse_check=='HbO' else 1)
    vis_mse_laura = summarized_method(hrf_dict_laura[run_key]['hrf_mse'],0 if mse_check=='HbO' else 1)
# set visualization min/max
vmin = np.min(np.concat([vis_mse,vis_mse_laura]))
vmax = np.max(np.concat([vis_mse,vis_mse_laura]))

# plot
# f, ax = plt.subplots(1, 3, figsize=(20, 8))
# scalp_plot(
#     run_dict[run_key]['conc_ts'],
#     geo3d,
#     vis_mse_laura,
#     ax[0],
#     cmap='jet',
#     vmin=vmin,
#     vmax=vmax,
#     optode_labels=False,
#     title="Original",
#     optode_size=6,
# )
# scalp_plot(
#     run_dict[run_key]['conc_ts'],
#     geo3d,
#     vis_mse,
#     ax[1],
#     cmap='jet',
#     vmin=vmin,
#     vmax=vmax,
#     optode_labels=False,
#     title="EEG-informed",
#     optode_size=6,
# )
mse_diff = (vis_mse_laura-vis_mse)/vis_mse_laura
vmin = 0
vmax = 1
f, ax = plt.subplots(1, 2, figsize=(10, 8))
ax[0].boxplot(mse_diff, label=f"Median = {np.median(mse_diff)*100:.02f}%")
ax[0].set_xticklabels(['MSE reduced ratio'],fontsize=15)
ax[0].grid()
ax[0].legend(fontsize=15)
scalp_plot(
    run_dict[run_key]['conc_ts'],
    geo3d,
    mse_diff,
    ax[1],
    cmap='jet',
    vmin=vmin,
    vmax=vmax,
    optode_labels=False,
    # title="Original",
    optode_size=6,
)

#%% add other EEG channels

#%% Earse DM during missing events


#%% Add other events to DM
# for mnt_incorrect trials, look for N1 (0-200ms after response)
# ignore city incorrect for now



#%% load glm results
subj_id_array = [670, 695, 721, 723]
ev_name = 'mnt_correct'
hrf_mse_list = []
hrf_mse_list_laura = []
for subj_id in subj_id_array:
    load_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    # load EEG-informed results
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse.pkl'), 'rb') as f:
        data = pickle.load(f)
        hrf_mse_list.append(data)
    # load Laura'save_dict
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse_laura.pkl'), 'rb') as f:
        data = pickle.load(f)
        hrf_mse_list_laura.append(data)

#%% get geo3d template
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d_695 = results['geo3d']
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
run_dict_695 = copy.deepcopy(run_dict)

#%% treat all runs as an independent measurement. Ignore subject variability
from scipy.stats import ranksums, shapiro, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
# extract hrf_mse values
mse_values = np.concat([[np.squeeze(np.sum(y.values[:,:,:,0],axis=1)) for y in x['hrf']] for x in hrf_mse_list])
mse_values_laura = np.concat([[np.squeeze(np.sum(y.values[:,:,:,0],axis=1)) for y in x['hrf']] for x in hrf_mse_list_laura])
mse_diff = mse_values_laura-mse_values
mse_diff_ratio = (mse_values_laura-mse_values)/mse_values_laura
median_mse_diff_ratio = np.median(mse_diff_ratio,axis=0)
# check if the difference is normally distributed
shapiro_stat, shapiro_p = shapiro(mse_diff.reshape(-1))
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p={shapiro_p:.4e}")

# test
stats, p_values = wilcoxon(mse_values.reshape(-1), mse_values_laura.reshape(-1), alternative='two-sided')
median_diff = np.median(mse_values.reshape(-1)-mse_values_laura.reshape(-1))
print(f"Median of EEG-informed = {np.median(mse_values.reshape(-1))}")
print(f"Median of Stim-only = {np.median(mse_values_laura.reshape(-1))}")
print(f"p = {p_values}")
print(f"Median MSE reduction = {median_diff}")
if p_values<0.05:
    if median_diff>0:
        print("Stim-only has lower MSE.")
    else:
        print("EEG-informed has lower MSE.")
else:
    print("No significant difference between two methods.")
# # FDR correction across 561 channels
# rejected, p_values_fdr = fdrcorrection(p_values, alpha=0.05)
# print(f"Any p<=0.05 = {np.any(p_values_fdr<=0.05)}")

#%% Wilcoxon test across subjects with FDR correction
stats, p_values = wilcoxon(mse_values, mse_values_laura, alternative='two-sided')
median_diff = np.median(mse_values-mse_values_laura,axis=0)
print(f"Any p<=0.05 = {np.any(p_values<=0.05)}")
# FDR correction across 561 channels
rejected, p_values_fdr = fdrcorrection(p_values, alpha=0.05)
print(f"Any p_fdr<=0.05 = {np.any(p_values_fdr<=0.05)}")

#%% Repeated measure ANOVA
from statsmodels.stats.anova import AnovaRM
hrf_mse_df = []
for subj_i in range(len(subj_id_array)):
    load_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    # load EEG-informed results
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse.pkl'), 'rb') as f:
        data = pickle.load(f)
        for run_i in range(3):
            tmp = dict(
                subject= subj_id_array[subj_i],
                run_key=f"run{run_i+1:02d}",
                condition='EEG_informed',
                value= np.sum(np.squeeze(data['hrf'][run_i].values[:,:,:,0]),axis=0)
            )
            hrf_mse_df.append(tmp)
    # load Laura'save_dict
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse_laura.pkl'), 'rb') as f:
        data = pickle.load(f)
        for run_i in range(3):
            tmp = dict(
                subject= subj_id_array[subj_i],
                run_key=f"run{run_i+1:02d}",
                condition='Stim_only',
                value= np.sum(np.squeeze(data['hrf'][run_i].values[:,:,:,0]),axis=0)
            )
            hrf_mse_df.append(tmp)

hrf_mse_df = pd.DataFrame(hrf_mse_df)
print("Data structure:")
print(hrf_mse_df.head(12))
print(f"\nTotal observations: {len(hrf_mse_df)}")

# =============================================================================
# TWO-WAY REPEATED MEASURES ANOVA
# =============================================================================

print("\n" + "="*70)
print("TWO-WAY REPEATED MEASURES ANOVA (statsmodels)")
print("="*70)

# statsmodels AnovaRM
aovrm = AnovaRM(
    data=hrf_mse_df,
    depvar='value',              # dependent variable
    subject='subject',           # subject identifier (must be column name)
    within=['condition', 'run_key'], # within-subject factors
    aggregate_func='mean'        # how to handle multiple observations (shouldn't matter here)
)

results = aovrm.fit()
print(results)

#%%
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt

# Your data structure
# Shape: (n_observations, n_channels) = (24, 561)
# where 24 = 4 subjects × 3 runs × 2 conditions

np.random.seed(42)

# Simulate data
n_subjects = 4
n_runs = 3
n_conditions = 2
n_channels = 561

# Create data with channel-specific effects
data_array = np.zeros((n_subjects * n_runs * n_conditions, n_channels))
metadata = []

idx = 0
for subj_id in subj_id_array:
    load_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    # load EEG-informed results
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse.pkl'), 'rb') as f:
        hrf_eeg = pickle.load(f)
    # load Laura'save_dict
    with open(os.path.join(load_file_path, f'sub-{subj_id}_{ev_name}_hrf_mse_laura.pkl'), 'rb') as f:
        hrf_stim = pickle.load(f)
    for run in range(1, n_runs + 1):
        # Different channels have different effects
        for ch in range(n_channels):
            value=np.sum(np.squeeze(hrf_eeg['hrf'][run-1].values[:,:,:,0]),axis=0)[ch]
            data_array[idx, ch] = value
        metadata.append({
            'Subject': subj_id,
            'Run': run,
            'Condition': 'EEG-informed',
            'idx': idx
        })
        idx += 1
        for ch in range(n_channels):
            value=np.sum(np.squeeze(hrf_stim['hrf'][run-1].values[:,:,:,0]),axis=0)[ch]
            data_array[idx, ch] = value
        metadata.append({
            'Subject': subj_id,
            'Run': run,
            'Condition': 'Stim-only',
            'idx': idx
        })
        idx += 1

df_meta = pd.DataFrame(metadata)

print(f"Data shape: {data_array.shape}")
print(f"Metadata shape: {df_meta.shape}")
print(df_meta.head(6))

#%% =============================================================================
# CHANNEL-WISE REPEATED MEASURES ANOVA
# =============================================================================

print("\n" + "="*70)
print("CHANNEL-WISE REPEATED MEASURES ANOVA")
print("="*70)

# Store results for each channel
f_values = np.zeros(n_channels)
p_values = np.zeros(n_channels)

for ch in range(n_channels):
    # Create dataframe for this channel
    df_ch = df_meta.copy()
    df_ch['Value'] = data_array[:, ch]
    
    # Run RM-ANOVA for this channel
    try:
        aovrm = AnovaRM(
            data=df_ch,
            depvar='Value',
            subject='Subject',
            within=['Condition', 'Run']
        )
        results = aovrm.fit()
        
        # Extract F and p for Condition effect
        f_values[ch] = results.anova_table.loc['Condition', 'F Value']
        p_values[ch] = results.anova_table.loc['Condition', 'Pr > F']
        
    except Exception as e:
        print(f"Warning: Channel {ch} failed: {e}")
        f_values[ch] = np.nan
        p_values[ch] = 1.0
    
    if (ch + 1) % 100 == 0:
        print(f"Processed {ch + 1}/{n_channels} channels...")

print(f"Completed {n_channels} channels")

#%% =============================================================================
# MULTIPLE COMPARISONS CORRECTION
# =============================================================================

print("\n" + "="*70)
print("MULTIPLE COMPARISONS CORRECTION")
print("="*70)

# Remove NaN values
valid_channels = ~np.isnan(p_values)
p_values_valid = p_values[valid_channels]

# FDR correction (Benjamini-Hochberg)
reject_fdr, p_corrected_fdr, _, _ = multipletests(
    p_values_valid, 
    alpha=0.05, 
    method='fdr_bh'
)

# Bonferroni correction (more conservative)
reject_bonf, p_corrected_bonf, _, _ = multipletests(
    p_values_valid,
    alpha=0.05,
    method='bonferroni'
)

# Create full arrays (including NaN positions)
p_fdr = np.ones(n_channels)
p_bonf = np.ones(n_channels)
sig_fdr = np.zeros(n_channels, dtype=bool)
sig_bonf = np.zeros(n_channels, dtype=bool)

p_fdr[valid_channels] = p_corrected_fdr
p_bonf[valid_channels] = p_corrected_bonf
sig_fdr[valid_channels] = reject_fdr
sig_bonf[valid_channels] = reject_bonf

print(f"Uncorrected: {np.sum(p_values < 0.05)} significant channels (p < 0.05)")
print(f"FDR corrected: {np.sum(sig_fdr)} significant channels (q < 0.05)")
print(f"Bonferroni: {np.sum(sig_bonf)} significant channels (p < 0.05)")

# Identify significant channels
sig_channels_fdr = np.where(sig_fdr)[0]
print(f"\nSignificant channels (FDR): {sig_channels_fdr[:10]}...")  # First 10

#%% =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Histogram of p-values
ax = axes[0, 0]
ax.hist(p_values[valid_channels], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
ax.set_xlabel('p-value')
ax.set_ylabel('Number of channels')
ax.set_title('Distribution of p-values (uncorrected)')
ax.legend()

# Panel 2: F-values across channels
ax = axes[0, 1]
ax.plot(f_values, alpha=0.7)
ax.axhline(y=stats.f.ppf(0.95, 1, 3), color='red', linestyle='--', 
           label='Critical F (p=0.05)')
ax.set_xlabel('Channel')
ax.set_ylabel('F-value')
ax.set_title('F-values for Condition effect')
ax.legend()

# Panel 3: -log10(p) values
ax = axes[1, 0]
log_p = -np.log10(p_values + 1e-10)  # Add small value to avoid log(0)
ax.plot(log_p, alpha=0.7, label='Uncorrected')
ax.scatter(sig_channels_fdr, log_p[sig_channels_fdr], 
           color='red', s=20, label='FDR significant', zorder=5)
ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
ax.set_xlabel('Channel')
ax.set_ylabel('-log₁₀(p)')
ax.set_title('Statistical significance across channels')
ax.legend()

# Panel 4: Effect sizes (for significant channels)
ax = axes[1, 1]
# Calculate effect size for each channel
effect_sizes = np.zeros(n_channels)
for ch in range(n_channels):
    df_ch = df_meta.copy()
    df_ch['Value'] = data_array[:, ch]
    
    # Average across runs
    df_avg = df_ch.groupby(['Subject', 'Condition'])['Value'].mean().reset_index()
    df_pivot = df_avg.pivot(index='Subject', columns='Condition', values='Value')
    
    diff = df_pivot['EEG-informed'] - df_pivot['Stim-only']
    effect_sizes[ch] = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

ax.scatter(np.arange(n_channels), effect_sizes, alpha=0.3, s=10)
ax.scatter(sig_channels_fdr, effect_sizes[sig_channels_fdr], 
           color='red', s=30, label='FDR significant')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Channel')
ax.set_ylabel("Cohen's d")
ax.set_title('Effect sizes (Condition A - B)')
ax.legend()

plt.tight_layout()
plt.savefig('multichannel_anova_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved as 'multichannel_anova_results.png'")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: TOP 10 MOST SIGNIFICANT CHANNELS")
print("="*70)

# Sort by p-value
sorted_idx = np.argsort(p_values)
top_channels = sorted_idx[:10]

summary_df = pd.DataFrame({
    'Channel': top_channels,
    'F-value': f_values[top_channels],
    'p-value': p_values[top_channels],
    'p-FDR': p_fdr[top_channels],
    'p-Bonferroni': p_bonf[top_channels],
    "Cohen's d": effect_sizes[top_channels],
    'Significant (FDR)': sig_fdr[top_channels]
})

print(summary_df.to_string(index=False))



#%%
f, ax = plt.subplots(1, 1)
ax.boxplot(mse_diff_ratio.reshape(-1), label=f"Median = {np.median(mse_diff_ratio.reshape(-1))*100:.02f}%")
ax.set_ylim([-5,5])
ax.set_yticks(np.arange(-5,5,1))
ax.set_xticklabels(['MSE reduction ratio'],fontsize=15)
ax.grid()
ax.legend(fontsize=15)

#%%
f, ax = plt.subplots(1, 1)
scalp_plot(
    run_dict_695[run_key]['conc_ts'],
    geo3d_695,
    median_mse_diff_ratio,
    ax = ax,
    cmap='RdBu_r',
    vmin=-1,
    vmax=1,
    optode_labels=False,
    optode_size=6,
)


