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

#%% get event trials
def get_valid_event_idx(ev_name, single_subj_epoch_dict):
    ev_idx_dict = dict()
    for run_key in single_subj_epoch_dict.keys():
        ev_idx_dict[run_key] = dict()
        if len(single_subj_epoch_dict[run_key][ev_name])>0:
            # get preserved trial index
            ev_preserved_idx = np.where([len(log) == 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
            # get rejected trial index
            ev_rejected_idx = np.where([len(log) != 0 for log in single_subj_epoch_dict[run_key][ev_name].drop_log])[0]
            # add dict
            ev_idx_dict[run_key]['preserved'] = ev_preserved_idx
            ev_idx_dict[run_key]['rejected'] = ev_rejected_idx
        else:
            ev_idx_dict[run_key]['preserved'] = []
            ev_idx_dict[run_key]['rejected'] = []
    return ev_idx_dict

#%% get ERP area
def get_ERP_area(ev_name, single_subj_epoch_dict, is_norm=True):
    # define output dict
    erp_area_dict = dict()
    # define ERP period of interest
    if ev_name.endswith('response'):
        n2_window=(0,0.2)
        p3_window=(0,0.2)
    else:
        n2_window=(0.4, 0.7)
        p3_window=(0.6, 1.1)
    # for each run
    for run_key in single_subj_epoch_dict.keys():
        erp_area_dict[run_key]=dict()
        if len(single_subj_epoch_dict[run_key][ev_name])>0:
            t_vector = single_subj_epoch_dict[run_key][ev_name].times
            # for each channel, extract ERP area
            ev_eeg = single_subj_epoch_dict[run_key][ev_name].pick(picks='eeg')
            for ch_name in ev_eeg.ch_names:
                ev_ch_eeg = ev_eeg.get_data()[:,ev_eeg.ch_names.index(ch_name),:]            
                # Extract N2 and P3 features
                area_list = []
                for eeg_i in range(len(ev_ch_eeg)):
                    n2_p3_features = extract_n2_p3_features(ev_ch_eeg[eeg_i], t_vector,
                                                            n2_window=n2_window,
                                                            p3_window=p3_window)
                    n2_area = np.abs(n2_p3_features['n2_area'])
                    p3_area = np.abs(n2_p3_features['p3_area'])
                    if ev_name.endswith('respons'):
                        area_list.append(n2_area)
                    else:
                        area_list.append(n2_area+p3_area)
                # rescale area to range 0 to 1. (0 as 0, 1 as max(area))
                area_list = np.array(area_list)
                if is_norm:
                    area_list = area_list/np.max(area_list)
                # store results
                erp_area_dict[run_key][ch_name] = area_list
        else:
            erp_area_dict[run_key] = []
    return erp_area_dict

#%% get mnt_correct trials 
mnt_correct_idx_dict = get_valid_event_idx('mnt_correct',single_subj_epoch_dict)
mnt_correct_area_dict = get_ERP_area('mnt_correct', single_subj_epoch_dict)

# get mnt_incorrect trials
mnt_incorrect_idx_dict = get_valid_event_idx('mnt_incorrect_response',single_subj_epoch_dict)
mnt_incorrect_area_dict = get_ERP_area('mnt_incorrect_response', single_subj_epoch_dict)

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
# add events per run.
def add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=None, select_chs=['cz']):
    """
    select_chs: select channels to add to design matrix
    """
    dm_dict = dict()
    for run_key in run_dict.keys():
        dm_dict[run_key] = dict()
        target_run = run_dict[run_key]['run']
        conc_o = run_dict[run_key]['conc_ts']
        chs_pruned = run_dict[run_key]['chs_pruned']
        ev_df = run_dict[run_key]['ev_df']
        # for each run, get drift and short-separation regressors (if any)
        if cfg_GLM['do_drift']:
            drift_regressors = model.get_drift_regressors([conc_o], cfg_GLM)
        elif cfg_GLM['do_drift_legendre']:
            drift_regressors = model.get_drift_legendre_regressors([conc_o], cfg_GLM)
        else:
            drift_regressors = None
        if cfg_GLM['do_short_sep']:
            ss_regressors = model.get_short_regressors([conc_o], [chs_pruned], cfg_GLM['geo3d'], cfg_GLM)
        else:
            ss_regressors = None
        # for each event, create a dm list
        if not select_event:
            select_event = ev_dict[run_key].keys()
        for ev_name in select_event:
            match ev_name:
                case 'mnt_correct':
                    target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
                    # rename trial_type
                    target_ev_df.loc[:,'trial_type'] = 'mnt-correct'
                case 'mnt_incorrect':
                    target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
                    # rename trial_type
                    target_ev_df.loc[:,'trial_type'] = 'mnt-incorrect'
            # check if event exist
            if len(target_ev_df)==0:
                # store in dm_dict
                dm_dict[run_key][ev_name] = []
                continue
            # create design matrix
            dm_list = []
            for ev_i, event_id in enumerate(ev_dict[run_key][ev_name]['idx']['preserved']):
                dm = glm.design_matrix.hrf_regressors(
                                            target_run,
                                            target_ev_df.iloc[[event_id]],
                                            glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                        )
                # rescale by Cz area
                #TODO: allow multiple channels in DM
                dm.common = dm.common*ev_dict[run_key][ev_name]['area']['cz'][ev_i]
                # append
                dm_list.append(dm)
            # Create a new design matrix object with the concatenated common regressors
            dms = dm_list.pop()
            dms_common = dms.common
            # merge all dms along time axis
            while len(dm_list)>0:
                dms_common += dm_list.pop().common
            # assign merged common back to dms
            dms.common = dms_common
            # add drift and short-separation regressors if any
            if drift_regressors:
                dms &= reduce(operator.and_, drift_regressors)
                dms.common = dms.common.fillna(0)
            if ss_regressors:
                dms &= reduce(operator.and_, ss_regressors)
                dms.common = dms.common.fillna(0)
            # store in dm_dict
            dm_dict[run_key][ev_name] = dms
     
    return dm_dict

#%%
dm_dict = add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

#%% check dm
plt_dm = dm_dict['run01']['mnt_correct']
# using xr.DataArray.plot
f, ax = plt.subplots(1,1,figsize=(12,10))
plt_dm.common.sel(chromo="HbO", time=plt_dm.common.time < 600).T.plot(vmin=-2,vmax=2)
plt.title("Shared Regressors")
#p.xticks(rotation=90)
plt.show()

#%% get GLM fitting results for each subject from shank Jun 02 2025
# 3. get betas and covariance
ev_name = 'mnt_correct'
glm_results_dict = dict()
for run_key in tqdm(run_dict.keys(),leave=True, position=0):
    results = glm.fit(run_dict[run_key]['run'], dm_dict[run_key][ev_name], noise_model=cfg_GLM['noise_model'])
    glm_results_dict[run_key] = results

#%% tmp
trial_type_list = ['mnt']
hrf_dict_laura = dict()
for run_key in tqdm(run_dict.keys()):
    hrf_dict_laura[run_key] = dict()
    betas = glm_results_dict_laura[run_key].sm.params
    cov_params = glm_results_dict_laura[run_key].sm.cov_params()
    run_unit = run_dict[run_key]['run'].pint.units
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
    fs = frequency.sampling_rate(run_dict[run_key]['run']).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    hrf_dict_laura[run_key]['hrf_estimate'] = hrf_estimate
    hrf_dict_laura[run_key]['hrf_mse'] = hrf_mse


#%% get HRF and MSE for each run
# 4. estimate HRF and MSE
trial_type_list = ['mnt']
hrf_dict = dict()
for run_key in tqdm(run_dict.keys()):
    hrf_dict[run_key] = dict()
    betas = glm_results_dict[run_key].sm.params
    cov_params = glm_results_dict[run_key].sm.cov_params()
    run_unit = run_dict[run_key]['run'].pint.units
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
    fs = frequency.sampling_rate(run_dict[run_key]['run']).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    hrf_dict[run_key]['hrf_estimate'] = hrf_estimate
    hrf_dict[run_key]['hrf_mse'] = hrf_mse

#%% GLM model from pf.GLM()
# Unknown pf.GLM() loaded. Required only 4 inputs (no geo3d).
def GLM_copy_from_pf(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )


    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = model.get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = model.get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = model.get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
    betas = results.sm.params
    cov_params = results.sm.cov_params()

    # 4. estimate HRF and MSE
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

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
    fs = frequency.sampling_rate(runs[0]).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse

#%% get Laura's HRF estimate, MSE, and model residual
ev_name = 'mnt_correct'
glm_results_dict_laura = dict()
hrf_dict_laura = dict()
for run_key in tqdm(run_dict.keys(),leave=True, position=0):
    ev_df = run_dict[run_key]['ev_df']
    hrf_dict_laura[run_key] = dict()
    match ev_name:
        case 'mnt_correct':
            target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
            # rename trial_type
            target_ev_df.loc[:,'trial_type'] = 'mnt-correct'
        case 'mnt_incorrect':
            target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
            # rename trial_type
            target_ev_df.loc[:,'trial_type'] = 'mnt-incorrect'
    results, hrf_estimate, hrf_mse = GLM_copy_from_pf([run_dict[run_key]['run']], cfg_GLM, cfg_GLM['geo3d'], [run_dict[run_key]['chs_pruned']], [target_ev_df])
    glm_results_dict_laura[run_key] = results
    hrf_dict_laura[run_key]['hrf_estimate'] = hrf_estimate
    hrf_dict_laura[run_key]['hrf_mse'] = hrf_mse

#%% save dict
save_dict = dict(
    glm=glm_results_dict,
    glm_laura=glm_results_dict_laura,
    hrf=hrf_dict,
    hrf_laura=hrf_dict_laura
)
with open('./output/mnt_correct_results.pkl','wb') as f:
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



#%% load Laura's results
laura_results_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
with gzip.open(os.path.join(laura_results_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
    hrf_laura = pickle.load(f)
