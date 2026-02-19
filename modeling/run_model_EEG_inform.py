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

#%%
# subj_id_array = [670,695,721,723]
model_type='basis'
subj_id_array =[726, 730]
# subj_id_array = [670, 671, 673, 695, 719, 721, 723, 726, 727, 730, 733]

for subj_id in tqdm(subj_id_array):
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

    # Get EEG DM
    eeg_dm_dict = model.create_eeg_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

    # combine EEG DMs from all runs into one big DM
    Y_all, eeg_dm, runs_updated = model.concatenate_runs_dms(run_dict, eeg_dm_dict)

    #%% assign DM
    if model_type=='full':
        # Combine EEG DM with Reduced DM to get full model
        dm_all = model.combine_dm(eeg_dm, reduced_dm)
    elif model_type=='reduced':
        dm_all = reduced_dm
    else:
        dm_all = basis_dm

    #%% get GLM fitting results for each subject from shank Jun 02 2025
    print(f"Start EEG-informed GLM fitting (sub-{subj_id})")
    if model_type=='full':
        glm_results, autoReg_dict = model.my_fit(Y_all, dm_all)
    else:
        file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
        with open(os.path.join(file_path,f'sub-{subj_id}_glm_mnt_full.pkl'),'rb') as f:
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
    if model_type=='full':
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

    #%% get HRF and MSE for each run
    if model_type!='basis':
        # 4. estimate HRF and MSE
        trial_type_list = ['mnt-correct','mnt-incorrect']

        betas = glm_results.sm.params
        cov_params = glm_results.sm.cov_params()
        run_unit = Y_all.pint.units
        # check if it is a full model
        if np.any(betas.regressor.str.find('eeg')>0):
            # TODO: find an elegant way to check if _stim regressor is presented
            """
            NOTE: The number of regressors is fixed.
            """
            basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])
            basis_hrf = xr.concat([basis_hrf,basis_hrf],dim='component')
        else:
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

    #%%
    save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    with open(os.path.join(save_file_path,f'sub-{subj_id}_glm_mnt_{model_type}.pkl'),'wb') as f:
        pickle.dump(result_dict,f)


#%% Sanity check
subj_id = 695
DO_TDDR = False
DO_DRIFT = False
DO_DRIFT_LEGENDRE = True
DRIFT_ORDER = 3
F_MAX = 0
F_MIN = 0
cfg_GLM = {
    'do_drift': DO_DRIFT,
    'do_drift_legendre': DO_DRIFT_LEGENDRE,
    'do_short_sep': True,
    'drift_order' : DRIFT_ORDER,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 18*units.s
    }

run_files = glob.glob(os.path.join(root_dir,  subject, 'nirs',  f"{subject}_task-gradCPT_run-*_nirs.snirf"))
runs_with_events = []
for run_file in run_files:
    # extract run number using regex
    match = re.search(r"run-(\d+)", os.path.basename(run_file))
    if match:
        run_num = match.group(1)
        events_file = os.path.join(root_dir, subject, 'nirs',  f"{subject}_task-gradCPT_run-{run_num}_events.tsv")

        if os.path.exists(events_file):
            runs_with_events.append(run_num)

print(f"{subject}: runs with events = {runs_with_events}")

cfg_dataset = {

    'root_dir' : root_dir,
    'subj_ids' : [subject],
    'file_ids' : [f'gradCPT_run-{run}' for run in runs_with_events], 
    'subj_id_exclude' :[]
}

cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_thresh' : [1, 40]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_thresh' : [1e-3, 0.84]*units.V, # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : False,
    'flag_use_psp' : False,
    'channel_sel': None
}


cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, 
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : DO_TDDR,
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : False,
}

cfg_bandpass = { 
    'fmin' : F_MIN,
    'fmax' : F_MAX
}


cfg_preprocess = {
    'median_filt' : 1, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'cfg_GLM': cfg_GLM
}


# if block averaging on OD:
cfg_mse = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1e-3*units.V,
    'blockaverage_val' : 0 ,
     'mse_min_thresh' : 1e-6
    }

n_files_per_subject = len(cfg_dataset['file_ids'])

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

REC_STR = 'conc_o'
possible_trial_types = ['mnt-correct', 'mnt-incorrect', 'city-incorrect']    

trial_presence_list = []
stims_pruned_list = []

for stim, run in zip(all_stims, all_runs):
    mnt_trials = stim[stim['trial_type'] == 'mnt'].copy()
    mnt_trials.loc[mnt_trials['response_code'] == 0, 'trial_type'] = 'mnt-correct'
    mnt_trials.loc[mnt_trials['response_code'] == -2, 'trial_type'] = 'mnt-incorrect'

    city_trials = stim[(stim['trial_type'] == 'city') & (stim['response_code'] == -1)]
    city_trials['trial_type'] = 'city-incorrect'
    
    # Combine the filtered trials
    stims_pruned = pd.concat([mnt_trials, city_trials], ignore_index=True)
    run.stim = stims_pruned
    stims_pruned_list.append(stims_pruned)

run_ts_list = [run[REC_STR] for run in all_runs]
results, hrf_estimate, hrf_mse = pf.GLM(run_ts_list, cfg_GLM, geo3d, all_chs_pruned, stims_pruned_list)
residual = results.sm.resid

# reset the values for bad channels 
amp = all_runs[0]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
n_chs = len(amp.channel)
idx_amp = np.where(amp < cfg_mse['mse_amp_thresh'])[0]
idx_sat = np.where(all_chs_pruned[0] == 0.0)[0]
bad_indices = np.unique(np.concat([idx_amp, idx_sat]))

hrf_estimate = hrf_estimate.transpose('channel', 'time', 'chromo', 'trial_type')
hrf_estimate = hrf_estimate - hrf_estimate.sel(time=(hrf_estimate.time < 0)).mean('time')

hrf_mse = hrf_mse.transpose('channel', 'time', 'chromo', 'trial_type')

present_trial_types = hrf_estimate.trial_type.values.tolist()
presence = [tt in present_trial_types for tt in possible_trial_types] # creates bool mask
trial_presence_list.append(presence)

hrf_estimate = hrf_estimate.reindex({'trial_type': possible_trial_types})
hrf_mse = hrf_mse.reindex({'trial_type': possible_trial_types})

hrf_per_subj = hrf_estimate.expand_dims('subj')
hrf_per_subj = hrf_per_subj.assign_coords(subj=subject)

hrf_mse_per_subj = hrf_mse.expand_dims('subj')
hrf_mse_per_subj = hrf_mse_per_subj.assign_coords(subj=subject)

trial_presence_mask = xr.DataArray(trial_presence_list, 
                                dims = ['subj', 'trial_type'],
                                coords = {'subj': subject,
                                        'trial_type': possible_trial_types}
                                        )

print('HRF estimation complete')
