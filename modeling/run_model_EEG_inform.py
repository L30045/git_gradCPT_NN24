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
from params_setting import *

#%% load HbO
subj_id_array = [670, 671, 673, 695]

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
    single_subj_EEG_dict, single_subj_rm_ch_dict = eeg_preproc_subj_level(subj_id, preproc_params)
    single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)

    
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

    #%%
    dm_dict = model.add_ev_to_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

    #%% get GLM fitting results for each subject from shank Jun 02 2025
    # 3. get betas and covariance
    for ev_name in ['mnt_correct']:
        glm_results_dict = dict()
        for run_key in tqdm(run_dict.keys(),leave=True, position=0):
            results = glm.fit(run_dict[run_key]['run'], dm_dict[run_key][ev_name], noise_model=cfg_GLM['noise_model'])
            glm_results_dict[run_key] = results

        #%% get HRF and MSE for each run
        # 4. estimate HRF and MSE
        trial_type_list = [ev_name]
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
        
        save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
        save_dict = dict(
            glm=glm_results_dict,
            hrf=hrf_dict,
        )
        with open(os.path.join(save_file_path,f'sub-{subj_id}_glm_{ev_name}_results.pkl'),'wb') as f:
            pickle.dump(save_dict,f)

        # get Laura's HRF estimate, MSE, and model residual
        glm_results_dict_laura = dict()
        hrf_dict_laura = dict()
        for run_key in tqdm(run_dict.keys(),leave=True, position=0):
            ev_df = run_dict[run_key]['ev_df']
            hrf_dict_laura[run_key] = dict()
            if ev_name=='mnt_correct':
            
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt_correct'
            elif ev_name=='mnt_incorrect':
                target_ev_df = ev_df[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0)]
                # rename trial_type
                target_ev_df.loc[:,'trial_type'] = 'mnt_incorrect'
            results, hrf_estimate, hrf_mse = model.GLM_copy_from_pf([run_dict[run_key]['run']], cfg_GLM, cfg_GLM['geo3d'], [run_dict[run_key]['chs_pruned']], [target_ev_df])
            glm_results_dict_laura[run_key] = results
            hrf_dict_laura[run_key]['hrf_estimate'] = hrf_estimate
            hrf_dict_laura[run_key]['hrf_mse'] = hrf_mse

        # save dict
        save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
        save_dict = dict(
            glm=glm_results_dict_laura,
            hrf=hrf_dict_laura
        )
        with open(os.path.join(save_file_path,f'sub-{subj_id}_glm_{ev_name}_results_laura.pkl'),'wb') as f:
            pickle.dump(save_dict,f)