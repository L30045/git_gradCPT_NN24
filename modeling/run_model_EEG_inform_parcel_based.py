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
import re
import cedalion
import cedalion.sigproc.frequency
import xarray as xr
import cedalion.models.glm as glm

#%% select model type
model_type='eeg_alpha'
is_overwrite = False # If True, force re-training GLM.
is_hpf = 'nohpf' not in model_type # high-pass filter conc_o before building DMs
hpf_freq = 0.02 * units.Hz
select_chromo = 'HbO'
cfg_GLM['do_short_sep'] = False # short-sep regressors are channel-space only; not available for parcel data

#%% find subjects with fNIRS and enough EEG epochs
_project_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24'
_eeg_deriv = os.path.join(_project_path, 'derivatives', 'eeg')
_MIN_EPOCHS = 500

_fnirs_subjects = {
    re.search(r'sub-(\d+)', f).group(1)
    for f in glob.glob(os.path.join(_project_path, 'sub-*', 'nirs', '*task-gradCPT*nirs.snirf'))
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

#%% start training GLM for each subject each channel
for subj_id in tqdm(subj_id_array):
    print(f"Start processing sub-{subj_id}")
    save_file_path = os.path.join(project_path, 'derivatives', 'eeg', f"sub-{subj_id}")
    pkl_path = os.path.join(save_file_path, f'sub-{subj_id}_glm_mnt_{model_type}.pkl')
    if not is_overwrite and os.path.exists(pkl_path):
        print(f"Skipping sub-{subj_id}: output already exists.")
        continue
    # load channel-space results (for stims / chs_pruned only; parcel data loaded below)
    der_dir = os.path.join(root_dir, 'derivatives', 'cedalion', 'pipeline_reorder', 'processed_data')
    hbo_file = os.path.join(der_dir, f"sub-{subj_id}", f"sub-{subj_id}_preprocessed_results_{NOISE_MODEL}.pkl")
    if not os.path.exists(hbo_file):
        print(f"Skipping sub-{subj_id}: preprocessed results file not found.")
        continue
    with gzip.open(hbo_file, 'rb') as f:
        results = pickle.load(f)

    all_chs_pruned = results['chs_pruned']
    all_stims = results['stims']
    geo3d = results['geo3d']
    cfg_GLM['geo3d'] = geo3d

    # load image-space (parcel) results
    image_file = os.path.join(der_dir, f"sub-{subj_id}",
                               f"sub-{subj_id}_task-gradCPT_adot-{ADOT_FLAG}_spatialdim-{spatial_dim}_IR_ts_{NOISE_MODEL}{flag}.pkl")
    if not os.path.exists(image_file):
        print(f"Skipping sub-{subj_id}: image-space results file not found.")
        continue
    with open(image_file, 'rb') as f:
        image_results = pickle.load(f)

    if weight_flag == 'aca':
        all_runs = image_results['parcel_ts_aca']
        vv = image_results['vertex_aca']
    elif weight_flag == 'post':
        all_runs = image_results['parcel_ts_post']
        vv = image_results['vertex_mse']
    else:
        all_runs = image_results['parcel_ts_none']
        vv = image_results['vertex_mse']

    n_runs = len(vv)
    vv = xr.concat(vv, dim='run').sum('run') / n_runs**2
    vp = vv.groupby('parcel').sum('vertex') / vv.groupby('parcel').count()**2

    if cfg_GLM['do_GSR']:
        aca_lst = image_results['vertex_aca']
        aca_p_lst = []
        for aca in aca_lst:
            aca_p = aca.groupby('parcel').sum('vertex') / aca.groupby('parcel').count()**2
            aca_p = aca_p.sel(parcel=aca_p.parcel != 'scalp')
            aca_p_lst.append(aca_p)
        cfg_GLM['GSR_weight'] = aca_p_lst

    all_runs = [run.assign_coords({'samples': ('time', np.arange(len(run.time)))}) for run in all_runs]

    all_runs_tmp = []
    for run in all_runs:
        run.time.attrs['units'] = units.s
        run = run.sel(parcel=run.parcel != 'scalp')
        all_runs_tmp.append(run)
    all_runs = all_runs_tmp

    if select_chromo is not None:
        all_runs = [x.sel(chromo=[select_chromo]) for x in all_runs]

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

    # find corresponding runs in all_runs (parcel space) and assign to run_dict,
    # matching via the channel-space stim table (all_stims) since parcel runs have no .stim
    for r_i, stim in enumerate(all_stims):
        for run_key in run_dict.keys():
            ev_df = run_dict[run_key]['ev_df']
            if len(ev_df) > 0 and len(stim) > 0 and np.all(stim.iloc[0] == ev_df.iloc[0]):
                run_dict[run_key]['run'] = all_runs[r_i]
                run_dict[run_key]['chs_pruned'] = all_chs_pruned[r_i]
                break

    # epoch length
    len_epoch = 12 # seconds
    t_conc_ts = run_dict[run_key]['run'].time
    sfreq_conc = 1/np.diff(t_conc_ts)[0]
    len_epoch_sample = np.ceil(len_epoch*sfreq_conc).astype(int)

    #%% get epoched EEG
    # load eeg to match the time
    single_subj_EEG_dict, single_subj_rm_ch_dict = utils.eeg_preproc_subj_level(subj_id, preproc_params)
    single_subj_epoch_dict, single_subj_vtc_dict, single_subj_react_dict, event_labels_lookup = utils.eeg_epoch_subj_level(f"sub-{subj_id}", single_subj_EEG_dict, preproc_params)

    
    # get mnt_correct trials
    mnt_correct_idx_dict = model.get_valid_event_idx('mnt_correct',single_subj_epoch_dict)
    mnt_correct_area_dict = model.get_alpha_power('mnt_correct', single_subj_epoch_dict)

    # get mnt_incorrect trials
    mnt_incorrect_idx_dict = model.get_valid_event_idx('mnt_incorrect_response',single_subj_epoch_dict)
    mnt_incorrect_area_dict = model.get_alpha_power('mnt_incorrect_response', single_subj_epoch_dict)

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
        local_run = run_dict[run_key]['run']
        # high pass filter
        if is_hpf:
            local_run = cedalion.sigproc.frequency.freq_filter(
                local_run, fmin=hpf_freq, fmax=0 * units.Hz, butter_order=4
            )
        run_list.append(local_run)
        pruned_chans_list.append(run_dict[run_key]['chs_pruned'])
        ev_df = run_dict[run_key]['ev_df'].copy()
        # rename trial_type
        ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]==0),'trial_type'] = 'mnt-correct-stim'
        ev_df.loc[(ev_df['trial_type']=='mnt')&(ev_df["response_code"]!=0),'trial_type'] = 'mnt-incorrect-stim'
        stim_list.append(ev_df[(ev_df['trial_type']=='mnt-correct-stim')|(ev_df['trial_type']=='mnt-incorrect-stim')])
    stim_dm = model.get_GLM_copy_from_pf_DM(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)
    Y_all, _, runs_updated = model.concatenate_runs(run_list, stim_list)

    # get drift and ss
    basis_dm = model.create_no_info_dm(run_list, cfg_GLM, cfg_GLM['geo3d'], pruned_chans_list, stim_list)

    # Get EEG DM
    eeg_dm_dict = model.create_eeg_dm(run_dict, ev_dict, cfg_GLM, select_event=['mnt_correct','mnt_incorrect'], select_chs=['cz'])

    # combine EEG DMs from all runs into one big DM
    Y_all, eeg_dm, runs_updated = model.concatenate_runs_dms(run_dict, eeg_dm_dict)

    # save DMs
    save_file_path = os.path.join(project_path, 'derivatives','eeg', f"sub-{subj_id}")
    save_dm_name = os.path.join(save_file_path, 'dm_dict.pkl')
    if not os.path.exists(save_dm_name):
        dm_dict = dict()
        dm_dict['basis']=basis_dm
        dm_dict['onlyEEG']=model.combine_dm(eeg_dm, basis_dm)
        dm_dict['onlyStim']=stim_dm
        dm_dict['full']=model.combine_dm(eeg_dm, stim_dm)
        dm_dict['Y_all']=Y_all
        with open(save_dm_name,'wb') as f:
            pickle.dump(dm_dict,f)

    #%% assign DM
    if model_type.startswith('full'):
        # Combine EEG DM with Reduced DM to get full model
        dm_all = model.combine_dm(eeg_dm, stim_dm)
    elif model_type.startswith('onlyStim'):
        dm_all = stim_dm
    elif model_type.startswith('onlyEEG'):
        dm_all = model.combine_dm(eeg_dm, basis_dm)
    else:
        dm_all = basis_dm

    #%% select chromo=HbO only to save time
    Y_all = Y_all.sel(chromo=['HbO'])
    dm_all.common = dm_all.common.sel(chromo=['HbO'])

    #%% get GLM fitting results for each subject (parcel space: OLS via cedalion glm.fit)
    print(f"Start EEG-informed GLM fitting (sub-{subj_id})")
    glm_results = glm.fit(Y_all, dm_all, noise_model=cfg_GLM['noise_model'])

    # 3. get betas and covariance
    result_dict = dict()
    # result_dict['resid'] = glm_results.sm.resid
    betas = glm_results.sm.params
    cov_params = glm_results.sm.cov_params()
    result_dict['betas']=betas
    result_dict['cov_params']=cov_params

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
    elif model_type.startswith('onlyStim'):
        param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
        # Create hypothesis strings
        hypotheses = [f'{name} = 0' for name in param_names]
        # Run F-test
        f_test_result = glm_results.sm.f_test(hypotheses)
        result_dict['f_test_stim_basis'] = f_test_result
    elif model_type.startswith('onlyEEG'):
        param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
        # Create hypothesis strings
        hypotheses = [f'{name} = 0' for name in param_names]
        # Run F-test
        f_test_result = glm_results.sm.f_test(hypotheses)
        result_dict['f_test_eeg_basis'] = f_test_result

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
    elif model_type.startswith('onlyStim'):
        param_names = [name for name in glm_results.sm.params.regressor.values if 'stim' in name]
        # Create hypothesis strings
        hypotheses = '+'.join(param_names)+' = 0'
        # Run F-test
        t_test_result = glm_results.sm.t_test(hypotheses)
        result_dict['t_test_0_stim'] = t_test_result
    elif model_type.startswith('onlyEEG'):
        param_names = [name for name in glm_results.sm.params.regressor.values if 'eeg' in name]
        # Create hypothesis strings
        hypotheses = '+'.join(param_names)+' = 0'
        # Run F-test
        t_test_result = glm_results.sm.t_test(hypotheses)
        result_dict['t_test_0_eeg'] = t_test_result

    #%% get HRF and MSE for each run
    if not model_type.startswith('basis'):
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
            basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])
            basis_hrf = model.xr.concat([basis_hrf,basis_hrf],dim='component')
        else:
            basis_hrf = model.glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(run_dict[run_key]['run'])


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
    # with open(os.path.join(save_file_path,f'sub-{subj_id}_dev_reduced.pkl'),'wb') as f:
    #     pickle.dump(result_dict,f)
print("All trainings completed.")
