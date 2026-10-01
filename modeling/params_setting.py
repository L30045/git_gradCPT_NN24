#%% libraries
import os
import sys
from cedalion import units

#%% path setting
project_path = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24/'
# sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
# import processing_func as pf
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/processing_modules_v26/')
import processing_func as pf
# import image_recon_func as irf
# sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/cedalion-pipeline/workflow/scripts/modules')
# import module_preprocess as mpf

# all cont_EEG_cz_3-stage* models are fit on the same parcel Y_all (sensitive parcels, HbO,
# per-run window from first event to last event + 15 s, drift/GSR OLS-regressed out),
# saved once per subject by run_model_cont_EEG_fNIRS.py
Y_ALL_SHARED_REG_TYPE = 'cont_EEG_cz_3-stage'

def get_shared_Y_all_path(data_dir, subject, hp_flag):
    return os.path.join(data_dir, 'Y_all', f'{subject}_parcel_Y_all_truncated_to_trials_{hp_flag}.pkl.gz')

def get_own_Y_all_path(prefix):
    return os.path.join(os.path.dirname(prefix), 'Y_all', os.path.basename(prefix) + '_Y_all.pkl.gz')

def get_Y_all_path(prefix):
    """Y_all path for a model output prefix '<dir>/sub-xxx_<eeg_reg_type>_<noise_model>_<Hp|noHp>'.
    cont_EEG_cz_3-stage* models load the shared parcel_Y_all_truncated_to_trials_<Hp|noHp> file,
    unless the model still has its own Y_all/<name>_Y_all.pkl.gz (legacy runs fit on a slightly
    different Y_all). Other models load their own Y_all/<name>_Y_all.pkl.gz."""
    own_path = get_own_Y_all_path(prefix)
    subject, reg_noise_hp = os.path.basename(prefix).split('_', 1)
    if reg_noise_hp.startswith(Y_ALL_SHARED_REG_TYPE) and not os.path.exists(own_path):
        hp_flag = reg_noise_hp.rsplit('_', 1)[1]
        return get_shared_Y_all_path(os.path.dirname(prefix), subject, hp_flag)
    return own_path

# per-subject model outputs: <subject dir>/betas/<name>_betas.pkl, <subject dir>/stats/<name>_stats.pkl
# and <subject dir>/dm/<name>_dm_all.pkl.gz, where prefix = '<subject dir>/<name>'
def get_betas_path(prefix):
    return os.path.join(os.path.dirname(prefix), 'betas', os.path.basename(prefix) + '_betas.pkl')

def get_stats_path(prefix):
    return os.path.join(os.path.dirname(prefix), 'stats', os.path.basename(prefix) + '_stats.pkl')

def get_dm_all_path(prefix):
    return os.path.join(os.path.dirname(prefix), 'dm', os.path.basename(prefix) + '_dm_all.pkl.gz')

def get_dm_dict_path(subj_dir, fname='dm_dict.pkl'):
    return os.path.join(subj_dir, 'dm', fname)

def get_prefix_from_betas_path(betas_path):
    subj_dir = os.path.dirname(os.path.dirname(betas_path))
    return os.path.join(subj_dir, os.path.basename(betas_path)[:-len('_betas.pkl')])

#%%
ch_names = ['fz','cz','pz','oz']
is_bpfilter = True
bp_f_range = [0.1, 45] #band pass filter range (Hz)
is_reref = True
reref_ch = ['tp9h','tp10h']
is_ica_rmEye = True
baseline_length = -0.2
epoch_reject_crit = dict(
                        eeg=100e-6 #unit:V
                        )
is_detrend = 1 # 0:constant, 1:linear, None

preproc_params = dict(
    is_bpfilter = is_bpfilter,
    bp_f_range = bp_f_range,
    is_reref = is_reref,
    reref_ch = reref_ch,
    is_ica_rmEye = is_ica_rmEye,
    baseline_length = baseline_length,
    epoch_reject_crit = epoch_reject_crit,
    is_detrend = is_detrend,
    ch_names = ch_names,
    is_overwrite = False
)

#%% GLM setup 
# RUN_PREPROCESS = True
# RUN_HRF_ESTIMATION = True
# SPLIT_VTC = False
# SAVE_RESIDUAL = False
# NOISE_MODEL = 'ar_irls'
# root_dir = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"

# if NOISE_MODEL == 'ols':
#     DO_TDDR = True
#     DO_DRIFT = True
#     DO_DRIFT_LEGENDRE = False
#     DRIFT_ORDER = 3
#     F_MIN = 0 * units.Hz
#     F_MAX = 0.5 * units.Hz
# elif NOISE_MODEL == 'ar_irls':
#     DO_TDDR = False
#     DO_DRIFT = False
#     DO_DRIFT_LEGENDRE = True
#     DRIFT_ORDER = 3
#     F_MAX = 0
#     F_MIN = 0
# else:
#     print('Not a valid noise model - please select ols or ar_irls')

# cfg_GLM = {
#     'do_drift': DO_DRIFT,
#     'do_drift_legendre': DO_DRIFT_LEGENDRE,
#     'do_short_sep': True,
#     'drift_order' : DRIFT_ORDER,
#     'distance_threshold' : 20*units.mm, # for ssr
#     'short_channel_method' : 'mean',
#     'noise_model' : NOISE_MODEL,
#     't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
#     't_std' : 1*units.s ,  
#     't_pre' : 2*units.s,
#     't_post' : 10*units.s
#     }

#%% Initial directory and analysis parameters
SPLIT_VTC = False
SAVE_RESIDUAL = False
USE_GSR = True
NOISE_MODEL = 'ar_irls'
root_dir = "/projectnb/nphfnirs/s/datasets/gradCPT_NN24/"
ADOT_FLAG = 'probe'
spatial_dim = 'vertex'
hrf_basis = 'cons_gaussians' # double_gamma_deriv, gamma_deriv, cons_gaussians
flag = ''

if NOISE_MODEL == 'ols':
    DO_TDDR = True
    DO_DRIFT = True
    DO_DRIFT_LEGENDRE = False
    DRIFT_ORDER = 3
    F_MIN = 0 * units.Hz
    F_MAX = 0.5 * units.Hz
elif NOISE_MODEL == 'ar_irls':
    DO_TDDR = False
    DO_DRIFT = False
    DO_DRIFT_LEGENDRE = True
    DRIFT_ORDER = 3
    F_MAX = 0
    F_MIN = 0
else:
    print('Not a valid noise model - please select ols or ar_irls')

cfg_GLM = {
    'do_drift': DO_DRIFT,
    'do_drift_legendre': DO_DRIFT_LEGENDRE,
    'do_short_sep': False,
    'drift_order' : DRIFT_ORDER,
    'do_GSR': USE_GSR,
    'GSR_weight': None,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : NOISE_MODEL,
    'HRF_basis': hrf_basis, 

    # double gamma deriv
    'peak_time': 4*units.s,
    'peak_disp': 1*units.s,
    'undershoot_time': 16*units.s,
    'undershoot_disp': 1*units.s,
    'ratio': 1/6,
    'duration': 18*units.s,

    # gamma deriv
    'tau': {'HbO': 1.8*units.s, 'HbR':2.5*units.s}, 
    'sigma': {'HbO':3*units.s, 'HbR':3*units.s}, 
    'dur': 4*units.s,

    # consecutive gaussians
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,  
    't_pre' : 2*units.s,
    't_post' : 18*units.s
    }


#%% excluded subject due to low fNIRS quality
excluded_subj = ['sub-641', 'sub-644', 'sub-655',
            'sub-657', 'sub-658', 
            'sub-671', 'sub-673', 
            'sub-733', 'sub-746',
            'sub-755',
            'sub-763', 'sub-764']