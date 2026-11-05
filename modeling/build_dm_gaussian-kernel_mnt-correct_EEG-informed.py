#%% Build the Gaussian-kernel design matrix (mnt-correct, -2 to 18 s) with EEG-informed regressors
# For each subject, builds the event regressors used by run_model_EEG_inform_parcel_based.py and saves them to
#   <data_save_path>/dm/gaussian-kernel_mnt-correct_EEG-informed_Cz-P3-amp-Woody.pkl
#
# Regressors (20 Gaussian kernels each: centres -2 ... 17 s, 1 s spacing, t_std 1 s):
#   'HRF mnt-correct-stim XX'  every mnt-correct trial with amplitude 1
#   'HRF mnt-correct-eeg XX'   every mnt-correct trial scaled by its single-trial N2-to-P3 peak-to-peak amplitude
#                              (Woody P3 peak minus the trial's minimum between 0.25 s and that peak; >= 0)
#                              (cz, else pz; preproc_pipe/extract_EEG_trial_features.py)
# Only mnt-correct trials are modelled (as in run_model_gaussian-kernel_mnt-correct_network-based.py and the
# Bayesian single-parcel model). No drift / short-separation / GSR columns: Y_all is the shared, 3-stage
# residualized parcel time series of run_model_cont_EEG_fNIRS.py (drift and GSR already OLS-regressed out).
#
# Time base: the DM rows are the samples of the shared Y_all (<data_save_path>/Y_all/
# <subject>_parcel_Y_all_truncated_to_trials_<Hp|noHp>.pkl.gz). Kernels are built on each run's full fNIRS
# sample grid (time = sample index * dt, 0 at the run start like the fNIRS run, onsets from the run's nirs
# events.tsv) and then sampled at the run's Y_all samples, so trials near the window edges are not cut.
# The DM carries Y_all's 'samples' coordinate so the model script can check the alignment.
#
# Woody scaling (as model.get_ERP_area / create_eeg_dm, but divided by the mean instead of the max): within each
# run, a clean trial's amplitude is divided by the run's mean amplitude over clean trials, so the average clean
# trial has scale 1; rejected trials (epoch > 100 uV on the channel, or NaN amplitude) get the mean scale (1).
# The peak-to-peak amplitude is >= 0, so every scale is >= 0. A run whose clean mean is <= 0 is an error.
#
# Saved pickle: dict with
#   'dm'        {'full': DesignMatrix (eeg then stim), 'onlyStim': ..., 'onlyEEG': ...}
#               ('full' / 'onlyEEG' only if the EEG feature table exists)
#   'cfg_HRF_basis' (seconds), 'eeg_amp_feature', 'eeg_amp_chs', 'trial_info' (per-run trials and scales),
#   'Y_all_path', 'hp_flag'
#%% load library
import os
import sys
import gzip
import glob
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from params_setting import *
import cedalion.models.glm as glm
from cedalion.models.glm.design_matrix import DesignMatrix
from extract_EEG_trial_features import get_features_path, load_trial_features, get_trial_modulator

#%% settings
is_overwrite = True  # rebuild existing DMs
hp_flag_Y = 'Hp'  # Y_all whose samples define the DM rows (Hp and noHp Y_all share the same windows)
run_keys = ['gradcpt1', 'gradcpt2', 'gradcpt3']  # order of the runs in Y_all
chromos = ['HbO', 'HbR']
cfg_HRF_basis = {'t_pre': 2*units.s, 't_post': 18*units.s, 't_delta': 1*units.s, 't_std': 1*units.s}
EEG_AMP_FEATURE = 'N2P3_ptp_woody'  # column suffix in the feature table: <ch>_<EEG_AMP_FEATURE>
EEG_AMP_CHS = ['cz', 'pz']        # first channel with clean trials in a run is used

gauss_kernels = glm.GaussianKernels(cfg_HRF_basis['t_pre'], cfg_HRF_basis['t_post'],
                                    cfg_HRF_basis['t_delta'], cfg_HRF_basis['t_std'])


#%% helpers
def get_woody_scale(subject, run_id, ev_df, feat_df):
    """Per-trial scale for the rows of ev_df (mnt-correct trials of one run).
    Returns (scale, channel, n clean trials); scale is None if no channel has clean trials."""
    for ch in EEG_AMP_CHS:
        col = f'{ch}_{EEG_AMP_FEATURE}'
        if col not in feat_df:
            continue
        f = get_trial_modulator(subject, run_id, ev_df, [col, f'{ch}_erp_ok'], feat_df=feat_df)
        amp = f[col].values.astype(float)
        ok = f[f'{ch}_erp_ok'].astype(bool).values & np.isfinite(amp)
        if ok.sum() == 0:
            continue
        # divide by the run's mean clean amplitude; rejected trials get the mean scale (1)
        amp_mean = amp[ok].mean()
        if amp_mean <= 0:
            raise ValueError(f"{subject} run {run_id}: mean clean {col} = {amp_mean:.3g} <= 0, cannot scale by it")
        scale = amp / amp_mean
        scale[~ok] = scale[ok].mean()
        return scale, ch, int(ok.sum())
    return None, None, 0


def run_regressors(stim, run_samples, dt):
    """Gaussian-kernel regressors (time x regressor x chromo) of stim on the full fNIRS run grid
    (time = sample index * dt), sampled at run_samples."""
    n_full = run_samples.max() + 1 + int(np.ceil(cfg_HRF_basis['t_post'].magnitude / dt))
    full_run = xr.DataArray(np.zeros((len(chromos), 1, n_full)), dims=('chromo', 'parcel', 'time'),
                            coords={'chromo': chromos, 'parcel': ['dummy'], 'time': np.arange(n_full) * dt,
                                    'samples': ('time', np.arange(n_full))}).pint.quantify('molar')
    full_run.time.attrs['units'] = units.s
    X = glm.design_matrix.hrf_regressors(full_run, stim, gauss_kernels).common.fillna(0)
    if hasattr(X.data, 'magnitude'):
        X = X.pint.dequantify()
    return X.isel(time=run_samples).transpose('time', 'regressor', 'chromo').drop_vars('samples', errors='ignore')


def build_subject(subject, data_save_path):
    Y_all_path = get_shared_Y_all_path(data_save_path, subject, hp_flag_Y)
    if not os.path.exists(Y_all_path):
        print(f"{subject}: {os.path.basename(Y_all_path)} not found, skipping.")
        return None
    with gzip.open(Y_all_path, 'rb') as f:
        Y_all = pickle.load(f)
    samples = Y_all.samples.values  # per-run fNIRS sample index; resets mark run starts in Y_all
    t_all = Y_all.time.values
    seg_bounds = np.concatenate([[0], np.where(np.diff(samples) <= 0)[0] + 1, [len(samples)]])
    if len(seg_bounds) - 1 != len(run_keys):
        print(f"{subject}: Y_all has {len(seg_bounds) - 1} runs, expected {len(run_keys)}, skipping.")
        return None
    seg_slices = [slice(seg_bounds[k], seg_bounds[k + 1]) for k in range(len(run_keys))]
    dt = np.median(np.diff(t_all))

    feat_path = get_features_path(subject)
    feat_df = load_trial_features(subject) if os.path.exists(feat_path) else None
    if feat_df is None:
        print(f"{subject}: EEG trial features not found, building onlyStim only.")

    X_stim_runs, X_eeg_runs, trial_info = [], [], {}
    for run_key, seg in zip(run_keys, seg_slices):
        run_id = int(run_key[-1])
        ev_df = pd.read_csv(os.path.join(project_path, subject, 'nirs',
                                         f"{subject}_task-gradCPT_run-{run_id:02d}_events.tsv"), sep='\t')
        mnt = ev_df[(ev_df['trial_type'] == 'mnt') & (ev_df['response_code'] == 0)]
        run_samples = samples[seg]
        info = dict(n_trials=len(mnt), trial_idx=mnt.index.values)

        stim = pd.DataFrame({'onset': mnt['onset'].values, 'duration': mnt['duration'].values,
                             'value': 1.0, 'trial_type': 'mnt-correct-stim'})
        X_stim = run_regressors(stim, run_samples, dt)
        X_stim_runs.append(X_stim.assign_coords(time=t_all[seg]))

        if feat_df is not None:
            scale, ch, n_ok = get_woody_scale(subject, run_id, mnt, feat_df)
            if scale is None:
                print(f"  {run_key}: no clean {EEG_AMP_FEATURE} trials, eeg regressors set to 0")
                X_eeg = xr.zeros_like(X_stim)
            else:
                stim_eeg = stim.assign(value=scale, trial_type='mnt-correct-eeg')
                X_eeg = run_regressors(stim_eeg, run_samples, dt)
            X_eeg = X_eeg.assign_coords(regressor=[r.replace('-stim', '-eeg') for r in X_stim.regressor.values])
            X_eeg_runs.append(X_eeg.assign_coords(time=t_all[seg]))
            info.update(channel=ch, n_clean=n_ok, scale=scale)
            print(f"  {run_key}: {len(mnt)} mnt-correct trials, {ch}_{EEG_AMP_FEATURE} ({n_ok} clean)")
        else:
            print(f"  {run_key}: {len(mnt)} mnt-correct trials")
        trial_info[run_key] = info

    def to_dm(runs):
        common = xr.concat(runs, dim='time').assign_coords(samples=('time', samples))
        return DesignMatrix(common=common)

    dms = {'onlyStim': to_dm(X_stim_runs)}
    if feat_df is not None:
        dms['onlyEEG'] = to_dm(X_eeg_runs)
        # eeg regressors first, then stim (the HRF expansion in the model script relies on this order)
        dms['full'] = to_dm([xr.concat([e, s], dim='regressor') for e, s in zip(X_eeg_runs, X_stim_runs)])

    return dict(dm=dms,
                cfg_HRF_basis={k: float(v.to(units.s).magnitude) for k, v in cfg_HRF_basis.items()},
                eeg_amp_feature=EEG_AMP_FEATURE, eeg_amp_chs=EEG_AMP_CHS, trial_info=trial_info,
                Y_all_path=Y_all_path, hp_flag=hp_flag_Y)


#%% main
if __name__ == '__main__':
    subjects = sorted(os.path.basename(os.path.dirname(os.path.dirname(p))).split('_')[0] for p in
                      glob.glob(os.path.join(project_path, 'derivatives', 'eeg', 'sub-*', 'Y_all',
                                             f'sub-*_parcel_Y_all_truncated_to_trials_{hp_flag_Y}.pkl.gz')))
    subjects = [s for s in subjects if s not in excluded_subj]
    print(f"{len(subjects)} subjects with a shared Y_all (not in excluded_subj): {subjects}")
    for subject in tqdm(subjects):
        data_save_path = os.path.join(project_path, 'derivatives', 'eeg', subject)
        save_path = get_gauss_eeg_dm_path(data_save_path)
        if os.path.exists(save_path) and not is_overwrite:
            print(f"{subject}: {GAUSS_EEG_DM_FNAME} exists, skipping.")
            continue
        print(f"Building DM for {subject}")
        out = build_subject(subject, data_save_path)
        if out is None:
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"{subject}: saved {list(out['dm'])} -> {save_path}")
