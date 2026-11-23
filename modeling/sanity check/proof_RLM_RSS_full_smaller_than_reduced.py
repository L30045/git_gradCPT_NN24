#%%
import numpy as np
from scipy import stats

def irls_fit(y, X, max_iter=50, tol=1e-6):
    """
    Simple IRLS (Iteratively Reweighted Least Squares) 
    using Huber weights.
    
    Returns: beta, weights, weighted_rss
    """
    n, p = X.shape
    
    # Initialize with OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        # Compute residuals
        residuals = y - X @ beta
        
        # Estimate scale (MAD)
        scale = np.median(np.abs(residuals)) / 0.6745
        
        if scale < 1e-10:
            break
        
        # Compute Huber weights
        r_scaled = residuals / scale
        k = 1.345  # Huber constant
        
        # Huber weight function
        weights = np.where(np.abs(r_scaled) <= k,
                           1.0,                      # Points near center
                           k / np.abs(r_scaled))     # Downweight outliers
        
        # Weighted least squares step
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    
    # Final residuals and weights
    residuals = y - X @ beta
    
    # Raw RSS (unweighted, in original space)
    raw_rss = np.sum(residuals**2)
    
    # Weighted RSS (what IRLS minimized)
    weighted_rss = np.sum(weights * residuals**2)
    
    return beta, weights, raw_rss, weighted_rss

#%% ============================================
# Generate data WITH outliers AND AR(1) errors
# ============================================
np.random.seed(42)
n = 100
phi_true = 0.7  # AR(1) coefficient

X_reduced = np.column_stack([np.ones(n), np.random.randn(n)])
beta_true = np.array([1.0, 2.0])

# Generate AR(1) errors
errors = np.random.randn(n)

# Add outliers at specific points
outlier_indices = [10, 25, 50, 75, 90]
errors[outlier_indices] = 10.0  # Large outliers!

y = X_reduced @ beta_true + errors

X_full = np.column_stack([X_reduced, np.random.randn(n)])

# ============================================
# Fit both models with IRLS
# ============================================
beta_r, w_r, rss_r_raw, rss_r_weighted = irls_fit(y, X_reduced)
beta_f, w_f, rss_f_raw, rss_f_weighted = irls_fit(y, X_full)

print("="*60)
print("ROBUST REGRESSION (IRLS) COMPARISON")
print("="*60)

print(f"\nReduced model:")
print(f"  RSS (raw, unweighted):    {rss_r_raw:.4f}")
print(f"  RSS (weighted):           {rss_r_weighted:.4f}")
print(f"  Weights range: [{w_r.min():.3f}, {w_r.max():.3f}]")

print(f"\nFull model:")
print(f"  RSS (raw, unweighted):    {rss_f_raw:.4f}")
print(f"  RSS (weighted):           {rss_f_weighted:.4f}")
print(f"  Weights range: [{w_f.min():.3f}, {w_f.max():.3f}]")

print(f"\n--- Key Comparison ---")
print(f"Raw RSS:      Full {'<' if rss_f_raw < rss_r_raw else '>='} Reduced?  {rss_f_raw < rss_r_raw}")
print(f"Weighted RSS: Full {'<' if rss_f_weighted < rss_r_weighted else '>='} Reduced?  {rss_f_weighted < rss_r_weighted}")

# Show that weights are DIFFERENT between models
weight_difference = np.max(np.abs(w_r - w_f))
print(f"\nMax weight difference between models: {weight_difference:.6f}")
if weight_difference > 0.01:
    print("⚠️  Weights are DIFFERENT between models!")
    print("    This is why the nesting guarantee breaks")


#%% Use my data
subj_id = 723
model_type='full'
# load HbO
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d = results['geo3d']
cfg_GLM['geo3d'] = geo3d

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

# assign DM
if model_type=='full':
    # Combine EEG DM with Reduced DM to get full model
    dm_all = model.combine_dm(eeg_dm, reduced_dm)
elif model_type=='reduced':
    dm_all = reduced_dm
else:
    dm_all = basis_dm

#%% extract one channel
select_ch = Y_all.channel.values[87]
Y_ch = Y_all.sel(chromo='HbO',channel=select_ch)
Y_ch = Y_ch.pint.dequantify()
X_full = pd.DataFrame(
            dm_all.common.values[:,:,0], columns=dm_all.common.regressor.values
        )
X_reduced = pd.DataFrame(
            reduced_dm.common.values[:,:,0], columns=reduced_dm.common.regressor.values
        )

#%% Check without whitening, if RLM obey the RSS_full <= RSS_reduced criteria.
M=sm.robust.norms.HuberT()
rlm_model = sm.RLM(Y_ch, X_full, M=M)
params_1 = rlm_model.fit()
resid_1 = pd.Series(Y_ch - X_full @ params_1.params)
print(f'Full RSS = {np.sum(resid_1**2)}')

M=sm.robust.norms.HuberT()
rlm_model = sm.RLM(Y_ch, X_reduced, M=M)
params_2 = rlm_model.fit()
resid_2 = pd.Series(Y_ch - X_reduced @ params_2.params)
print(f'Reduced RSS = {np.sum(resid_2**2)}')

if np.sum(resid_1**2) <= np.sum(resid_2**2):
    print("Without whitening, Full RSS <= Reduced RSS!")

#%% Check my AR IRLS
autoReg_save = reduced_model_result['autoReg_dict'][select_ch]
params_full, autoReg = model.my_ar_irls_GLM(Y_ch, X_full, autoReg=autoReg_save)

y = Y_ch.copy()
x = X_reduced.copy()


# Apply the AR filter to the lhs and rhs of the model
yf = model.prewhiten_arp_lfilter(y, autoReg)
xf = model.prewhiten_design_matrix_lfilter(x, autoReg)
xf_full = model.prewhiten_design_matrix_lfilter(X_full.copy(), autoReg)

rlm_model = sm.RLM(yf, xf, M=M)
params_reduced = rlm_model.fit()

# resid in whiten space
resid_full = pd.Series(yf - xf_full @ params_full.params)
print(f'Full RSS = {np.sum(resid_full**2)}')
resid_reduced = pd.Series(yf - xf @ params_reduced.params)
print(f'Reduced RSS = {np.sum(resid_reduced**2)}')

if np.sum(resid_full**2) <= np.sum(resid_reduced**2):
    print("With whitening, Full RSS <= Reduced RSS!")

#%% 
resid_full_sm = params_full.resid
resid_reduced_sm = params_reduced.resid
# resid in whiten space
if np.sum(resid_full_sm**2) <= np.sum(resid_reduced_sm**2):
    print("With whitening, Full RSS <= Reduced RSS!")

#%% AutoReg doesn't match
filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
# load full model
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
    full_model_result = pickle.load(f)
# load stim only results
with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
    reduced_model_result = pickle.load(f)
print("Full >>")
print(full_model_result['autoReg_dict'][select_ch])
print("Reduced >>")
print(reduced_model_result['autoReg_dict'][select_ch])
print("My single channel >>")
print(autoReg)

#%% Using the saved betas for residual. still different.
resid_mine = Y_ch - X_full @ full_model_result['betas'].sel(chromo='HbO',channel=select_ch)
resid_save = full_model_result['resid'].sel(chromo='HbO',channel=select_ch)
