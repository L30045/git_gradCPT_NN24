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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# load template run and geo3d
subj_id = 695
hbo_file = os.path.join(project_path,f"derivatives/cedalion/processed_data/sub-{subj_id}/sub-{subj_id}_preprocessed_results_ar_irls.pkl")
with gzip.open(hbo_file, 'rb') as f:
    results = pickle.load(f)

all_runs = results['runs']
all_chs_pruned = results['chs_pruned']
all_stims = results['stims']
geo3d_695 = results['geo3d']

#%% For each subject, load residuals
subj_id_array = [670, 695, 721, 723]
subj_stat_list = []
for subj_id in subj_id_array:
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"

    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load stim only results
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        reduced_model_result = pickle.load(f)

    # For each channel, run a F-test
    subj_stat = dict()
    ch_f_stat = np.zeros(full_model_result['resid'].shape[0])
    ch_p_val = np.zeros(ch_f_stat.shape)
    for ch_i in range(full_model_result['resid'].shape[0]):
        # Get residual (compare HbO only)
        resid_full = full_model_result['resid'][ch_i,0,:].values
        resid_reduced = reduced_model_result['resid'][ch_i,0,:].values

        # Calculate RSS from residuals
        rss_reduced = np.sum(resid_reduced**2)
        rss_full = np.sum(resid_full**2)

        # Also need to know:
        # full: 63, stim-only: 39, drift_ss:15, stim-only_correct_trial_type: 
        n_observations = len(resid_full)  # number of observations (same for both)
        p_reduced = reduced_model_result['betas'].shape[-1]  # number of predictors in reduced model
        p_full = full_model_result['betas'].shape[-1] # number of predictors in full model

        # Calculate degrees of freedom
        df_resid_reduced = n_observations - p_reduced - 1  # subtract 1 for intercept
        df_resid_full = n_observations - p_full - 1
        df_diff = df_resid_reduced - df_resid_full  # = p_full - p_reduced

        # Calculate F-statistic
        f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_resid_full)

        # Calculate p-value
        p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid_full)

        # save results
        ch_f_stat[ch_i] = f_stat
        ch_p_val[ch_i] = p_value

    # correct p-values using FDR
    rejected, p_values_fdr = fdrcorrection(ch_p_val, alpha=0.05)

    # save to subject stats results
    subj_stat['f_stat'] = ch_f_stat
    subj_stat['p_value'] = ch_p_val
    subj_stat['p_value_fdr'] = p_values_fdr

    subj_stat_list.append(subj_stat)


# print(f"RSS (reduced): {rss_reduced:.4f}")
# print(f"RSS (full): {rss_full:.4f}")
# print(f"F-statistic: {f_stat:.4f}")
# print(f"df: ({df_diff}, {df_resid_full})")
# print(f"p-value: {p_value:.6f}")

# # Interpretation
# if p_value < 0.05:
#     print("✓ Additional regressors significantly improve the model")
# else:
#     print("✗ Additional regressors do NOT significantly improve the model")

# #%% Calculate Log-likelihood
# # Use this if your AR-IRLS provides beta, ar_coefs, and sigma2
# ll_reduced = model.calculate_ar_loglikelihood(
#     y, X_reduced, 
#     beta_reduced, ar_coefs_reduced, sigma2_reduced
# )
# ll_full = model.calculate_ar_loglikelihood(
#     y, X_full,
#     beta_full, ar_coefs_full, sigma2_full
# )

#%% visualize channel significance ratio for each subject
plt.figure()
ratios = [np.sum(x['p_value_fdr']<=0.05)/len(x['p_value_fdr'])*100 for x in subj_stat_list]
bars = plt.bar(np.arange(len(subj_stat_list)), ratios)

# Add text labels on top of bars
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{ratio:.2f}%',  # Format to 3 decimal places
             ha='center', va='bottom', fontsize=14)

plt.xlabel("Subject ID", fontsize=14)
plt.ylabel("Proportion of significant channels (FDR p ≤ 0.05)", fontsize=12)
plt.xticks(np.arange(len(subj_stat_list)), subj_id_array, ha='center')
plt.ylim([0,100])
plt.grid()
plt.tight_layout()

#%% visualize which channels are significant for each subject
f, axs = plt.subplots(2, 2, figsize=(10,8))
axs = axs.flatten()
for subj_i in range(len(subj_stat_list)):
    plt_p_value = subj_stat_list[subj_i]['p_value_fdr'].copy()
    plt_p_value = np.where(plt_p_value <= 0.05, 1, np.nan)
    scalp_plot(
        all_runs[0]['conc_o'],
        geo3d_695,
        plt_p_value,
        ax = axs[subj_i],
        cmap='RdBu_r',
        vmin=0,
        vmax=1,
        optode_labels=False,
        optode_size=6,
        title=f"Sub-{subj_id_array[subj_i]}"
    )
plt.tight_layout()

#%% Check if Residuals in Full is always <= Reduced model in all the channels
check_chs = []
for ch_i in range(full_model_result['resid'].shape[0]):
    # Get residual (compare HbO only)
    resid_full = full_model_result['resid'][ch_i,0,:].values
    resid_reduced = reduced_model_result['resid'][ch_i,0,:].values
    # Calculate RSS from residuals
    rss_reduced = np.sum(resid_reduced**2)
    rss_full = np.sum(resid_full**2)
    # check if rss_full is always <= rss_reduced
    if not rss_full<=rss_reduced:
        check_chs.append(ch_i)

# For each channel showing the problem
for ch in check_chs:
    # Get residual (compare HbO only)
    resid_full = full_model_result['resid'][ch_i,0,:].values
    resid_reduced = reduced_model_result['resid'][ch_i,0,:].values
    # Calculate RSS from residuals
    rss_reduced = np.sum(resid_reduced**2)
    rss_full = np.sum(resid_full**2)
    
    difference = rss_reduced - rss_full
    relative_diff = difference / rss_reduced
    
    print(f"\nChannel {ch}:")
    print(f"  RSS reduced: {rss_reduced:.15e}")
    print(f"  RSS full:    {rss_full:.15e}")
    print(f"  Difference:  {difference:.15e}")
    print(f"  Relative:    {relative_diff:.10e} ({relative_diff*100:.8f}%)")
    
    # Check if within floating point precision
    if np.abs(relative_diff) < 1e-10:
        print(f"  → Likely numerical precision error (< 1e-10 relative)")
    elif np.abs(relative_diff) < 1e-8:
        print(f"  → Possibly numerical precision error (< 1e-8 relative)")
    else:
        print(f"  → Real problem (not just precision error)")


#%% f test from model results
subj_id_array = [670, 695, 721, 723]
sig_list = []

for subj_id in subj_id_array:
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # f_score_full_reduced = model.extract_val_across_channels(full_model_result['f_test'], chromo='HbO', stat_val='F')
    p_val_full_reduced = model.extract_val_across_channels(full_model_result['f_test'], chromo='HbO', stat_val='p')
    sig_list.append(np.sum(p_val_full_reduced<=0.05)/len(p_val_full_reduced))

#%%
plt.figure()
ratios = np.array(sig_list)*100
bars = plt.bar(np.arange(len(subj_id_array)), ratios)

# Add text labels on top of bars
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{ratio:.2f}%',  # Format to 3 decimal places
             ha='center', va='bottom', fontsize=14)

plt.xlabel("Subject ID", fontsize=14)
plt.ylabel("Proportion of significant channels (FDR p ≤ 0.05)", fontsize=12)
plt.xticks(np.arange(len(subj_id_array)), subj_id_array, ha='center')
plt.ylim([0,100])
plt.grid()
plt.tight_layout()

