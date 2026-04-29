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

subj_id_array = [670, 695, 721, 723, 726, 730]
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
# subj_id_array = [670, 695, 721, 723]
# subj_stat_list = []
# for subj_id in subj_id_array:
#     filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"

#     # load full model
#     with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full.pkl"), 'rb') as f:
#         full_model_result = pickle.load(f)
#     # load stim only results
#     with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
#         reduced_model_result = pickle.load(f)

#     # For each channel, run a F-test
#     subj_stat = dict()
#     ch_f_stat = np.zeros(full_model_result['resid'].shape[0])
#     ch_p_val = np.zeros(ch_f_stat.shape)
#     for ch_i in range(full_model_result['resid'].shape[0]):
#         # Get residual (compare HbO only)
#         resid_full = full_model_result['resid'][ch_i,0,:].values
#         resid_reduced = reduced_model_result['resid'][ch_i,0,:].values

#         # Calculate RSS from residuals
#         rss_reduced = np.sum(resid_reduced**2)
#         rss_full = np.sum(resid_full**2)

#         # Also need to know:
#         # full: 63, stim-only: 39, drift_ss:15, stim-only_correct_trial_type: 
#         n_observations = len(resid_full)  # number of observations (same for both)
#         p_reduced = reduced_model_result['betas'].shape[-1]  # number of predictors in reduced model
#         p_full = full_model_result['betas'].shape[-1] # number of predictors in full model

#         # Calculate degrees of freedom
#         df_resid_reduced = n_observations - p_reduced - 1  # subtract 1 for intercept
#         df_resid_full = n_observations - p_full - 1
#         df_diff = df_resid_reduced - df_resid_full  # = p_full - p_reduced

#         # Calculate F-statistic
#         f_stat = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_resid_full)

#         # Calculate p-value
#         p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid_full)

#         # save results
#         ch_f_stat[ch_i] = f_stat
#         ch_p_val[ch_i] = p_value

#     # correct p-values using FDR
#     rejected, p_values_fdr = fdrcorrection(ch_p_val, alpha=0.05)

#     # save to subject stats results
#     subj_stat['f_stat'] = ch_f_stat
#     subj_stat['p_value'] = ch_p_val
#     subj_stat['p_value_fdr'] = p_values_fdr

#     subj_stat_list.append(subj_stat)


# # print(f"RSS (reduced): {rss_reduced:.4f}")
# # print(f"RSS (full): {rss_full:.4f}")
# # print(f"F-statistic: {f_stat:.4f}")
# # print(f"df: ({df_diff}, {df_resid_full})")
# # print(f"p-value: {p_value:.6f}")

# # # Interpretation
# # if p_value < 0.05:
# #     print("✓ Additional regressors significantly improve the model")
# # else:
# #     print("✗ Additional regressors do NOT significantly improve the model")

# # #%% Calculate Log-likelihood
# # # Use this if your AR-IRLS provides beta, ar_coefs, and sigma2
# # ll_reduced = model.calculate_ar_loglikelihood(
# #     y, X_reduced, 
# #     beta_reduced, ar_coefs_reduced, sigma2_reduced
# # )
# # ll_full = model.calculate_ar_loglikelihood(
# #     y, X_full,
# #     beta_full, ar_coefs_full, sigma2_full
# # )

# #%% visualize channel significance ratio for each subject
# plt.figure()
# ratios = [np.sum(x['p_value_fdr']<=0.05)/len(x['p_value_fdr'])*100 for x in subj_stat_list]
# bars = plt.bar(np.arange(len(subj_stat_list)), ratios)

# # Add text labels on top of bars
# for i, (bar, ratio) in enumerate(zip(bars, ratios)):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
#              f'{ratio:.2f}%',  # Format to 3 decimal places
#              ha='center', va='bottom', fontsize=14)

# plt.xlabel("Subject ID", fontsize=14)
# plt.ylabel("Proportion of significant channels (FDR p ≤ 0.05)", fontsize=12)
# plt.xticks(np.arange(len(subj_stat_list)), subj_id_array, ha='center')
# plt.ylim([0,100])
# plt.grid()
# plt.tight_layout()

# #%% visualize which channels are significant for each subject
# f, axs = plt.subplots(2, 2, figsize=(10,8))
# axs = axs.flatten()
# for subj_i in range(len(subj_stat_list)):
#     plt_p_value = subj_stat_list[subj_i]['p_value_fdr'].copy()
#     plt_p_value = np.where(plt_p_value <= 0.05, 1, np.nan)
#     scalp_plot(
#         all_runs[0]['conc_o'],
#         geo3d_695,
#         plt_p_value,
#         ax = axs[subj_i],
#         cmap='RdBu_r',
#         vmin=0,
#         vmax=1,
#         optode_labels=False,
#         optode_size=6,
#         title=f"Sub-{subj_id_array[subj_i]}"
#     )
# plt.tight_layout()

# #%% Check if Residuals in Full is always <= Reduced model in all the channels
# check_chs = []
# for ch_i in range(full_model_result['resid'].shape[0]):
#     # Get residual (compare HbO only)
#     resid_full = full_model_result['resid'][ch_i,0,:].values
#     resid_reduced = reduced_model_result['resid'][ch_i,0,:].values
#     # Calculate RSS from residuals
#     rss_reduced = np.sum(resid_reduced**2)
#     rss_full = np.sum(resid_full**2)
#     # check if rss_full is always <= rss_reduced
#     if not rss_full<=rss_reduced:
#         check_chs.append(ch_i)

# # For each channel showing the problem
# for ch_i in check_chs:
#     # Get residual (compare HbO only)
#     resid_full = full_model_result['resid'][ch_i,0,:].values
#     resid_reduced = reduced_model_result['resid'][ch_i,0,:].values
#     # Calculate RSS from residuals
#     rss_reduced = np.sum(resid_reduced**2)
#     rss_full = np.sum(resid_full**2)
    
#     difference = rss_reduced - rss_full
#     relative_diff = difference / rss_reduced
    
#     print(f"\nChannel {ch_i}:")
#     print(f"  RSS reduced: {rss_reduced:.15e}")
#     print(f"  RSS full:    {rss_full:.15e}")
#     print(f"  Difference:  {difference:.15e}")
#     print(f"  Relative:    {relative_diff:.10e} ({relative_diff*100:.8f}%)")
    
#     # Check if within floating point precision
#     if np.abs(relative_diff) < 1e-10:
#         print(f"  → Likely numerical precision error (< 1e-10 relative)")
#     elif np.abs(relative_diff) < 1e-8:
#         print(f"  → Possibly numerical precision error (< 1e-8 relative)")
#     else:
#         print(f"  → Real problem (not just precision error)")

#%% f test from model results
sig_list = []
model_type = 'full'
model_cmp = 'f_test_full_stim'

# fig, axs = plt.subplots(3,2,figsize=(10,8))
# axs = axs.flatten()

for s_i, subj_id in enumerate(subj_id_array):
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)

    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_{model_type}_noEEG_rejected.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load reduced model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        reduced_model_result = pickle.load(f)
    # f_score_full_reduced = model.extract_val_across_channels(full_model_result['f_test'], chromo='HbO', stat_val='F')
    p_val_full_reduced = model.extract_val_across_channels(full_model_result[model_cmp],
                                                           chromo='HbO', stat_val='p')
    # remove bad channels from analysis
    p_val_full_reduced = p_val_full_reduced[clean_chs_idx]
    # correct p-values using FDR
    rejected, p_values_fdr = fdrcorrection(p_val_full_reduced, alpha=0.05)
    sig_list.append(np.sum(rejected)/len(rejected))

    # RSS
    rss_all = np.sum(full_model_result['resid'].sel(chromo='HbO').values**2,axis=1)
    rss_reduced = np.sum(reduced_model_result['resid'].sel(chromo='HbO').values**2,axis=1)
    rss_ratio = np.log(rss_reduced)- np.log(rss_all)
    rss_ratio[bad_indices] = np.nan

    # visualize RSS scalp plot (log scale)
    # model.scalp_plot(
    #     all_runs[0]['conc_o'],
    #     geo3d_695,
    #     rss_ratio,
    #     ax = axs[s_i],
    #     cmap='RdBu_r',
    #     vmin=-1.5,        
    #     vmax=1.5,
    #     optode_labels=False,
    #     optode_size=6,
    # )

#
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

#%% Check if all F are larger in Full vs Basis than in Reduced vs Basis
for s_i, subj_id in enumerate(subj_id_array):
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)

    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full_noEEG_rejected.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load reduced model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        reduced_model_result = pickle.load(f)
    # f_score_full_reduced = model.extract_val_across_channels(full_model_result['f_test'], chromo='HbO', stat_val='F')
    f_full_basis = model.extract_val_across_channels(full_model_result['f_test_full_basis'],
                                                           chromo='HbO', stat_val='F')
    f_full_stim= model.extract_val_across_channels(full_model_result['f_test_full_stim'],
                                                           chromo='HbO', stat_val='F')                                                           
    f_reduced_basis = model.extract_val_across_channels(reduced_model_result['f_test_stim_basis'],
                                                           chromo='HbO', stat_val='F')
    # remove bad channels from analysis
    check_f = np.all(f_full_basis[clean_chs_idx]>=f_full_stim[clean_chs_idx])
    print(f"sub-{subj_id}: All F in Full vs. Basis >= in Full vs. Reduced: {check_f}")
    check_f = np.all(f_full_basis[clean_chs_idx]>=f_reduced_basis[clean_chs_idx])
    print(f"sub-{subj_id}: All F in Full vs. Basis >= in Reduced vs. Basis: {check_f}")

#%% R-squared between stim-only vs EEG-only models
stim_r_list = []
eeg_r_list = []
stim_r_z_list = []
eeg_r_z_list = []
stim_r_z_squared_list = []
eeg_r_z_squared_list = []
stim_r_squared_logit_list = []
eeg_r_squared_logit_list = []
clean_chs_list = []

for s_i, subj_id in enumerate(subj_id_array):
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load eeg model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_onlyEEG.pkl"), 'rb') as f:
        eeg_model_result = pickle.load(f)
        betas_eeg = eeg_model_result['betas']
    # load stim model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_reduced.pkl"), 'rb') as f:
        stim_model_result = pickle.load(f)
        betas_stim = stim_model_result['betas']
    # load clean channels
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
    clean_chs_list.append(clean_chs_idx)
    # load and DM
    with open(os.path.join(filepath,'dm_dict.pkl'),'rb') as f:
        dm_dict = pickle.load(f)
        eeg_dm = dm_dict['onlyEEG']
        stim_dm = dm_dict['onlyStim']
        Y_all = dm_dict['Y_all']
    # estimate HbO
    eeg_HbO_estimate = model.xr.dot(betas_eeg, eeg_dm.common, dims='regressor')
    stim_HbO_estimate = model.xr.dot(betas_stim, stim_dm.common, dims='regressor')
    # calculate correlation coefficient
    eeg_r = model.xr.corr(eeg_HbO_estimate, Y_all, dim='time')
    stim_r = model.xr.corr(stim_HbO_estimate, Y_all, dim='time')
    eeg_r_list.append(eeg_r)
    stim_r_list.append(stim_r)
    # Fisher z-transform correlation coefficient
    eeg_r_z = np.arctanh(eeg_r)
    stim_r_z = np.arctanh(stim_r)
    eeg_r_z_list.append(eeg_r_z)
    stim_r_z_list.append(stim_r_z)
    # R-squared as square of z-transformed correlation coefficient (is this valid?)
    eeg_r_z_squared = eeg_r_z **2
    stim_r_z_squared = stim_r_z **2
    eeg_r_z_squared_list.append(eeg_r_z_squared)
    stim_r_z_squared_list.append(stim_r_z_squared)
    # logit transform R-squared
    eeg_r_squared = eeg_r **2
    stim_r_squared = stim_r **2
    eeg_r_squared_logit = model.logit_transform(eeg_r_squared)
    stim_r_squared_logit = model.logit_transform(stim_r_squared)
    eeg_r_squared_logit_list.append(eeg_r_squared_logit)
    stim_r_squared_logit_list.append(stim_r_squared_logit)

eeg_r_squared_z_list = [np.arctanh(x**2) for x in eeg_r_list]
stim_r_squared_z_list = [np.arctanh(x**2) for x in stim_r_list]

"""
TODO: Comparison of R-squared between two models will be very difficult:
    1. The distribution is not normal.
    2. Our data is autocorrelated.
    If we want to logit transform R-squared to approximate normal distribution, our samples need to be
    independent to do the transform. When we have two AR-IRLS models, it is difficult to decide which 
    whiten-space to be in.

    For now, I'll run pairwise ttest.
"""

#%% Pairwise ttest for each subject
def pairwise_ttest_per_subj(eeg_r_list, stim_r_list, clean_chs_list):
    t_stat_list = []
    p_val_list = []
    is_higher_list = []
    for eeg_r, stim_r, clean_chs in zip(eeg_r_list, stim_r_list, clean_chs_list):
        # get only clean channels
        eeg_r_clean = eeg_r.sel(chromo='HbO').values[clean_chs]
        stim_r_clean = stim_r.sel(chromo='HbO').values[clean_chs]
        # pairwise ttest
        t_stat, p_val = stats.ttest_rel(eeg_r_clean, stim_r_clean)
        t_stat_list.append(t_stat)
        p_val_list.append(p_val)
        is_higher_list.append(np.mean(eeg_r_clean)>np.mean(stim_r_clean))
    return t_stat_list, p_val_list, is_higher_list

#%%
t_stat_list, p_val_list, is_higher_list = pairwise_ttest_per_subj(eeg_r_squared_z_list,
                                                                stim_r_squared_z_list,
                                                                clean_chs_list)
for s_i, subj_id in enumerate(subj_id_array):
    if p_val_list[s_i]<=0.05:
        print(f"sub-{subj_id}: R-squared (EEG) is significantly {'higher' if is_higher_list[s_i] else 'lower'}.")
    else:
        print(f"sub-{subj_id}: No siginificance found between two models.")
    
#%% contrast ttest
t_eeg_sig_list = []
t_stim_sig_list = []
t_eeg_stim_sig_list = []
for subj_id in subj_id_array:
    filepath = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg/sub-{subj_id}"
    # load full model
    with open(os.path.join(filepath,f"sub-{subj_id}_glm_mnt_full_noEEG_rejected_ttest.pkl"), 'rb') as f:
        full_model_result = pickle.load(f)
    # load clean chs
    fnirs_result_path = f"/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/cedalion/processed_data/sub-{subj_id}"
    with gzip.open(os.path.join(fnirs_result_path, f"sub-{subj_id}_conc_o_hrf_estimates_ar_irls.pkl.gz"), 'rb') as f:
        results = pickle.load(f)
        hrf_per_subj = results['hrf_per_subj']
        hrf_mse_per_subj = results['hrf_mse_per_subj']
        bad_indices = results['bad_indices']
    clean_chs_idx = np.arange(len(hrf_per_subj.channel.values))
    clean_chs_idx = np.delete(clean_chs_idx,bad_indices)
    t_0_eeg = model.extract_val_across_channels(full_model_result['t_test_0_eeg'],
                                                            chromo='HbO', stat_val='P>|z|', is_table=True)
    t_0_stim = model.extract_val_across_channels(full_model_result['t_test_0_stim'],
                                                            chromo='HbO', stat_val='P>|z|', is_table=True)
    t_0_eeg_stim = model.extract_val_across_channels(full_model_result['t_test_0_eeg_stim'],
                                                            chromo='HbO', stat_val='P>|z|', is_table=True)# remove bad channels from analysis
    # calculate proportion of channels show significant
    t_eeg_sig = np.sum(t_0_eeg[clean_chs_idx]<=0.05)/len(clean_chs_idx)
    t_stim_sig = np.sum(t_0_stim[clean_chs_idx]<=0.05)/len(clean_chs_idx)
    t_eeg_stim_sig = np.sum(t_0_eeg_stim[clean_chs_idx]<=0.05)/len(clean_chs_idx)

    t_eeg_sig_list.append(t_eeg_sig)
    t_stim_sig_list.append(t_stim_sig)
    t_eeg_stim_sig_list.append(t_eeg_stim_sig)

#%%
#
fig, axes = plt.subplots(3,1, figsize=(5, 10), sharey=True)
plot_data = [
    (t_eeg_sig_list, "EEG regressor (t-test)"),
    (t_stim_sig_list, "Stim regressor (t-test)"),
    (t_eeg_stim_sig_list, "EEG+Stim regressor (t-test)"),
]

for ax, (sig_list_data, title) in zip(axes, plot_data):
    ratios = np.array(sig_list_data) * 100
    bars = ax.bar(np.arange(len(subj_id_array)), ratios)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{ratio:.2f}%',
                ha='center', va='bottom', fontsize=14)
    ax.set_xlabel("Subject ID", fontsize=14)
    ax.set_ylabel("Proportion of significant channels (p ≤ 0.05)", fontsize=12)
    ax.set_xticks(np.arange(len(subj_id_array)))
    ax.set_xticklabels(subj_id_array, ha='center')
    ax.set_ylim([0, 100])
    ax.set_title(title, fontsize=13)
    ax.grid()

plt.tight_layout()
# %%
