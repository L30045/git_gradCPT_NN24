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

