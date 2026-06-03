#%% Load libraries
import numpy as np
import scipy as sp
import scipy.signal
import scipy.stats
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import mne
import statsmodels.api as sm

git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from params_setting import project_path, ch_names
from model import my_ar_irls_GLM

#%% Parameters
EEG_DERIV_DIR = os.path.join(project_path, 'derivatives', 'eeg')
SUBJECTS      = [670, 673, 719, 721, 723, 726, 727, 730, 733, 746, 751]

FREQ_BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  12),
    'beta':  (12, 30),
}

POWER_TMIN   = 0.0   # seconds post-stimulus
POWER_TMAX   = 0.8   # seconds post-stimulus (city_correct trial window)
PMAX         = 30    # max AR order for BIC selection in my_ar_irls_GLM
SELECTED_CHS = ['fz','cz','pz','oz']  # channels used for band-power extraction

#%% Helper: compute mean band power for one epoch (1D signal in µV)
def bandpower(signal, sfreq, fmin, fmax):
    nperseg = min(len(signal), int(sfreq))
    freqs, psd = sp.signal.welch(signal, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapezoid(psd[mask], freqs[mask])


#%% Load data: city_correct trials only, extract per-trial band power + VTC
records = []

for subj_id in SUBJECTS:
    subj_key     = f'sub-{subj_id}'
    subj_eeg_dir = os.path.join(EEG_DERIV_DIR, subj_key)
    print(f'Loading {subj_key} ...')

    for run_id in range(1, 4):
        fif_candidates = [
            os.path.join(subj_eeg_dir,
                f'{subj_key}_task-gradCPT_run-{run_id:02d}_preproc_eeg.fif'),
            os.path.join(subj_eeg_dir,
                f'{subj_key}_task-GradCPT_run-{run_id:02d}_preproc_eeg.fif'),
        ]
        fif_file = next((f for f in fif_candidates if os.path.isfile(f)), None)
        ev_file  = os.path.join(subj_eeg_dir,
            f'{subj_key}_task-gradCPT_run-{run_id:02d}_events.tsv')

        if fif_file is None or not os.path.isfile(ev_file):
            continue

        raw   = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        ev_df = pd.read_csv(ev_file, sep='\t')

        # city_correct: trial_type == 'city' and response_code != 0
        sel_df = ev_df[
            (ev_df['trial_type'] == 'city') & (ev_df['response_code'] != 0)
        ].reset_index(drop=True)
        if len(sel_df) == 0:
            continue

        avail_chs = [c for c in SELECTED_CHS if c in raw.ch_names]
        if not avail_chs:
            continue
        
        eeg_data = raw.copy().pick(avail_chs).get_data() * 1e6  # V → µV
        # shape: (n_ch, n_times)

        i_start_offset = int(POWER_TMIN * sfreq)
        i_end_offset   = int(POWER_TMAX * sfreq)

        for _, row in sel_df.iterrows():
            i_on    = int(float(row['onset']) * sfreq)
            i_start = i_on + i_start_offset
            i_end   = i_on + i_end_offset
            if i_start < 0 or i_end > eeg_data.shape[1]:
                continue

            vtc = float(row['VTC_smoothed']) if 'VTC_smoothed' in row else float(row['VTC'])

            trial_rec = dict(subj_id=subj_id, run_id=run_id, vtc=vtc)
            for band_name, (flo, fhi) in FREQ_BANDS.items():
                powers = [bandpower(eeg_data[ci, i_start:i_end], sfreq, flo, fhi)
                          for ci in range(len(avail_chs))]
                trial_rec[f'power_{band_name}'] = float(np.mean(powers))
            records.append(trial_rec)

df = pd.DataFrame(records)
print(f'\nTotal city_correct trials: {len(df)}, subjects: {df["subj_id"].nunique()}')


#%% Build design matrix and run my_ar_irls_GLM per subject per band
# Per-subject model: y = intercept + VTC + ε  (intercept absorbs subject mean power)
# VTC is z-scored within each subject so betas are comparable across subjects.

per_subj_betas  = {band: [] for band in FREQ_BANDS}
per_subj_ids    = []

for subj_id in sorted(df['subj_id'].unique()):
    df_s = df[df['subj_id'] == subj_id].reset_index(drop=True)
    if len(df_s) < 10:
        continue

    # Design matrix as pd.DataFrame (required by my_ar_irls_GLM)
    X = pd.DataFrame({'intercept': np.ones(len(df_s)), 'VTC': df_s['vtc'].values})

    per_subj_ids.append(subj_id)
    for band_name in FREQ_BANDS:
        y = pd.Series(np.log10(df_s[f'power_{band_name}'].values + 1e-30))
        result, arcoef = my_ar_irls_GLM(y, X, pmax=PMAX,
                                        M=sm.robust.norms.HuberT())
        per_subj_betas[band_name].append(result.params['VTC'])

    print(f'  {subj_id}: alpha VTC β = {per_subj_betas["alpha"][-1]:.4f}')


#%% Group-level inference: one-sample t-test (H0: VTC β = 0) per band
print('\n--- Group-level one-sample t-test (H0: VTC β = 0) ---')
group_stats = {}
for band_name, betas in per_subj_betas.items():
    betas_arr          = np.array(betas)
    t_stat, p_val      = sp.stats.ttest_1samp(betas_arr, 0)
    group_stats[band_name] = dict(
        betas=betas_arr,
        mean=betas_arr.mean(),
        sem=betas_arr.std() / np.sqrt(len(betas_arr)),
        t=t_stat, p=p_val,
    )
    print(f'  {band_name:>6}: mean β={betas_arr.mean():.4f}  '
          f't({len(betas_arr)-1})={t_stat:.3f}  p={p_val:.4f}  N={len(betas_arr)}')


#%% Plot: VTC betas per band (bar + individual subject points)
bands  = list(FREQ_BANDS.keys())
colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
n_subj = len(per_subj_ids)

fig, ax = plt.subplots(figsize=(8, 4))
for i, (band_name, color) in enumerate(zip(bands, colors)):
    gs    = group_stats[band_name]
    betas = gs['betas']
    ax.bar(i, gs['mean'], color=color, alpha=0.7,
           yerr=gs['sem'], capsize=4, error_kw=dict(linewidth=1.5))
    ax.scatter(
        np.full(n_subj, i) + np.random.uniform(-0.15, 0.15, n_subj),
        betas, color=color, s=30, zorder=3, alpha=0.8,
    )
    sig = '*' if gs['p'] < 0.05 else ('~' if gs['p'] < 0.1 else 'ns')
    ax.text(i, gs['mean'] + gs['sem'] + 0.002, sig,
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(range(len(bands)))
ax.set_xticklabels(
    [f'{b}\n({FREQ_BANDS[b][0]}–{FREQ_BANDS[b][1]} Hz)' for b in bands])
ax.set_ylabel('VTC β (log₁₀ power per z-VTC)')
ax.set_title(
    f'AR-IRLS (my_ar_irls_GLM): VTC effect on EEG band power\n'
    f'city_correct trials  (N={n_subj} subjects)  (* p<0.05, ~ p<0.1)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot: band power violin plot across 10 VTC bins
N_BINS   = 10
vtc_vals = df['vtc'].values
bin_edges   = np.percentile(vtc_vals, np.linspace(0, 100, N_BINS + 1))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_ids     = np.digitize(vtc_vals, bin_edges[1:-1])  # 0 … N_BINS-1

fig, axes = plt.subplots(len(FREQ_BANDS), 1, figsize=(10, 3 * len(FREQ_BANDS)),
                         sharex=True)
for ax, (band_name, color) in zip(axes, zip(FREQ_BANDS.keys(), colors)):
    log_power = np.log10(df[f'power_{band_name}'].values + 1e-30)
    bin_data  = [log_power[bin_ids == b] for b in range(N_BINS)]

    parts = ax.violinplot(bin_data, positions=bin_centers,
                          widths=(bin_edges[1] - bin_edges[0]) * 0.8,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    parts['cmedians'].set_color('k')
    parts['cmedians'].set_linewidth(1.5)

    ax.set_ylabel(f'log₁₀ power\n({band_name})', fontsize=9)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('VTC (10 percentile bins, low→high)', fontsize=10)
fig.suptitle(
    f'EEG band power across 10 VTC bins — violin plot\n'
    f'city_correct trials  (N={len(df)} trials, channels: {SELECTED_CHS})',
    fontsize=11)
plt.tight_layout()
plt.show()

#%% Plot: PSD of the VTC signal (city_correct trials) with EEG band range markers
# VTC is sampled at trial rate (~1 trial / 0.8 s ≈ 1.25 Hz).
# Compute Welch PSD per subject-run, then average across runs/subjects.

band_colors = {
    'delta': 'steelblue',
    'theta': 'darkorange',
    'alpha': 'forestgreen',
    'beta':  'crimson',
}

vtc_psd_list  = []
vtc_psd_freqs = None

for subj_id in SUBJECTS:
    subj_key     = f'sub-{subj_id}'
    subj_eeg_dir = os.path.join(EEG_DERIV_DIR, subj_key)

    for run_id in range(1, 4):
        ev_file = os.path.join(subj_eeg_dir,
            f'{subj_key}_task-gradCPT_run-{run_id:02d}_events.tsv')
        if not os.path.isfile(ev_file):
            continue

        ev_df  = pd.read_csv(ev_file, sep='\t')
        sel_df = ev_df[
            (ev_df['trial_type'] == 'city') & (ev_df['response_code'] != 0)
        ].reset_index(drop=True)
        if len(sel_df) < 4:
            continue

        vtc_col = 'VTC_smoothed' if 'VTC_smoothed' in sel_df.columns else 'VTC'
        vtc_signal = sel_df[vtc_col].values

        # trial-rate sampling frequency: 1 / mean inter-trial interval (s)
        trial_fs = 1.0 / np.median(np.diff(sel_df['onset'].values))

        nperseg = min(len(vtc_signal), max(4, len(vtc_signal) // 4))
        freqs, psd = sp.signal.welch(vtc_signal, fs=trial_fs, nperseg=nperseg)

        vtc_psd_list.append(psd)
        if vtc_psd_freqs is None:
            vtc_psd_freqs = freqs

# Trim all PSDs to the shortest length (different runs may differ slightly)
min_len       = min(len(p) for p in vtc_psd_list)
vtc_psd_arr   = np.array([p[:min_len] for p in vtc_psd_list])
vtc_psd_freqs = vtc_psd_freqs[:min_len]

mean_vtc_psd = vtc_psd_arr.mean(axis=0)
sem_vtc_psd  = vtc_psd_arr.std(axis=0) / np.sqrt(len(vtc_psd_arr))

fig, ax = plt.subplots(figsize=(8, 4))

ax.semilogy(vtc_psd_freqs, mean_vtc_psd, color='k', linewidth=1.8,
            label=f'Mean VTC PSD (N={len(vtc_psd_arr)} runs)')
ax.fill_between(vtc_psd_freqs,
                mean_vtc_psd - sem_vtc_psd,
                mean_vtc_psd + sem_vtc_psd,
                color='k', alpha=0.15, label='±1 SEM')

# Dashed vertical lines at EEG band boundaries
for band_name, (flo, fhi) in FREQ_BANDS.items():
    color = band_colors[band_name]
    for f_edge in (flo, fhi):
        if f_edge <= vtc_psd_freqs[-1]:
            ax.axvline(f_edge, color=color, linestyle='--', linewidth=1.2, alpha=0.8,
                       label=f'{band_name} boundary ({f_edge} Hz)' if f_edge == flo else None)

ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Power (a.u.²/Hz)', fontsize=11)
ax.set_xlim(0, vtc_psd_freqs[-1])
ax.set_title(
    'PSD of VTC signal — city_correct trials\n'
    'Dashed lines: EEG band boundaries (delta/theta/alpha/beta)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
