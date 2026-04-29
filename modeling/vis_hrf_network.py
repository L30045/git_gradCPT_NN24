#%% load library
import numpy as np
import pickle
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys

git_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(os.path.join(git_path, 'preproc_pipe'))
from params_setting import project_path

#%% settings
subj_id_array = [670, 695, 721, 723, 726, 730]
model_type = 'full_noEEG_rejected_ttest'

#%% load HRF estimates per subject
hrf_by_subj = {}
for subj_id in subj_id_array:
    file_path = os.path.join(
        project_path, 'derivatives', 'eeg', f"sub-{subj_id}",
        f"sub-{subj_id}_glm_mnt_{model_type}.pkl"
    )
    if not os.path.isfile(file_path):
        print(f"Missing: {file_path}")
        continue
    with open(file_path, 'rb') as f:
        result = pickle.load(f)
    if 'hrf_estimate' not in result:
        print(f"No hrf_estimate in result for sub-{subj_id}")
        continue
    hrf = result['hrf_estimate']
    try:
        hrf = hrf.pint.dequantify()
    except Exception:
        pass
    hrf_by_subj[subj_id] = hrf
    print(f"Loaded sub-{subj_id}: dims={hrf.dims}, shape={hrf.shape}")

if not hrf_by_subj:
    raise RuntimeError("No HRF estimates loaded.")

#%% inspect dimensions from the first subject
example_hrf = next(iter(hrf_by_subj.values()))
print("\nDimensions:", example_hrf.dims)
print("Coordinates:")
for k, v in example_hrf.coords.items():
    if v.ndim <= 1:
        print(f"  {k}: {list(v.values)}")

trial_types = list(example_hrf.trial_type.values)
time_axis   = example_hrf.time.values
networks    = list(example_hrf.channel.values)
has_chromo  = 'chromo' in example_hrf.dims
chromos     = list(example_hrf.chromo.values) if has_chromo else [None]

n_networks = len(networks)
chromo_colors = {'HbO': 'crimson', 'HbR': 'steelblue'}

#%% plot: for each chromo, two figures (first 9 networks / rest), both trial types per subplot
tt_colors = {'mnt-correct': 'forestgreen', 'mnt-incorrect': 'darkorange'}
network_batches = [networks[:9], networks[9:]]
batch_labels    = ['networks 1–9', f'networks 10–{len(networks)}']

for chromo in chromos:
    chromo_label = chromo if chromo is not None else ''

    for batch_nets, batch_label in zip(network_batches, batch_labels):
        if not batch_nets:
            continue

        n_batch = len(batch_nets)
        ncols   = min(3, n_batch)
        nrows   = int(np.ceil(n_batch / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
        axes_flat = np.array(axes).flatten() if n_batch > 1 else [axes]

        for ax, net in zip(axes_flat, batch_nets):
            for tt in trial_types:
                color = tt_colors.get(tt, 'gray')

                traces = []
                for subj_id, hrf in hrf_by_subj.items():
                    try:
                        sel = dict(trial_type=tt, channel=net)
                        if has_chromo:
                            sel['chromo'] = chromo
                        trace = hrf.sel(**sel).values.astype(float)
                        traces.append(trace)
                        ax.plot(time_axis, trace, color=color, linewidth=0.8, alpha=0.2)
                    except Exception as e:
                        print(f"sub-{subj_id} {net} {chromo} {tt}: {e}")

                if traces:
                    arr  = np.array(traces)
                    mean = arr.mean(axis=0)
                    sem  = sp.stats.sem(arr, axis=0) if len(arr) > 1 else np.zeros_like(mean)
                    ax.fill_between(time_axis, mean - sem, mean + sem, alpha=0.2, color=color)
                    ax.plot(time_axis, mean, color=color, linewidth=2,
                            label=f"{tt} (N={len(traces)})")

            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('HRF amplitude')
            ax.set_title(net)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[n_batch:]:
            ax.set_visible(False)

        title = f'HRF per network'
        if chromo_label:
            title += f'  |  {chromo_label}'
        title += f'  |  {batch_label}  |  mean ± SEM  (N={len(hrf_by_subj)} subjects)'
        fig.suptitle(title, fontsize=13)
        plt.tight_layout()

plt.show()

# %%
