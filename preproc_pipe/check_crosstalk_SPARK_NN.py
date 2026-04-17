#%% load library
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import mne
mne.viz.set_browser_backend("matplotlib")
import os
import copy
import pandas as pd
from tqdm import tqdm
import pickle
import glob
import time
import sys
# from spectral_connectivity import Multitaper, Connectivity
# from spectral_connectivity.transforms import prepare_time_series


filepath = os.path.abspath("/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/sourcedata/raw/sub-SPARK-testing-004")
eegpath = os.path.join(filepath,'eeg')
cptpath = os.path.join(filepath,'gradCPT')
# read replace opto location
opt_loc_csv = pd.read_csv(os.path.join(filepath, 'Replace_optodes_list.csv'))
nearest_1020 = opt_loc_csv['Nearest 10-20 system '].copy()
nearest_1020.iloc[4] = 'FCz'

# functions
def check_flat_channels(EEG):
    eeg_data = EEG.get_data(picks='eeg')
    eeg_chs = np.array([x["ch_name"] for x in EEG.info["chs"] if x["kind"]==2])
    flat_ch_idx = []
    # check flat channels
    for ch_i in range(eeg_data.shape[0]):
        if np.mean(eeg_data[ch_i]-np.mean(eeg_data[ch_i]))==0:
            print(f"Warning: flat channel detected. ({eeg_chs[ch_i]})")
            flat_ch_idx.append(eeg_chs[ch_i])
    return flat_ch_idx

#%% Get resting session from EEG and fNIRS
# load EEG and plot spectrum of EEG
eeg_csv = pd.read_csv(os.path.join(eegpath,"sub-004_task-Rest_run-01.csv.csv"), skiprows=12, index_col=False)
ch_cols = [c for c in eeg_csv.columns if c.strip().startswith('CH')]
eeg_csv = eeg_csv[['Time(s)', 'TRIGGER(DIGITAL)'] + ch_cols]
# truncate EEG by trigger: keep rows between first and last button press (trigger ~ 0)
trig = eeg_csv['TRIGGER(DIGITAL)']
pressed = trig[trig == 0].index
eeg_csv = eeg_csv.loc[pressed[0]:pressed[-1]].reset_index(drop=True)
# put eeg_csv into mne raw object
sfreq = np.round(np.median(1/np.diff(eeg_csv['Time(s)'])))  # Hz
ch_names = nearest_1020.tolist()
ch_types = ['eeg'] * len(ch_cols)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
data = eeg_csv[ch_cols].values.T * 1e-6  # convert uV to V
EEG = mne.io.RawArray(data, info)
# assign 10-20 system locations
montage = mne.channels.make_standard_montage('standard_1020')
EEG.set_montage(montage, on_missing='warn')
# check flat channels
rm_ch = check_flat_channels(EEG)
# drop bad channels
if rm_ch:
    EEG.drop_channels(rm_ch)

# Compute PSD up to 50Hz to check for fNIRS crosstalk at 9Hz harmonics
fnirs_sfreq = 9  # Hz
spectrum = EEG.compute_psd(fmin=0, fmax=50, n_fft=2048)
psd, freqs = spectrum.get_data(return_freqs=True)

# plot per-channel PSD with fNIRS harmonic markers — 2 figures x 2x4 grid
harmonics = np.arange(fnirs_sfreq, 51, fnirs_sfreq)
eeg_ch_names_psd = EEG.ch_names
channels_per_fig = 8

for fig_i in range(2):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for subplot_i in range(channels_per_fig):
        ei = fig_i * channels_per_fig + subplot_i
        ax = axes_flat[subplot_i]
        ax.semilogy(freqs, psd[ei], color='k', linewidth=1)
        for h in harmonics:
            ax.axvline(h, color='r', linestyle='--', linewidth=0.8,
                       label=f'{h:.0f} Hz' if h == harmonics[0] else None)
        ax.set_title(eeg_ch_names_psd[ei])
        ax.grid(True, alpha=0.3)
        if subplot_i >= 4:
            ax.set_xlabel('Frequency (Hz)')
        if subplot_i % 4 == 0:
            ax.set_ylabel('PSD (V²/Hz)')
        if subplot_i == 0:
            ax.legend(fontsize=7)
    plt.suptitle(f'EEG PSD with fNIRS harmonics — channels {fig_i*8+1}–{fig_i*8+8}')
    plt.tight_layout()
    plt.show()

#%% load fNIRS
import sys
import numpy as np
import xarray as xr
import cedalion
import cedalion.nirs
from cedalion import units
from cedalion.sigproc import quality
# import my own functions from a different directory
sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules')
import processing_func as pf

nirspath = '/projectnb/nphfnirs/s/datasets/gradCPT_NN24_pilot/sub-SPARK-testing-004/nirs'
nirs_fname = 'sub-SPARK-testing-004_task-RS_run-01_nirs.snirf'
records = cedalion.io.read_snirf( os.path.join(nirspath, nirs_fname) )

#=====================
cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_thresh' : [1, 40]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_thresh' : [1e-5, 0.84]*units.V, # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : False,
    'flag_use_psp' : False,
    'channel_sel': None
}

rec = records[0]
rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 )
rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 )

# if first value is 1e-18 then replace with second value
indices = np.where(rec['amp'][:,0,0] == 1e-18)
rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
indices = np.where(rec['amp'][:,1,0] == 1e-18)
rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
    
rec['amp'] = rec['amp'].pint.dequantify().pint.quantify('V')

geo3d = rec.geo3d
rec, chs_pruned = pf.prune_channels(
                                        rec, cfg_prune['amp_thresh'], 
                                        cfg_prune['sd_thresh'], 
                                        cfg_prune['snr_thresh']
                                    )

#%% calculate spectrum for each source,
"""
1. get the magnitude from all the channels related to the source.
2. ignore channels with nan magnitude
3. calculate power spectrum for the preserved channels
4. average the spectrum
"""
from scipy.signal import welch
from tqdm import tqdm

amp = rec['amp'].pint.dequantify()

t = amp.time.values
sfreq_nirs = 1.0 / float(np.median(np.diff(t)))

sources = np.unique(amp.coords['source'].values)
wavelengths = amp.coords['wavelength'].values

src_spectra = {}  # {source: {wavelength: (freqs, mean_psd) or None}}

for src in tqdm(sources):
    src_spectra[src] = {}
    src_mask = amp.coords['source'].values == src
    src_amp = amp.isel(channel=src_mask)  # (time, n_src_ch, wavelength)

    for wl in wavelengths:
        wl_data = src_amp.sel(wavelength=wl).values  # (time, n_src_ch) or (time,)
        if wl_data.ndim == 1:
            wl_data = wl_data[:, np.newaxis]

        # ignore channels with nan magnitude
        valid = ~np.any(np.isnan(wl_data), axis=0)
        wl_valid = wl_data[:, valid]

        if wl_valid.shape[1] == 0:
            src_spectra[src][wl] = None
            continue

        # calculate power spectrum for each valid channel
        psds = []
        for ch_i in range(wl_valid.shape[1]):
            f, psd = welch(wl_valid[:, ch_i], fs=sfreq_nirs,
                           nperseg=min(256, wl_valid.shape[0]))
            psds.append(psd)

        # average spectrum across channels
        mean_psd = np.mean(psds, axis=0)
        src_spectra[src][wl] = (f, mean_psd)

#%% Spectrum of fNIRS
from matplotlib import pyplot as plt
fig, axes = plt.subplots(len(sources), len(wavelengths),
                         figsize=(5 * len(wavelengths), 2.5 * len(sources)),
                         sharex=True, sharey=True)
axes = np.atleast_2d(axes)

for si, src in enumerate(sources):
    for wi, wl in enumerate(wavelengths):
        ax = axes[si, wi]
        result = src_spectra[src][wl]
        if result is not None:
            f, mean_psd = result
            ax.semilogy(f, mean_psd)
        ax.set_title(f'{src}, {wl:.0f} nm')
        if si == len(sources) - 1:
            ax.set_xlabel('Frequency (Hz)')
        if wi == 0:
            ax.set_ylabel('PSD (V²/Hz)')
        ax.grid(True, alpha=0.3)

plt.suptitle('fNIRS Power Spectrum per Source')
plt.tight_layout()
plt.show()

#%% Calculate covariance power spectrum between EEG and fNIRS
from scipy.signal import coherence, resample_poly
from math import gcd

eeg_data = EEG.get_data()  # (n_eeg_ch, n_eeg_samples)
sfreq_eeg = int(EEG.info['sfreq'])

nirs_amp = rec['amp'].pint.dequantify()
sfreq_nirs = int(round(1.0 / float(np.median(np.diff(nirs_amp.time.values)))))

# downsample EEG to fNIRS sampling rate
g = gcd(sfreq_eeg, sfreq_nirs)
eeg_ds = resample_poly(eeg_data, sfreq_nirs // g, sfreq_eeg // g, axis=1)

nirs_vals = nirs_amp.values  # (time, channel, wavelength)
n_samples = min(eeg_ds.shape[1], nirs_vals.shape[0])
eeg_ds = eeg_ds[:, :n_samples]
nirs_vals = nirs_vals[:n_samples]  # (n_samples, n_nirs_ch, n_wl)

wavelengths = nirs_amp.coords['wavelength'].values
nperseg = min(256, n_samples)

eeg_ch_names = EEG.ch_names
n_eeg_ch = eeg_ds.shape[0]

# for each EEG channel: compute coherence with each valid fNIRS channel, then average across fNIRS channels
# result shape: (n_eeg_ch, n_wl, n_freqs)
coh_per_eeg = np.full((n_eeg_ch, len(wavelengths), nperseg // 2 + 1), np.nan)

for ei in range(n_eeg_ch):
    for wi, wl in enumerate(wavelengths):
        nirs_wl = nirs_vals[:, :, wi]  # (n_samples, n_nirs_ch)
        cohs = []
        for ni in range(nirs_wl.shape[1]):
            if np.any(np.isnan(nirs_wl[:, ni])):
                continue
            f, Cxy = coherence(eeg_ds[ei], nirs_wl[:, ni],
                               fs=sfreq_nirs, nperseg=nperseg)
            cohs.append(Cxy)
        if cohs:
            coh_per_eeg[ei, wi, :len(f)] = np.mean(cohs, axis=0)

# smooth coherence with a 1Hz moving average
df = sfreq_nirs / nperseg  # Hz per frequency bin
win = max(1, int(round(1.0 / df)))
kernel = np.ones(win) / win
coh_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'),
                                  axis=2, arr=np.nan_to_num(coh_per_eeg))
# restore nans where original was nan
coh_smooth[np.all(np.isnan(coh_per_eeg), axis=2)[:, :, np.newaxis].repeat(coh_per_eeg.shape[2], axis=2)] = np.nan

# plot: 2 figures x 8 channels each (2x4 grid), both wavelengths per subplot
wl_colors = ['b', 'r']
channels_per_fig = 8

for fig_i in range(2):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for subplot_i in range(channels_per_fig):
        ei = fig_i * channels_per_fig + subplot_i
        ax = axes_flat[subplot_i]
        for wi, wl in enumerate(wavelengths):
            ch_coh = coh_smooth[ei, wi]
            if not np.all(np.isnan(ch_coh)):
                ax.plot(f, ch_coh, color=wl_colors[wi], linewidth=1.2,
                        label=f'{wl:.0f} nm')
        ax.axhline(0.8, color='r', linestyle='--', linewidth=1)
        ax.set_title(eeg_ch_names[ei])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        if subplot_i >= 4:
            ax.set_xlabel('Frequency (Hz)')
        if subplot_i % 4 == 0:
            ax.set_ylabel('Coherence')
        if subplot_i == 0:
            ax.legend(fontsize=8)
    plt.suptitle(f'EEG–fNIRS Coherence (channels {fig_i*8+1}–{fig_i*8+8})')
    plt.tight_layout()
    plt.show()
