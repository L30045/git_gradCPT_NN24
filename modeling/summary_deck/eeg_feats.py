# per-trial Cz features for mnt-correct trials, all subjects: N2-P3 ptp (Woody), P3 latency (Woody),
# Hilbert amplitude summed over the trial (onset -> onset + duration) for delta/theta/alpha/beta, VTC zone
import glob, os, numpy as np, pandas as pd, mne
D='/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg'
BANDS={'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
rows=[]
for p in sorted(glob.glob(f'{D}/sub-*/features/*_EEG_trial_features.csv.gz')):
    df=pd.read_csv(p); subj=df.subject.iloc[0] if 'subject' in df else os.path.basename(p)[:7]
    thr=np.nanmedian(df.VTC_smoothed)  # subject median over all trials (as single_Subj_analysis_ERP_ERSP)
    for run,g in df.groupby('run'):
        f=f'{D}/{subj}/preprocessed_EEG_and_events/{subj}_task-gradCPT_run-{run:02d}_preproc_eeg.fif'
        ev=pd.read_csv(f.replace('_preproc_eeg.fif','_events.tsv'),sep='\t')
        if not os.path.exists(f): continue
        raw=mne.io.read_raw_fif(f,preload=True,verbose=False)
        if 'cz' not in raw.ch_names: continue
        raw.pick(['cz']); fs=raw.info['sfreq']
        env={}
        for b,(lo,hi) in BANDS.items():
            x=raw.copy().filter(lo,hi,verbose=False).apply_hilbert(envelope=True,verbose=False).get_data()[0]*1e6
            env[b]=x
        mc=g[(g.trial_type=='mnt')&(g.response_code==0)]
        for _,t in mc.iterrows():
            dur=ev.loc[t.trial_idx,'duration']
            i0=int(round(t.onset_eeg*fs)); i1=int(round((t.onset_eeg+dur)*fs))
            r=dict(subject=subj,run=run,trial_idx=t.trial_idx,vtc=t.VTC_smoothed,inzone=t.VTC_smoothed<thr,
                   ok=bool(t.cz_erp_ok),ptp=t.cz_N2P3_ptp_woody,p3lat=t.cz_P3_lat_woody,p3amp=t.cz_P3_amp_woody)
            for b in BANDS: r[b]=env[b][i0:i1].sum()/fs if i1<=len(env[b]) else np.nan
            rows.append(r)
    print(subj, len(rows), flush=True)
pd.DataFrame(rows).to_csv('eeg_trial_feats.csv',index=False)
