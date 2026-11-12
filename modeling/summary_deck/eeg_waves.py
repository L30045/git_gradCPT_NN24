# per-subject average Cz ERP and Hilbert-amplitude time courses (mnt-correct, -0.2..1.6 s), in-zone vs out-of-zone
import glob, os, pickle, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt, detrend
D='/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg'
BANDS={'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
tr=pd.read_csv('eeg_trial_feats.csv')
out={}
for subj,gs in tr.groupby('subject'):
    acc={'erp':{True:[],False:[]}, **{b:{True:[],False:[]} for b in BANDS}}
    for run,g in gs.groupby('run'):
        f=f'{D}/{subj}/preprocessed_EEG_and_events/{subj}_task-gradCPT_run-{run:02d}_preproc_eeg.fif'
        raw=mne.io.read_raw_fif(f,preload=True,verbose=False).pick(['cz']); fs=raw.info['sfreq']
        x=raw.get_data()[0]*1e6
        feats=pd.read_csv(f'{D}/{subj}/features/{subj}_EEG_trial_features.csv.gz')
        on=feats.set_index(['run','trial_idx']).loc[[(run,t) for t in g.trial_idx],'onset_eeg'].values
        env={b:raw.copy().filter(lo,hi,verbose=False).apply_hilbert(envelope=True,verbose=False).get_data()[0]*1e6 for b,(lo,hi) in BANDS.items()}
        sos=butter(4,10,fs=fs,output='sos')
        n0,n1=int(0.2*fs),int(1.6*fs)
        for o,z,ok in zip(on,g.inzone.values,g.ok.values):
            i=int(round(o*fs))
            if i-n0<0 or i+n1>len(x): continue
            for b in BANDS: acc[b][z].append(env[b][i-n0:i+n1])
            if ok:
                e=detrend(x[i-n0:i+n1]); e=sosfiltfilt(sos,e); e=e-e[:n0].mean(); acc['erp'][z].append(e)
    out[subj]={k:{z:np.mean(v[z],0) if len(v[z]) else None for z in (True,False)} for k,v in acc.items()}
    out[subj]['n']={z:len(acc['erp'][z]) for z in (True,False)}
    print(subj,out[subj]['n'],flush=True)
t=np.arange(-int(0.2*500),int(1.6*500))/500
pickle.dump(dict(t=t,subj=out),open('eeg_waves.pkl','wb'))
