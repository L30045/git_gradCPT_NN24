# predicted HbO response (uM) to an average mnt-correct trial, every parcel, 6 subjects, 4 designs, -2..18 s
import pickle, numpy as np, pandas as pd, mne, os
D='/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg'
S=[670,695,721,723,726,730]; T=np.round(np.arange(-2,18.0001,0.1),1)
def bl(t,y): return y-y[...,t<0].mean(-1,keepdims=True)
out={}
for sid in S:
    s=f'sub-{sid}'; B=f'{D}/{s}/betas/{s}_'; r={}
    cmp=pickle.load(open(f'{D}/{s}/cmp_HRF_DM/{s}_cmp_HRF_DM_mnt-correct.pkl','rb'))
    scale=cmp['mean_scale']; ea=cmp['eeg_avg']; fs=cmp['fs_eeg']
    for k,f,sc in [('bench','event-based_onParcel_gaussian-kernel_mnt-correct_onlyStim_ar_irls_Hp',1.0),
                   ('eeginf','event-based_onParcel_gaussian-kernel_mnt-correct_onlyEEG-woodyN2P3ptp_ar_irls_Hp',scale)]:
        b=pickle.load(open(B+f+'_betas.pkl','rb'))
        h=b['hrf_estimate'].isel(trial_type=0).sel(chromo='HbO')
        t=h.time.values; v=h.pint.dequantify().values if hasattr(h,'pint') else h.values
        v=np.asarray(v,float)*1e6*sc
        r[k]=pd.DataFrame(np.array([np.interp(T,t,x) for x in v]),index=h.parcel.values,columns=T)
    # cont-EEG within mnt-correct trials: kernel (delay -2..18 s at 500 Hz) * average within-trial Cz
    b=pickle.load(open(B+'cont_EEG_cz_mnt_correct_incorrect_pre2post18_bl0.2-0_ar_irls_Hp0.01_betas.pkl','rb'))
    K=np.asarray(b['betas_eeg_per_type']['mnt_correct'].sel(chromo='HbO').values,float)
    R=np.array([np.convolve(ea,k) for k in K]); tau=-b['t_pre']+np.arange(R.shape[1])/fs
    R=bl(tau,R)*1e6
    r['cont']=pd.DataFrame(np.array([np.interp(T,tau,x) for x in R]),index=b['betas_eeg_per_type']['mnt_correct'].parcel.values,columns=T)
    # cont-EEG entire run: kernel (delay 0..15 s) * event-locked average of continuous Cz around mnt-correct onsets
    feats=pd.read_csv(f'{D}/{s}/features/{s}_EEG_trial_features.csv.gz')
    segs=[]; pre,post=17.0,18.0
    for run,g in feats.groupby('run'):
        raw=mne.io.read_raw_fif(f'{D}/{s}/preprocessed_EEG_and_events/{s}_task-gradCPT_run-{run:02d}_preproc_eeg.fif',verbose=False,preload=True).pick(['cz'])
        x=raw.get_data()[0]; f2=raw.info['sfreq']; n0,n1=int(pre*f2),int(post*f2)
        for o in g[(g.trial_type=='mnt')&(g.response_code==0)].onset_eeg:
            i=int(round(o*f2))
            if i-n0>=0 and i+n1<=len(x): segs.append(x[i-n0:i+n1])
    xbar=np.mean(segs,0); tx=-pre+np.arange(len(xbar))/f2
    b=pickle.load(open(B+'cont_EEG_cz_3-stage_bspline-test_ar_irls_Hp_betas.pkl','rb'))
    K=np.asarray(b['betas_eeg'].sel(chromo='HbO').values,float)   # (parcel, 7500): delay k/500 s
    full=np.array([np.convolve(xbar,k) for k in K])               # index n -> time tx[0] + n/fs
    tt=tx[0]+np.arange(full.shape[1])/f2
    keep=(tt>=-2.05)&(tt<=18.05); full=full[:,keep]; tt=tt[keep]
    full=bl(tt,full)*1e6
    r['contrun']=pd.DataFrame(np.array([np.interp(T,tt,x) for x in full]),index=b['betas_eeg'].parcel.values,columns=T)
    r['n_trials_run_avg']=len(segs)
    out[s]=r; print(s,{k:v.shape for k,v in r.items() if hasattr(v,'shape')},flush=True)
pickle.dump(dict(T=T,res=out),open('hrf_all.pkl','wb'))
