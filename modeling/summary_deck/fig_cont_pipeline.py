import pickle, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt
from style import *
D='/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg'; s='sub-723'
feats=pd.read_csv(f'{D}/{s}/features/{s}_EEG_trial_features.csv.gz'); g=feats[feats.run==2]
ev=pd.read_csv(f'{D}/{s}/preprocessed_EEG_and_events/{s}_task-gradCPT_run-02_events.tsv',sep='\t')
raw=mne.io.read_raw_fif(f'{D}/{s}/preprocessed_EEG_and_events/{s}_task-gradCPT_run-02_preproc_eeg.fif',preload=True,verbose=False).pick(['cz'])
x=raw.get_data()[0]*1e6; fs=raw.info['sfreq']; t=np.arange(len(x))/fs
mc=g[(g.trial_type=='mnt')&(g.response_code==0)]
mask=np.zeros(len(x),bool); xm=np.zeros(len(x))
for _,r in mc.iterrows():
    dur=ev.loc[r.trial_idx,'duration']; i0=int(round(r.onset_eeg*fs)); i1=int(round((r.onset_eeg+dur)*fs))
    off=x[i0-int(0.2*fs):i0].mean(); xm[i0:i1]=x[i0:i1]-off; mask[i0:i1]=True
# window: pick two close mnt-correct trials
on=mc.onset_eeg.values; j=np.argmin(np.diff(on)+1e3*(np.diff(on)<3)); t0=on[j]-3; t1=t0+24
w=(t>=t0)&(t<=t1)
B=pickle.load(open(f'{D}/{s}/betas/{s}_cont_EEG_cz_mnt_correct_incorrect_pre2post18_bl0.2-0_ar_irls_Hp0.01_betas.pkl','rb'))['basis_da'].values
delay=-2+np.arange(B.shape[0])/fs
sel=[2,6,10]; cc=['#2a78d6','#1baf7a','#6250d6']
figs=[]
def mk(): f,a=plt.subplots(figsize=(6.2,2.5),constrained_layout=True); return f,a
def shade(a):
    for o in on[(on>t0-1)&(on<t1)]:
        a.axvspan(o-t0,o-t0+0.8,color='#eb6834',alpha=0.18,lw=0)
f,a=mk(); a.plot(t[w]-t0,x[w],color='#555',lw=0.6); shade(a); a.set(ylabel='Cz (µV)',title='1. Continuous Cz (orange: mnt-correct trials)',xlim=(0,24)); f.savefig('pipe_c1.png',dpi=220)
f,a=mk(); a.plot(t[w]-t0,xm[w],color='#1f1f1f',lw=0.8); shade(a); a.set(ylabel='Cz (µV)',title='2. Zero outside trials, baseline −0.2–0 s removed',xlim=(0,24)); f.savefig('pipe_c2.png',dpi=220)
f,a=plt.subplots(figsize=(3.2,2.5),constrained_layout=True)
for k in range(B.shape[1]): a.plot(delay,B[:,k],color='#d0d0d0',lw=0.8)
for k,c in zip(sel,cc): a.plot(delay,B[:,k],color=c,lw=2)
a.set(xlabel='Delay (s)',title='3. 20 cubic B-splines (−2 to 18 s)'); f.savefig('pipe_c3.png',dpi=220)
# regressors at EEG rate, then lowpass + sample at fNIRS rate
nf=8.98; sos=butter(4,nf/2*0.9,fs=fs,output='sos'); k0=int(2*fs)
f,a=mk()
for k,c in zip(sel,cc):
    r=np.convolve(xm,B[:,k])[:len(xm)+k0][k0:]/fs   # shift so delay -2 aligns
    r=np.convolve(xm,B[:,k])/fs; tr=np.arange(len(r))/fs-2
    rl=sosfiltfilt(sos,r); ts=np.arange(t0,t1,1/nf); rs=np.interp(ts,tr,rl)
    a.plot(tr[(tr>=t0)&(tr<=t1)]-t0,r[(tr>=t0)&(tr<=t1)],color=c,lw=0.6,alpha=0.6)
    a.plot(ts-t0,rs,'o',color=c,ms=2.5)
shade(a); a.axhline(0,color='#ccc',lw=0.6)
a.set(xlabel='Run time (s)',ylabel='µV·s',title='4. Convolve, low-pass, read at fNIRS samples (3 of 20 columns)',xlim=(0,24)); f.savefig('pipe_c4.png',dpi=220)
print('window',t0)
