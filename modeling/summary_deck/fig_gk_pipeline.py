import pickle, numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt, detrend
from style import *
P='/projectnb/nphfnirs/s/datasets/gradCPT_NN24'; D=f'{P}/derivatives/eeg'; s='sub-723'
feats=pd.read_csv(f'{D}/{s}/features/{s}_EEG_trial_features.csv.gz')
mc=feats[(feats.trial_type=='mnt')&(feats.response_code==0)&feats.cz_erp_ok&(feats.run==1)]
raw=mne.io.read_raw_fif(f'{D}/{s}/preprocessed_EEG_and_events/{s}_task-gradCPT_run-01_preproc_eeg.fif',preload=True,verbose=False).pick(['cz'])
x=raw.get_data()[0]*1e6; fs=raw.info['sfreq']; sos=butter(4,10,fs=fs,output='sos')
n0,n1=int(0.2*fs),int(1.6*fs); t=np.arange(-n0,n1)/fs
def ep(o):
    i=int(round(o*fs)); e=detrend(x[i-n0:i+n1]); e=sosfiltfilt(sos,e); return e-e[:n0].mean()
q=mc.cz_N2P3_ptp_woody.quantile([0.15,0.5,0.9]).values
pick=[mc.iloc[(mc.cz_N2P3_ptp_woody-v).abs().argmin()] for v in q]
cols=['#9a9a9a','#555555','#1f1f1f']
# panel 1: single-trial ERPs with N2 / P3 marks
fig,ax=plt.subplots(figsize=(5,3.6),constrained_layout=True)
for r,c in zip(pick,cols):
    e=ep(r.onset_eeg); ax.plot(t*1000,e,color=c,lw=1.6,label=f'ptp = {r.cz_N2P3_ptp_woody:.0f} µV')
    n2t,n2a=r.cz_N2P3_N2lat_woody,r.cz_N2P3_N2amp_woody; p3t=r.cz_P3_lat_woody; p3a=n2a+r.cz_N2P3_ptp_woody
    ax.plot([n2t*1000],[n2a],'v',color=CONT,ms=8); ax.plot([p3t*1000],[p3a],'^',color=EEGINF,ms=8)
ax.axvspan(250,800,color='#f1f1ee',zorder=0,lw=0); ax.axvline(0,color='#bbb',lw=0.8); ax.axhline(0,color='#ccc',lw=0.6)
ax.set(xlabel='Time from onset (ms)',ylabel='Cz (µV)',xlim=(-200,1200),title='Single-trial Cz ERP (3 trials)')
ax.plot([],[],'v',color=CONT,label='N2 trough'); ax.plot([],[],'^',color=EEGINF,label='Woody P3 peak'); ax.set_ylim(-22,58); ax.legend(fontsize=8.5,loc='upper left',ncol=2)
fig.savefig('pipe_gk_1.png',dpi=220)
# panel 2: scale distribution
dm=pickle.load(open(f'{D}/{s}/dm/gaussian-kernel_mnt-correct_EEG-informed_Cz-P3-ptp-Woody.pkl','rb'))
sc=np.concatenate([v['scale'] for v in dm['trial_info'].values()])
fig,ax=plt.subplots(figsize=(5,3.6),constrained_layout=True)
ax.hist(sc,bins=25,color=EEGINF,edgecolor='white',lw=1); ax.axvline(1,color=BENCH,lw=2)
ax.text(1.03,ax.get_ylim()[1]*0.9,'benchmark: every trial = 1',fontsize=10)
ax.set(xlabel='aᵢ / ā  (ptp ÷ run mean)',ylabel='mnt-correct trials',title=f'Trial weights, sub-723 (n = {len(sc)})')
fig.savefig('pipe_gk_2.png',dpi=220)
# panels 3-4: onset weights and one kernel regressor over a 60 s window of run 1
c=dm['dm']['full'].common.sel(chromo='HbO'); samples=c.samples.values
b=np.concatenate([[0],np.where(np.diff(samples)<=0)[0]+1,[len(samples)]]); seg=slice(b[0],b[1])
dt=np.median(np.diff(c.time.values)); trun=samples[seg]*dt
ev=pd.read_csv(f'{P}/{s}/nirs/{s}_task-gradCPT_run-01_events.tsv',sep='\t')
on=ev[(ev.trial_type=='mnt')&(ev.response_code==0)].onset.values; w=dm['trial_info']['gradcpt1']['scale']
t0=on[3]-5; win=(trun>=t0)&(trun<=t0+60); m=(on>=t0)&(on<=t0+60)
fig,ax=plt.subplots(figsize=(5,3.6),constrained_layout=True)
ax.vlines(on[m]-t0-0.25,0,1,color=BENCH,lw=2.2,label='Benchmark (1)')
ax.vlines(on[m]-t0+0.25,0,w[m],color=EEGINF,lw=2.2,label='EEG-informed (aᵢ/ā)')
ax.set(xlabel='Run time (s)',ylabel='Onset weight',xlim=(0,60),title='Weighted mnt-correct onsets'); ax.legend(fontsize=9,loc='upper right'); ax.set_ylim(0,max(1.6,w[m].max()*1.15))
fig.savefig('pipe_gk_3.png',dpi=220)
k=7  # kernel centred at 5 s
fig,ax=plt.subplots(figsize=(5,3.6),constrained_layout=True)
ys=c.sel(regressor=f'HRF mnt-correct-stim {k:02d}').values[seg]; ye=c.sel(regressor=f'HRF mnt-correct-eeg {k:02d}').values[seg]
ax.plot(trun[win]-t0,ys[win],color=BENCH,lw=1.8,label='Benchmark column'); ax.plot(trun[win]-t0,ye[win],color=EEGINF,lw=1.8,label='EEG-informed column')
ax.set(xlabel='Run time (s)',ylabel='Regressor value',xlim=(0,60),title='Kernel 8 of 20 (centred at +5 s)'); ax.legend(fontsize=9,loc='upper right')
fig.savefig('pipe_gk_4.png',dpi=220)
# DM example: conventional design-matrix image (time down, regressors across)
names=[f'HRF mnt-correct-stim {i:02d}' for i in range(20)]
Xs=c.sel(regressor=names).values[seg][win]; Xe=c.sel(regressor=[n.replace('stim','eeg') for n in names]).values[seg][win]
tt=trun[win]-t0; vmax=max(Xs.max(),Xe.max())
fig,axs=plt.subplots(1,2,figsize=(10.5,6),constrained_layout=True,sharey=True)
for ax,X,ttl,wt,col in [(axs[0],Xs,'Benchmark GK: every onset weight = 1',np.ones(m.sum()),BENCH),(axs[1],Xe,'EEG-informed GK: onset weight = aᵢ/ā',w[m],EEGINF)]:
    im=ax.imshow(X,aspect='auto',cmap='Greys',vmin=0,vmax=vmax,extent=[-0.5,19.5,tt[-1],tt[0]],interpolation='nearest')
    ax.set_xticks([0,2,7,12,19],['−2','0','5','10','17']); ax.set_xlabel('Kernel centre (s)'); ax.set_title(ttl); ax.grid(False)
    for o,ww in (zip(on[m]-t0,wt) if col!=BENCH else []):
        ax.annotate(f'{ww:.2f}',xy=(19.5,o),xytext=(21.0,o),fontsize=9,color=col,va='center',ha='left',annotation_clip=False,
                    arrowprops=dict(arrowstyle='-',color=col,lw=0.8))
axs[0].set_ylabel('Run time (s), sub-723 run 1')
fig.colorbar(im,ax=axs,fraction=0.03,pad=0.09,label='Regressor value')
fig.savefig('dm_example.png',dpi=200)
print('ok', len(sc), sc.mean())
