import pickle, numpy as np, pandas as pd
from scipy import stats
from style import *
D='/projectnb/nphfnirs/s/datasets/gradCPT_NN24/derivatives/eeg'
H=pickle.load(open('hrf_all.pkl','rb')); T=H['T']; R=H['res']; S=list(R)
P=['DorsAttnB_PostC_2_RH','SomMotA_25_LH']
C={'bench':BENCH,'eeginf':EEGINF,'cont':CONT,'contrun':CONTRUN}
L={'bench':'Benchmark (event GK)','eeginf':'EEG-informed GK','cont':'Cont-EEG (mnt-correct trials)','contrun':'Cont-EEG (entire run)'}
CK={'bench':'onlyStim','eeginf':'onlyEEG','cont':'contEEG'}
cmp={s:pickle.load(open(f'{D}/{s}/cmp_HRF_DM/{s}_cmp_HRF_DM_mnt-correct.pkl','rb'))['curves'] for s in S}
def curve(s,p,k):
    if k in CK:
        c=cmp[s][p][CK[k]]; return np.interp(T,c['t'],c['h']),np.interp(T,c['t'],c['se'])
    return R[s][k].loc[p].values,None
def ci(m): return m.mean(0), m.std(0,ddof=1)/np.sqrt(len(m))*stats.t.ppf(0.975,len(m)-1)
def two_parcel(k,fname):
    fig,axs=plt.subplots(2,7,figsize=(16,5.6),sharex=True,constrained_layout=True)
    for r,p in enumerate(P):
        acc={kk:[] for kk in ('bench',k)}
        for j,s in enumerate(S):
            a=axs[r,j]; ba=cmp[s][p]['block_average']; a.plot(ba['t'],ba['h'],color='#c4c4c4',lw=0.8,zorder=1)
            for kk in ('bench',k):
                h,se=curve(s,p,kk); acc[kk].append(h)
                if se is not None: a.fill_between(T,h-1.96*se,h+1.96*se,color=C[kk],alpha=0.13,lw=0)
                a.plot(T,h,color=C[kk],lw=1.8)
            a.axhline(0,color='#ccc',lw=0.7); a.axvline(0,color='#ccc',lw=0.7)
            if r==0: a.set_title(s)
        a=axs[r,6]
        for kk in ('bench',k):
            mu,c=ci(np.array(acc[kk])); a.fill_between(T,mu-c,mu+c,color=C[kk],alpha=0.16,lw=0); a.plot(T,mu,color=C[kk],lw=2.2,label=L[kk])
        a.axhline(0,color='#ccc',lw=0.7); a.axvline(0,color='#ccc',lw=0.7)
        if r==0: a.set_title(f'Mean (N = {len(S)})',fontweight='bold')
        axs[r,0].set_ylabel(f'{p}\nHbO (µM)',fontsize=10)
    for a in axs[1]: a.set_xlabel('Time from onset (s)')
    axs[1,0].plot([],[],color='#c4c4c4',lw=0.8,label='Block average of Y')
    h1,l1=axs[0,6].get_legend_handles_labels(); h2,l2=axs[1,0].get_legend_handles_labels()
    fig.legend(h1+h2,l1+l2,loc='outside lower center',ncol=3,fontsize=11)
    fig.savefig(fname,dpi=200)
NET={'VisCent':'Visual Central','SomMotA':'Somatomotor A','SomMotB':'Somatomotor B','DorsAttnA':'Dorsal Attention A','DorsAttnB':'Dorsal Attention B',
 'SalVentAttnA':'Ventral Attention A','SalVentAttnB':'Ventral Attention B','ContA':'Control A','ContB':'Control B','DefaultA':'Default Mode A','DefaultB':'Default Mode B','TempPar':'Temporal Parietal'}
def netcurve(s,k,net):
    df=R[s][k]; return df.loc[[p for p in df.index if p.split('_')[0]==net]].mean(0).values
def grid(k,fname):
    fig,axs=plt.subplots(3,4,figsize=(16,9),sharex=True,constrained_layout=True)
    for a,net in zip(axs.flat,NET):
        for kk in ('bench',k):
            mu,c=ci(np.array([netcurve(s,kk,net) for s in S]))
            a.fill_between(T,mu-c,mu+c,color=C[kk],alpha=0.16,lw=0); a.plot(T,mu,color=C[kk],lw=2,label=L[kk])
        npar=sum(p.split('_')[0]==net for p in R[S[0]]['bench'].index)
        a.set_title(f'{NET[net]} ({npar} parcels)'); a.axhline(0,color='#ccc',lw=0.7); a.axvline(0,color='#ccc',lw=0.7)
    for a in axs[-1]: a.set_xlabel('Time from onset (s)')
    for a in axs[:,0]: a.set_ylabel('HbO (µM)')
    h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=12)
    fig.savefig(fname,dpi=170)
two_parcel('eeginf','fig_2p_eeginf.png'); two_parcel('cont','fig_2p_cont.png'); two_parcel('contrun','fig_2p_contrun.png')
grid('eeginf','fig_net_eeginf.png'); grid('cont','fig_net_cont.png'); grid('contrun','fig_net_contrun.png')
# stats
w=(T>=0)&(T<=18)
def pr(a,b):
    a=a[:,w]-a[:,w].mean(1,keepdims=True); b=b[:,w]-b[:,w].mean(1,keepdims=True)
    return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
rms=lambda x: np.sqrt((x[:,w]**2).mean(1))
out={}
for k in ('eeginf','cont','contrun'):
    rs,am=[],[]
    for s in S:
        ps=R[s]['bench'].index.intersection(R[s][k].index)
        a=R[s]['bench'].loc[ps].values; b=R[s][k].loc[ps].values
        rs.append(np.median(pr(a,b))); am.append(np.median(rms(b)/rms(a)))
    # 2-parcel
    r2=[np.corrcoef(curve(s,p,'bench')[0][w],curve(s,p,k)[0][w])[0,1] for s in S for p in P]
    a2=[np.abs(curve(s,p,k)[0][w]).max()/np.abs(curve(s,p,'bench')[0][w]).max() for s in S for p in P]
    netr=[np.corrcoef(np.mean([netcurve(s,'bench',n) for s in S],0)[w],np.mean([netcurve(s,k,n) for s in S],0)[w])[0,1] for n in NET]
    out[k]=dict(parcel_r_med=np.median(rs),parcel_r_subj=np.round(rs,2).tolist(),rms_ratio_med=np.median(am),
               two_r_med=np.median(r2),two_r_rng=(min(r2),max(r2)),two_amp_med=np.median(a2),net_r=np.round(netr,2).tolist(),net_r_med=np.median(netr))
    print(k,{kk:(np.round(v,3) if isinstance(v,float) else v) for kk,v in out[k].items()})
# F-tests at the 2 parcels
for k,ft in [('bench','onlyStim'),('eeginf','onlyEEG'),('cont','contEEG')]:
    ps=[cmp[s][p][ft]['p'] for s in S for p in P]; print(k,'p<0.05:',sum(np.array(ps)<0.05),'of',len(ps))
pickle.dump(out,open('hrf_stats.pkl','wb'))
