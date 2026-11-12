import numpy as np, pandas as pd, pickle
from scipy import stats
from style import *
tr=pd.read_csv('eeg_trial_feats.csv')
BANDS=['delta','theta','alpha','beta']
MOD=['sub-670','sub-695','sub-721','sub-723','sub-726','sub-730']
def subj_means(col, clean):
    d=tr[tr.ok] if clean else tr
    m=d.groupby(['subject','inzone'])[col].agg(['mean','count']).unstack()
    m=m[(m['count'][True]>=10)&(m['count'][False]>=10)]
    return m['mean'][True], m['mean'][False]
def paired(ax,col,clean,ylab,title,scale=1):
    a,b=subj_means(col,clean); a,b=a*scale,b*scale
    for s in a.index:
        ax.plot([0,1],[a[s],b[s]],color='#b5b5b5' if s not in MOD else '#555',lw=1,zorder=1)
    ax.scatter(np.zeros(len(a)),a,color=IN,s=28,zorder=3); ax.scatter(np.ones(len(b)),b,color=OUT,s=28,zorder=3)
    ax.plot([-0.18,0.18],[a.mean()]*2,color=IN,lw=3); ax.plot([0.82,1.18],[b.mean()]*2,color=OUT,lw=3)
    t,p=stats.ttest_rel(a,b)
    ax.set_xticks([0,1],['In-zone','Out-of-zone']); ax.set_xlim(-0.5,1.5); ax.set_ylabel(ylab)
    ax.set_title(f'{title}\npaired t({len(a)-1}) = {t:.2f}, p = {p:.2f}'); ax.grid(axis='x',visible=False)
    return dict(feature=title,n=len(a),mean_in=a.mean(),mean_out=b.mean(),t=t,p=p)
res=[]
fig,axs=plt.subplots(1,2,figsize=(8.5,4.6),constrained_layout=True)
res.append(paired(axs[0],'ptp',True,'N2–P3 peak-to-peak (µV)','ERP amplitude'))
res.append(paired(axs[1],'p3lat',True,'P3 latency (ms)','ERP latency',1000))
fig.savefig('fig_zone_erp.png',dpi=200)
fig,axs=plt.subplots(1,4,figsize=(13,4.4),constrained_layout=True)
for ax,b in zip(axs,BANDS):
    res.append(paired(ax,b,False,'Σ Hilbert amplitude in trial (µV·s)' if b=='delta' else '',f'{b.capitalize()}'))
fig.savefig('fig_zone_hilbert.png',dpi=200)
r=pd.DataFrame(res); print(r.round(3).to_string()); r.to_csv('zone_stats.csv',index=False)
# appendix: waveforms
W=pickle.load(open('eeg_waves.pkl','rb')); t=W['t']*1000
subs=[s for s,v in W['subj'].items() if v['n'][True]>=10 and v['n'][False]>=10]
def band(ax,key,ylab,title):
    for z,c,l in [(True,IN,'In-zone'),(False,OUT,'Out-of-zone')]:
        m=np.array([W['subj'][s][key][z] for s in subs]); mu=m.mean(0); se=m.std(0,ddof=1)/np.sqrt(len(m))
        ax.fill_between(t,mu-1.96*se,mu+1.96*se,color=c,alpha=0.18,lw=0); ax.plot(t,mu,color=c,lw=2,label=l)
    ax.axvline(0,color='#bbb',lw=0.8); ax.set_xlabel('Time from mnt-correct onset (ms)'); ax.set_ylabel(ylab); ax.set_title(title)
fig,ax=plt.subplots(figsize=(7.5,4.4),constrained_layout=True)
band(ax,'erp','Cz (µV)',f'Cz ERP, mnt-correct, N = {len(subs)} (mean ± 95% CI)')
ax.axvspan(250,800,color='#eee',zorder=0,lw=0); ax.text(260,ax.get_ylim()[1]*0.92,'P3 window',fontsize=9,color='#666')
ax.axhline(0,color='#bbb',lw=0.8); ax.legend(); fig.savefig('fig_app_erp_zone.png',dpi=200)
fig,axs=plt.subplots(1,4,figsize=(14,4),constrained_layout=True)
for ax,b in zip(axs,BANDS): band(ax,b,'Hilbert amplitude (µV)' if b=='delta' else '',b.capitalize())
axs[0].legend(); fig.savefig('fig_app_hilbert_tc.png',dpi=200)
print('N subjects waves', len(subs))
