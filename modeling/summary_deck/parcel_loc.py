import numpy as np, xarray as xr, cedalion.dot
from cedalion.vis.anatomy.image_recon import image_recon
from cedalion.vis.blocks import plot_labeled_points
head=cedalion.dot.get_standard_headmodel('icbm152')
vp=head.brain.vertices.parcel.values; n=head.brain.nvertices
vals=np.full(n,np.nan); vals[vp=='DorsAttnB_PostC_2_RH']=0.25; vals[vp=='SomMotA_25_LH']=0.85
X=xr.DataArray(np.stack([vals,np.zeros(n)],-1),dims=['vertex','chromo'],coords={'chromo':['HbO','HbR'],'is_brain':('vertex',np.ones(n,bool))})
for view in ['superior','left','right']:
    p0,_,_=image_recon(X,head,cmap='coolwarm',clim=(0,1),view_type='hbo_brain',view_position=view,off_screen=True,wdw_size=(900,900))
    plot_labeled_points(p0,head.landmarks.sel(label=['Cz']))
    p0.screenshot(f'parcel_{view}.png'); p0.close(); print(view)
