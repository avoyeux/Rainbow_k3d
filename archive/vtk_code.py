from pyevtk.hl import gridToVTK
import numpy as np
import glob
from scipy.io import readsav
list_save = glob.glob('cube*.save')
list_save.sort()
x=0
y=0
z=0

for fi_save in list_save:
    res=readsav(fi_save)
    data=res['cube'].transpose(2,1,0)
    if np.size(x) == 1:
        x=np.arange(res['xt_min'],res['xt_max'],res['dx'])
        y=np.arange(res['yt_min'],res['yt_max'],res['dy'])
        z=np.arange(res['zt_min'],res['zt_max'],res['dz'])
    vtk_name='cube'+fi_save[-8:-5]
    gridToVTK(vtk_name, x, y, z, pointData = {'monsoon':data})



