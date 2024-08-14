from mpl_toolkits.mplot3d import Axes3D
from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt
import glob

path = '/home/alfred/Documents/helio_304/prog/'
list = glob.glob(path + 'cube*.save')
list.sort()

for i, file in enumerate(list):
	res = readsav(file)
	pngfile = 'polyfit_' + file[-8:-5] + '.png'
#res=readsav('/home/alfred/Documents/helio_304/prog/cube353.save')

	data=res['cube']
	datat=data.transpose(2,1,0)
	xx=np.arange(res['xt_min'],res['xt_max'],res['dx'])
	yy=np.arange(res['yt_min'],res['yt_max'],res['dy'])
	zz=np.arange(res['zt_min'],res['zt_max'],res['dz'])


	datatcopy=datat.copy()

	datatcopy[0:37,:,:]=0
	datatcopy[50:65,20:70,120:180]=0

	wcopy=np.where(datatcopy>0.06)

	x=xx[wcopy[0]]
	y=yy[wcopy[1]]
	z=zz[wcopy[2]]

	ypoly=np.polyfit(z,y,2)
	ypolynome=np.poly1d(ypoly)

	zlin=np.linspace(-80000,180000,100)

	xpoly=np.polyfit(z,x,4)
	xpolynome=np.poly1d(xpoly)

	xlin=xpolynome(zlin)
	ylin=ypolynome(zlin)

	Rsun=695700.0
	u = np.linspace(3*np.pi/4, np.pi+np.pi/6, 100)
	v = np.linspace(0.25*np.pi, 0.75*np.pi, 100)
	#v = np.linspace(0, np.pi, 100)
	xsun = Rsun * np.outer(np.cos(u), np.sin(v))
	ysun = Rsun * np.outer(np.sin(u), np.sin(v))
	zsun = Rsun * np.outer(np.ones(np.size(u)), np.cos(v))


	fig=plt.figure()
	ax=Axes3D(fig)
	#ax.set_proj_type('ortho')
	w=np.where(datat>0.06)
	ax.view_init(elev=0., azim = 0)
	ax.plot(xx[w[0]],yy[w[1]],zz[w[2]],'o')
	ax.plot(xx[wcopy[0]],yy[wcopy[1]],zz[wcopy[2]],'o',color='red')
	ax.plot(xlin,ylin,zlin,color='black',lw=5)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	# Plot the sun surface
	ax.plot_surface(xsun, ysun, zsun, color='orange')
	#ax.set_aspect('equal', 'datalim')
	plt.xlim(-Rsun,Rsun)
	plt.ylim(-Rsun,Rsun)
	plt.savefig(pngfile, dpi=300)
	#plt.show()
	plt.close(fig)
 
