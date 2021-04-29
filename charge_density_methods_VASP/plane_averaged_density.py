from numpy import array, shape, zeros, argmin
from numpy.linalg import norm
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT

def find_vacuum_potential(ifile,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
        
    x,y,e,lv,coord=calc_plane_averaged_density(ifile,dim=dim)
    
    mindiff=array([norm(lv[dim]) for i in range(shape(e)[dim])])
    for i in range(shape(e)[dim]):
        for j in coord[:,dim]:
            for k in range(3):
                if mindiff[i]>abs(x[i]-j+norm(lv[dim])*k):
                    mindiff[i]=abs(x[i]-j+norm(lv[dim])*k)
    
    min_index=argmin(mindiff)
    return y[min_index]

def calc_plane_averaged_density(ifile,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
    
    if 'filetype' in args:
        filetype=args['filetype']
    else:
        filetype='LOCPOT'
    
    if filetype=='LOCPOT':
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    else:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    
    x=array([i*norm(lv[dim])/(shape(e)[dim]-1) for i in range(shape(e)[dim])])
    y=zeros(shape(e)[dim])
    for i in range(shape(e)[dim]):
        y[i]+=sum(e[:,:,i].flatten())
        for j in range(3):
            if j!=dim:
                y[i]/=shape(e)[j]
                
    return x,y,e,lv,coord

def plot_plane_averaged_density(ifile):
    x,y=calc_plane_averaged_density(ifile)[:2]
    plt.figure()
    plt.plot(x,y)
    plt.ylabel('electrostatice potential / eV')
    plt.xlabel('position / $\AA$')
    plt.show()
