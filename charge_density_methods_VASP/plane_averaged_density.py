from numpy import array, shape, zeros, argmax
from numpy.linalg import norm
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT

def find_vacuum_potential(ifile,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
        
    x,y,e,lv,coord=calc_plane_averaged_density(ifile,dim=dim)
    
    maxdiff=zeros(shape(e)[dim])
    for i in range(shape(e)[dim]):
        for j in coord[:,dim]:
            for k in range(-1,2):
                if maxdiff[i]<abs(x[i]-j+norm(lv[dim])*k):
                    maxdiff[i]=abs(x[i]-j+norm(lv[dim])*k)
    
    max_index=argmax(maxdiff)
    return y[max_index]

def calc_plane_averaged_density(ifile,filetype='LOCPOT',**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
    
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

def plot_plane_averaged_density(ifile,filetype='LOCPOT'):
    x,y=calc_plane_averaged_density(ifile,filetype)[:2]
    plt.figure()
    plt.plot(x,y)
    plt.ylabel('electrostatice potential / eV')
    plt.xlabel('position / $\AA$')
    plt.show()
