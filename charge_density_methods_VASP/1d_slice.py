import numpy as np
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT

def calc_plane_averaged_density(ifile,atoms,filetype='LOCPOT',**args):
    
    if filetype=='LOCPOT':
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    else:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    npts=np.shape(e)
    
    x=np.array([i/(npts[2]-1)*np.linalg.norm(lv[2]) for i in range(npts[2])])
    y=[np.zeros(npts[2]) for i in range(len(atoms))]
    for i in range(len(atoms)):
        pos=coord[atoms[i]-1,:2]
        pos=np.dot(pos,np.linalg.inv(lv[:2,:2]))
        pos*=npts[:2]
        pos=pos.astype(int)
        y[i]=e[pos[0],pos[1],:]
                
    return x,y,atoms,e,lv,coord

def plot_plane_averaged_density(ifile,filetype='LOCPOT'):
    x,y,atoms=calc_plane_averaged_density(ifile,filetype)[:3]
    plt.figure()
    for i,j in zip(y,atoms):
        plt.plot(x,i,label=j)
    if filetype=='LOCPOT':
        plt.ylabel('electrostatice potential / eV')
    elif 'CHG' in filetype:
        plt.ylabel('charge density / # electrons $A^{-3}$')
    plt.legend()
    plt.xlabel('position / $\AA$')
    plt.show()
