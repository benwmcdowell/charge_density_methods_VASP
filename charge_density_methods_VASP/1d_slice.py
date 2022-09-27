import numpy as np
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT

def calc_density(ifile,atoms,filetype='LOCPOT',**args):
    
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

def plot_density(ifile,filetype='LOCPOT',linestyle='default',linecolors='default',lw='default'):
    x,y,atoms=calc_density(ifile,filetype)[:3]
    if linestyle=='default':
        linestyle=['solid' for i in range(len(y))]
    if lw=='default':
        lw=[1 for i in range(len(y))]
    
    plt.figure()
    for i in range(len(y)):
        if linecolors=='default':
            plt.plot(x,y[i],label=atoms[i],linestyle=linestyle[i],lw=lw[i])
        else:
            plt.plot(x,y[i],label=atoms[i],linestyle=linestyle[i],lw=lw[i],color=linecolors[i])
    if filetype=='LOCPOT':
        plt.ylabel('electrostatic potential / eV')
    elif 'CHG' in filetype:
        plt.ylabel('charge density / # electrons $A^{-3}$')
    plt.legend()
    plt.xlabel('position / $\AA$')
    plt.show()
