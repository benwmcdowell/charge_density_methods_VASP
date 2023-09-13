import numpy as np
import matplotlib.pyplot as plt
import os

from lib import parse_CHGCAR, parse_LOCPOT, parse_doscar, parse_poscar

def find_vacuum_potential(ifile,doscar=False,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
        
    x,y,e,lv,coord=calc_plane_averaged_density(ifile,dim=dim)
    
    if doscar:
        ef=parse_doscar(doscar)[2]
        e-=ef
    
    maxdiff=np.zeros(np.shape(e)[dim])
    for i in range(np.shape(e)[dim]):
        for j in coord[:,dim]:
            for k in range(-1,2):
                if maxdiff[i]<abs(x[i]-j+np.linalg.norm(lv[dim])*k):
                    maxdiff[i]=abs(x[i]-j+np.linalg.norm(lv[dim])*k)
    
    max_index=np.argmax(maxdiff)
    return y[max_index]

def calc_plane_averaged_density(ifile,filetype='LOCPOT',read_data_from_file=None,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
    
    if filetype=='LOCPOT' and not read_data_from_file:
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    elif 'CHG' in filetype and not read_data_from_file:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    elif read_data_from_file:
        os.chdir(ifile)
        e=np.load(read_data_from_file)
        try:
            lv,coord,atomtypes,atomnums=parse_poscar('./POSCAR')[:4]
        except FileNotFoundError:
            lv,coord,atomtypes,atomnums=parse_poscar('./CONTCAR')[:4]
    
    x=np.array([i*np.linalg.norm(lv[dim])/(np.shape(e)[dim]-1) for i in range(np.shape(e)[dim])])
    y=np.zeros(np.shape(e)[dim])
    for i in range(np.shape(e)[dim]):
        y[i]+=sum(e[:,:,i].flatten())
        for j in range(3):
            if j!=dim:
                y[i]/=np.shape(e)[j]
                
    return x,y,e,lv,coord

def plot_plane_averaged_density(ifile,filetype='LOCPOT',read_data_from_file=None):
    x,y=calc_plane_averaged_density(ifile,filetype,read_data_from_file=read_data_from_file)[:2]
    plt.figure()
    plt.plot(x,y)
    if filetype=='LOCPOT':
        plt.ylabel('electrostatice potential / eV')
    elif 'CHG' in filetype:
        plt.ylabel('charge density / # electrons $A^{-3}$')
    plt.xlabel('position / $\AA$')
    plt.show()
