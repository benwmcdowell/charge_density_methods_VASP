import numpy as np
import matplotlib.pyplot as plt
import os

from lib import parse_CHGCAR, parse_LOCPOT, parse_doscar

def calc_density(ifile,atoms,filetype='CHGCAR',slice_path='default',**args):
    
    if filetype=='LOCPOT':
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    else:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    npts=np.shape(e)
    
    x=np.array([i/(npts[2]-1)*np.linalg.norm(lv[2]) for i in range(npts[2])])
    y=[np.zeros(npts[2]) for i in range(len(atoms))]
    bond_partners=[]
    for i in range(len(atoms)):
        pos=coord[atoms[i]-1,:2]
        pos=np.dot(pos,np.linalg.inv(lv[:2,:2]))
        pos*=np.array(npts[:2])-1
        pos=pos.astype(int)
        if slice_path=='default':
            y[i]=e[pos[0],pos[1],:]
        elif slice_path=='bond':
            bond_partners.append([])
            tempvar=[]
            for j in range(len(coord)):
                if atoms[i]-1==j:
                    tempvar.append(np.max([np.linalg.norm(k) for k in lv]))
                else:
                    tempvar.append(np.linalg.norm(coord[i-1]-coord[j]))
            nearest_dist=np.min(tempvar)
            for j in tempvar:
                if j<=nearest_dist+0.1:
                    bond_partners[-1].append(tempvar.index(j))
    if slice_path=='bond':
        for i in range(len(atoms)):
            for j in bond_partners[i]:
                a=(coord[atoms[i]-1,:2]-coord[j,:2])/(coord[atoms[i]-1,2]-coord[j,2])
                b=coord[atoms[i]-1,:2]-a*coord[atoms[i]-1,2]
                for k in range(npts[2]):
                    temppos=b+k*np.linalg.norm(lv[2])/(npts[2]-1)*a
                    temppos=np.dot(temppos,np.linalg.inv(lv[:2,:2]))
                    for l in range(2):
                        while temppos[l]>1 or temppos[l]<0:
                            if temppos[l]>1:
                                temppos[l]-=1
                            if temppos[l]<0:
                                temppos[l]+=1
                    temppos*=np.array(npts[:2])-1
                    temppos=temppos.astype(int)
                    y[i][k]=e[temppos[0],temppos[1],k]
                    
    return x,y,atoms,e,lv,coord

def plot_density(ifile,atoms,filetype='CHGCAR',linestyle='default',linecolors='default',lw='default',slice_path='default',overlay_pot=False,ref=False,overlay_levels=False,hide_legend=False):
    x,y,atoms=calc_density(ifile,atoms,filetype,slice_path=slice_path)[:3]
    if linestyle=='default':
        linestyle=['solid' for i in range(len(y))]
    if lw=='default':
        lw=[1 for i in range(len(y))]
        
    if ref:
        os.chdir(ref)
        ef=parse_doscar('./DOSCAR')[2]
        for i in range(len(y)):
            y[i]-=ef
            
    if overlay_pot:
        xpot,pot=calc_density(overlay_pot,atoms,filetype,slice_path=slice_path,filetype='LOCPOT')[:3]
        pot-=ef
    
    plt.figure()
    for i in range(len(y)):
        if linecolors=='default':
            plt.plot(x,y[i],label=atoms[i],linestyle=linestyle[i],lw=lw[i])
        else:
            plt.plot(x,y[i],label=atoms[i],linestyle=linestyle[i],lw=lw[i],color=linecolors[i])
    if filetype=='LOCPOT':
        if ref:
            plt.ylabel('E - $E_F$ / eV')
        else:
            plt.ylabel('electrostatic potential / eV')
    elif 'CHG' in filetype:
        plt.ylabel('charge density / # electrons $A^{-3}$')
        
    if overlay_levels:
        for i in overlay_levels:
            plt.plot(x,[i for j in range(len(x))],linestyle='dashed',color='black',lw=2)
        
    if not hide_legend:
        plt.legend()
    plt.xlabel('position / $\AA$')
    plt.show()
