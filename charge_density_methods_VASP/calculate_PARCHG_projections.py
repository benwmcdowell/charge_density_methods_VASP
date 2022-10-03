import numpy as np
import os

from lib import parse_CHGCAR, parse_LOCPOT

def calc_PARCHG_projections(ref,fp):
    os.chdir(fp)
    listdir=os.listdir(fp)
    eref,lv=parse_CHGCAR(ref)[:2]
    npts=np.shape(eref)
    bands=[]
    overlaps=[]
    average_zpos=[]
    zmask=np.zeros(npts)
    energies=[]
    for i in range(npts[2]):
        zmask[:,:,i]=i/(npts[2]-1)*np.linalg.norm(lv[2])
    for i in listdir:
        if 'PARCHG' in i:
            bands.append(int(i.split('.')[1])-1)
            etemp=parse_CHGCAR(i)[0]
            energies.append(float('.'.join(i.split('.')[-2:])))
            overlaps.append(np.linalg.norm(eref*etemp)/np.linalg.norm(etemp))
            average_zpos.append(np.linalg.norm(zmask*etemp/np.linalg.norm(etemp)))
            
    overlaps=np.array(overlaps)
    overlaps-=np.min(overlaps)
    overlaps/=np.max(overlaps)
    bands=np.array(bands)        
    
    return bands,overlaps,average_zpos,energies