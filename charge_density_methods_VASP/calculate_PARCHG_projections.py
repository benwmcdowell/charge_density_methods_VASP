import numpy as np
import os

from lib import parse_CHGCAR, parse_LOCPOT

def calc_PARCHG_projections(ref,fp):
    os.chdir(fp)
    listdir=os.listdir(fp)
    eref=parse_CHGCAR(ref)[0]
    bands=[]
    overlaps=[]
    for i in listdir:
        if 'PARCHG' in i:
            bands.append(int(i.split('.')[1])-1)
            etemp=parse_CHGCAR(i)[0]
            overlaps.append(np.linalg.norm(eref*etemp)/np.linalg.norm(etemp))
            
    overlaps=np.array(overlaps)
    overlaps-=np.min(overlaps)
    overlaps/=np.max(overlaps)
    bands=np.array(bands)        
    
    return bands,overlaps