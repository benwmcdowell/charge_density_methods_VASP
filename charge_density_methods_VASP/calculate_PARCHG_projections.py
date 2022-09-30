import numpy as np
import os

from lib import parse_CHGCAR, parse_LOCPOT

def calc_PARCHG_projections(ref,fp):
    os.chdir(fp)
    listdir=os.listdir(fp)
    eref=parse_CHGCAR(ref)[0]
    bands=[]
    overlap=[]
    for i in listdir:
        if 'PARCHG' in i:
            bands.append(int(i.split('.')[1])-1)
            etemp=parse_CHGCAR(i)[0]
            overlap.append(np.dot(eref,etemp)/np.linalg.norm(eref)/np.linalg.norm(etemp))
            
    return bands,overlap