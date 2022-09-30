import numpy as np
import os

def rename_PARCHG(fp):
    os.chdir(fp)
    listdir=os.listdir(fp)
    ef=parse_doscar('./DOSCAR')
    eigenval=parse_eigenval('./EIGENVAL')[0]
    for i in listdir:
        if 'PARCHG' in i:
            nband=int(i.split('.')[1])-1
            os.rename(i,'.'.join([i,str(eigenval[nband-1]-ef)]))
            
    print('eigenenergies added to PARCHG files')

def parse_eigenval(ifile):
    with open(ifile) as file:
        for i in range(6):
            line=file.readline().split()
        kpts=int(line[1])
        nstates=int(line[2])
        eigenval=np.zeros((nstates*kpts))
        for j in range(kpts):
            for i in range(2):
                file.readline()
            for i in range(nstates):
                line=file.readline().split()
                if len(line)==5:
                    #if calc is spin polarized
                   eigenval[i+j*nstates]+=(float(line[1])+float(line[2]))/2
                elif len(line)==3:
                    eigenval[i+j*nstates]+=float(line[1])
        
    return eigenval,nstates

#reads DOSCAR
def parse_doscar(filepath):
    with open(filepath,'r') as file:
        line=file.readline().split()
        for i in range(5):
            line=file.readline().split()
        ef=float(line[3])
    return ef

