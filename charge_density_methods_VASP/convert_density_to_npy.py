import numpy as np
import sys
import getopt
import os
from pathlib import Path

def convert_density_to_npy(ifile,ofile,ref=False,filetype='LOCPOT'):
    if filetype=='LOCPOT':
        e=parse_LOCPOT(ifile)[0]
    else:
        e=parse_CHGCAR(ifile)[0]
    
    if ref:
        for i in ref:
            if filetype=='LOCPOT':
                tempe=parse_LOCPOT(i[0])[0]
            else:
                tempe=parse_CHGCAR(i[0])[0]
            e-=tempe*i[1]
    
    np.save(ofile,e)
    
#reads the total charge density from a CHGCAR file
def parse_CHGCAR(ifile, **args):
    if 'scale' in args:
        rescale=False
    else:
        rescale=True
    #reads atomic positions and lattice vectors
    lv=np.zeros((3,3))
    with open(ifile,'r') as chgcar:
        for i in range(8):
            line=chgcar.readline().split()
            if i==1:
                sf=float(line[0])
            if i>1 and i<5:
                for j in range(3):
                    lv[i-2][j]=float(line[j])*sf
            if i==5:
                atomtypes=line
            if i==6:
                atomnums=line
                for j in range(len(atomnums)):
                    atomnums[j]=int(atomnums[j])
            if i==7:
                mode=line[0]
        coord=np.zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=np.dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        e=np.zeros((dim[0],dim[1],dim[2]))
        searching=True
        while searching:
            line=chgcar.readline().split()
            for i in line:
                e[x][y][z]=float(i)
                x+=1
                if x==dim[0]:
                    x=0
                    y+=1
                if y==dim[1]:
                    y=0
                    z+=1
                if z==dim[2]:
                    searching=False
                    break
    if rescale:
        print('charge density values rescaled to electrons per cubic Angstrom')
        vol=np.dot(np.cross(lv[0],lv[1]),lv[2])
        e/=vol
    
    return e, lv, coord, atomtypes, atomnums

#reads the total charge density from a CHGCAR file
def parse_LOCPOT(ifile):
    #reads atomic positions and lattice vectors
    lv=np.zeros((3,3))
    with open(ifile,'r') as chgcar:
        for i in range(8):
            line=chgcar.readline().split()
            if i==1:
                sf=float(line[0])
            if i>1 and i<5:
                for j in range(3):
                    lv[i-2][j]=float(line[j])*sf
            if i==5:
                atomtypes=line
            if i==6:
                atomnums=line
                for j in range(len(atomnums)):
                    atomnums[j]=int(atomnums[j])
            if i==7:
                mode=line[0]
        coord=np.zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=np.dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        pot=np.zeros((dim[0],dim[1],dim[2]))
        searching=True
        counter=0
        while searching:
            line=chgcar.readline().split()
            if not line:
                break
            for i in line:
                pot[x][y][z]+=float(i)
                x+=1
                if x==dim[0]:
                    x=0
                    y+=1
                    if y==dim[1]:
                        y=0
                        z+=1
                        if z==dim[2]:
                            z=0
                            counter+=1
                            if counter==2:
                                searching=False
                                break
                            for i in range(round(sum(atomnums)/5)+1):
                                line=chgcar.readline()
        
    if counter==2:
        pot/=2.0
    return pot, lv, coord, atomtypes, atomnums

if __name__=='__main__':
    ref=[]
    filetype='LOCPOT'
    try:
        opts,args=getopt.getopt(sys.argv[1:],'i:o:r:t:',['ifile=','ofile=','ref=','type='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-i','--ifile']:
            ifile=Path.cwd() / j
            ifile=ifile.resolve()
        if i in ['-o','--ofile']:
            ofile=Path.cwd() / j
            ofile=ofile.resolve()
        if i in ['-r','--ref']:
            j=j.split(',')
            if len(j)>1:
                sf=float(j[1])
                j=j[0]
            else:
                sf=1.0
                j=j[0]
            path=Path.cwd() / j
            ref.append([path.resolve(),sf])
        if i in ['-t','--type']:
            filetype=j
    if len(ref)==0:
        ref=False
        
    convert_density_to_npy(ifile,ofile,ref=ref,filetype=filetype)