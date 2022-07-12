import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
import matplotlib.ticker as mplt
import numpy as np
import os

class bader_charges_2d():
    def __init__(self,fp):
        self.fp=fp
        os.chdir(self.fp)
        self.x,self.y,self.z,self.charge=parse_bader_ACF('./ACF.dat')[:4]
        self.numvalence=parse_potcar('./POTCAR')
        self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./POSCAR')[:4]
        
    def calc_net_charges(self):
        self.net_charge=[]
        for i in range(len(self.charge)):
            for j in range(len(self.atomnums)):
                if i < sum(self.atomnums[:j+1]):
                    break
            self.net_charge.append(self.charge[i]-self.numvalence[j])
        
#volume to plot is a list of 3,2 arrays, where each is a min-max pair of x,y,z coordinates
    def plot_atom_charges(self,volume_to_plot,s=200,direct=False):
        self.fig,self.ax=plt.subplots(1,1)
        atoms_to_plot=[]
        for i in range(len(self.charge)):
            for j in volume_to_plot:
                if direct:
                    for k in range(2):
                        j[:,k]=np.dot(j[:,k],self.lv)
                for k in range(3):
                    if self.coord[i,k]<np.max(j[k]) and self.coord[i,k]>np.min(j[k]) and i not in atoms_to_plot:
                        pass
                    else:
                        break
                else:
                    atoms_to_plot.append(i)
            
        ref=np.max(abs(np.array([self.net_charge[i] for i in atoms_to_plot])))
        self.cnorm=mplc.Normalize(vmin=-ref,vmax=ref)
        colors=[mplcm.bwr(self.cnorm(self.net_charge[i])) for i in atoms_to_plot]
        sizes=[s for i in range(len(atoms_to_plot))]
        x=[self.x[i] for i in atoms_to_plot]
        y=[self.y[i] for i in atoms_to_plot]
        self.ax.scatter(x,y,color=colors,s=sizes)
            
        cbar=self.fig.colorbar(mplcm.ScalarMappable(norm=self.cnorm,cmap=mplcm.bwr))
        cbar.set_label('net electron count')
        cbar.ax.yaxis.set_major_formatter(mplt.FormatStrFormatter('%+.3f'))
        self.ax.set_aspect('equal')
        self.fig.show()
        
#reads the ACF file output by Bader analysis and returns contents
def parse_bader_ACF(ifile):
    with open(ifile, 'r') as file:
        x=[]
        y=[]
        z=[]
        charge=[]
        min_dist=[]
        vol=[]
        for i in range(2):
            line=file.readline()
        while True:
            line=file.readline().split()
            try:
                x.append(float(line[1]))
                y.append(float(line[2]))
                z.append(float(line[3]))
                charge.append(float(line[4]))
                min_dist.append(float(line[5]))
                vol.append(float(line[6]))
            #stops reading the file when '--------' is reached
            except IndexError:
                break
    
    return x, y, z, charge, min_dist, vol

#reads the number of valence electrons for each atom type for the POTCAR file
def parse_potcar(ifile):
    with open(ifile, 'r') as file:
        numvalence=[]
        counter=0
        while True:
            line=file.readline()
            if not line:
                break
            if counter==1:
                numvalence.append(float(line.split()[0]))
            if 'End of Dataset' in line:
                counter=-1
            counter+=1
        
    return numvalence

def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=np.array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=np.array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=np.dot(coord[i],latticevectors)
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums