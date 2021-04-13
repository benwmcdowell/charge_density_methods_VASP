from numpy import array, zeros, shape, dot
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from lib import parse_CHGCAR
'''
def find_bond_vectors(ifile,from_type,to_type,num_bonds):
    e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    dim=shape(e)
    
    periodic_coord=[]
    for l in range(len(coord)):
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    periodic_coord.append(coord[l]+lv[0]*i+lv[1]*j+lv[2]*k)
    periodic_coord=array(periodic_coord)
    periodic_density=zeros(tuple([i*3 for i in shape(e)]))
    periodic_pos=zeros((shape(e)[0],shape(e)[1],shape(e)[2],3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                periodic_density[i*dim[0]:(i+1)*dim[0],j*dim[1]:(j+1)*dim[1],k*dim[2]:(k+1)*dim[2]]+=e
                periodic_pos[i*dim[0]:(i+1)*dim[0],j*dim[1]:(j+1)*dim[1],k*dim[2]:(k+1)*dim[2],:]+=pos+(i-1)*lv[0]+(j-1)*lv[1]+(k-1)*lv[2]

    for i in range(len(atomtypes)):
        if atomtypes[i]==from_type:
            start_indices=[sum(atomnums[:i])+j for j in range(atomnums[i])]
        if atomtypes[i]==to_type:
            end_indices=i
    
    start_coord=[]
    end_coord=[]
    for i in start_indices:
        start_coord.append(coord[i])
        mindiff=[max([norm(lv[i]) for i in range(3)]) for j in range(num_bonds)]
        temp_end=[zeros((3)) for i in range(num_bonds)]
        for j in range(sum(atomnums[:end_indices])*27,sum(atomnums[:end_indices+1])*27):
            tempdiff=norm(periodic_coord[j]-coord[i])
            if tempdiff<max(mindiff) and tempdiff!=0:
                temp_index=mindiff.index(max(mindiff))
                mindiff[temp_index]=tempdiff
                temp_end[temp_index]=periodic_coord[j]
        end_coord.append(temp_end)
'''

class charge_density:
    def __init__(self,ifile):
        self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        self.dim=array(shape(self.e))
    
    def interpolate_density(self,start_coord,end_coord,**args):
        if 'npts' in args:
            npts=args['npts']
        else:
            npts=min(self.dim)
            
        if 'direct' not in args:
            end_coord=dot(end_coord,self.lv)
            start_coord=dot(start_coord,self.lv)
            
        self.edensity=zeros(npts)
        bond_vector=end_coord-start_coord
        pos=array([start_coord+bond_vector*i/(npts-1) for i in range(npts)])
        for i in range(npts):
            temp_pos=[round(j) for j in dot(pos[i],inv(self.lv))*self.dim]
            for j in range(3):
                while temp_pos[j]>=self.dim[j] or temp_pos[j]<0:
                    if temp_pos[j]>=self.dim[j]:
                        temp_pos[j]-=self.dim[j]
                    if temp_pos[j]<0:
                        temp_pos[j]+=self.dim[j]
            self.edensity[i]+=self.e[temp_pos[0]][temp_pos[1]][temp_pos[2]]
        
        self.distance=array([norm(pos[i]-pos[0]) for i in range(npts)])
        
    def plot_slice(self,title):
        plt.figure()
        plt.plot(self.distance,self.edensity)
        plt.xlabel('position / $\AA$')
        plt.ylabel('# of electrons')
        plt.show()