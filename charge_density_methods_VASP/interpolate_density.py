from numpy import array, zeros, shape, dot, average
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from lib import parse_CHGCAR

class charge_density:
    def __init__(self,ifile):
        self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        self.dim=array(shape(self.e))
        self.distance=[]
        self.edensity=[]
    
    def interpolate_density(self,start_coord,end_coord,**args):
        if 'npts' in args:
            self.npts=args['npts']
        else:
            self.npts=min(self.dim)
            
        if 'direct' not in args:
            end_coord=dot(end_coord,self.lv)
            start_coord=dot(start_coord,self.lv)
            
        bond_vector=end_coord-start_coord
        pos=array([start_coord+bond_vector*i/(self.npts-1) for i in range(self.npts)])        
        self.distance.append(array([norm(bond_vector) for i in range(self.npts)]))
        self.edensity.append(zeros(self.npts))
        
        for i in range(self.npts):
            temp_pos=[round(j) for j in dot(pos[i],inv(self.lv))*self.dim]
            for j in range(3):
                while temp_pos[j]>=self.dim[j] or temp_pos[j]<0:
                    if temp_pos[j]>=self.dim[j]:
                        temp_pos[j]-=self.dim[j]
                    if temp_pos[j]<0:
                        temp_pos[j]+=self.dim[j]
            self.edensity[-1][i]+=self.e[temp_pos[0]][temp_pos[1]][temp_pos[2]]
    
    #by specifying the atom types 'to' and 'from' the atomic coordinates are automatically calculated
    #for example, using 'Ag' as to_type and 'Ag' as from_type will slice the electron density along the vectors between num_bonds nearest neighbors
    #just average the arrays in self.edensity and self.distance to get the average behavior of the charge distribution amongst num_bonds nearest neighbors
    def find_bond_vectors(self,from_type,to_type,num_bonds):
        periodic_coord=[]
        for l in range(len(self.coord)):
            for i in range(-1,2):
                for j in range(-1,2):
                    for k in range(-1,2):
                        periodic_coord.append(self.coord[l]+self.lv[0]*i+self.lv[1]*j+self.lv[2]*k)
        periodic_coord=array(periodic_coord)
    
        for i in range(len(self.atomtypes)):
            if self.atomtypes[i]==from_type:
                start_indices=[sum(self.atomnums[:i])+j for j in range(self.atomnums[i])]
            if self.atomtypes[i]==to_type:
                end_indices=i
        
        start_coord=[]
        end_coord=[]
        for i in start_indices:
            start_coord.append(self.coord[i])
            mindiff=[max([norm(self.lv[i]) for i in range(3)]) for j in range(num_bonds)]
            temp_end=[zeros((3)) for i in range(num_bonds)]
            for j in range(sum(self.atomnums[:end_indices])*27,sum(self.atomnums[:end_indices+1])*27):
                tempdiff=norm(periodic_coord[j]-self.coord[i])
                if tempdiff<max(mindiff) and tempdiff!=0:
                    temp_index=mindiff.index(max(mindiff))
                    mindiff[temp_index]=tempdiff
                    temp_end[temp_index]=periodic_coord[j]
            end_coord.append(temp_end)
            
        for i in range(len(start_coord)):
            for j in range(len(end_coord[i])):
                self.interpolate_density(start_coord[i],end_coord[i][j],direct=False)
                
        self.edensity=average(self.edensity, axis=0)
        self.distance=average(self.distance, axis=0)
        
    def plot_density(self, **args):
        plt.figure()
        plt.plot(self.distance,self.edensity)
        plt.xlabel('position / $\AA$')
        plt.ylabel('# of electrons')
        if 'title' in args:
            plt.title(args['title'])
        plt.show()
