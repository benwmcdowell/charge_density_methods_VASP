from numpy import array, zeros, shape, dot, average
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT, write_CHGCAR

class charge_density:
    def __init__(self,ifile,**args):
        #optional arguments for filetype are CHGCAR or LOCPOT -- CHGCAR is default
        if 'filetype' in args:
            self.filetype=args['filetype']
        else:
            self.filetype='CHGCAR'
        if self.filetype=='CHGCAR':
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        else:
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_LOCPOT(ifile)
        self.dim=array(shape(self.e))
        self.distance=[]
        self.edensity=[]
        
    def subract_ref(self,ref):
        if self.filetype=='CHGCAR':
            e=parse_CHGCAR(ref)[0]
        else:
            e=parse_LOCPOT(ref)[0]
        self.e-=e
    
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
        self.distance.append(array([norm(bond_vector)*i/(self.npts-1) for i in range(self.npts)]))
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
    def find_bond_vectors(self,from_type,to_type,num_bonds,**args):
        if 'nums' in args:
            nums=args['nums']
        else:
            nums=[]
        start_indices=[]
        for i in range(len(self.atomtypes)):
            if self.atomtypes[i]==from_type:
                for j in range(self.atomnums[1]):
                    if j+1 in nums or len(nums)==0:
                        start_indices.append(j+sum(self.atomnums[:i]))
            if self.atomtypes[i]==to_type:
                end_indices=i
        
        start_coord=[]
        end_coord=[]
        for i in start_indices:
            start_coord.append(self.coord[i])
            mindiff=[max([norm(self.lv[i]) for i in range(3)]) for j in range(num_bonds)]
            temp_end=[zeros((3)) for i in range(num_bonds)]
            for j in range(sum(self.atomnums[:end_indices]),sum(self.atomnums[:end_indices+1])):
                for k in range(-1,2):
                    for l in range(-1,2):
                        for m in range(-1,2):
                            disp=self.lv[0]*k+self.lv[1]*l+self.lv[2]*m
                            tempdiff=norm(self.coord[j]+disp-self.coord[i])
                            if tempdiff<max(mindiff) and tempdiff!=0:
                                temp_index=mindiff.index(max(mindiff))
                                mindiff[temp_index]=tempdiff
                                temp_end[temp_index]=self.coord[j]+disp
            end_coord.append(temp_end)
            
        for i in range(len(start_coord)):
            for j in range(len(end_coord[i])):
                self.interpolate_density(start_coord[i],end_coord[i][j],direct=False)
        
        self.start=start_coord
        self.end=end_coord
        self.edensity=average(self.edensity, axis=0)
        self.distance=average(self.distance, axis=0)
        
    def plot_density(self, **args):
        if len(shape(self.edensity))>1:
            self.edensity=average(self.edensity, axis=0)
            self.distance=average(self.distance, axis=0)
        plt.figure()
        plt.plot(self.distance,self.edensity)
        plt.xlabel('position / $\AA$')
        if self.filetype=='CHGCAR':
            plt.ylabel('charge density / electrons $\AA^{-3}$')
        else:
            plt.ylabel('electrostatic potential / eV')
        if 'title' in args:
            plt.title(args['title'])
        plt.show()
    
    def min_density(self):
        print('minimum electron density along bond is: {} electrons per cubic Angstrom'.format(min(self.edensity)))
        
#overlays density slices from multiple class instances of charge_density()
#useful for comparing charge density slices along bonds for different systems
def overlay_densities(distances,densities,labels):
    for i in range(len(distances)):
        if len(shape(distances[i]))>1:
            distances[i]=average(distances[i], axis=0)
            densities[i]=average(densities[i], axis=0)
    plt.figure()
    for i in range(len(distances)):
        plt.plot(distances[i]-distances[i][-1]/2,densities[i],label=labels[i])
    plt.xlabel('position / $\AA$')
    plt.ylabel('electrons $\AA^{-3}$')
    plt.legend()
    plt.show()

