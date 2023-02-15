import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lib import parse_CHGCAR, parse_LOCPOT, parse_doscar, parse_poscar

class density_data:
    def __init__(self,ifile,**args):
        if 'filetype' in args:
            filetype=args['filetype']
        else:
            filetype='LOCPOT'
        
        if filetype=='LOCPOT':
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_LOCPOT(ifile)
        try:
            if 'CHG' in filetype:
                self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        except TypeError:
            pass
        
        if filetype==None:
            self.npts=1000
            self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar(ifile)[:4]
            self.e=np.zeros((self.npts,self.npts,1))
            
        self.normdiff=False
        if 'ref' in args:
            for i in args['ref']:
                if filetype=='LOCPOT':
                    tempvar=parse_LOCPOT(i)[0]
                else:
                    tempvar=parse_CHGCAR(i)[0]
                self.e-=tempvar
            self.normdiff=True
                
        if 'eref' in args:
            self.ef=parse_doscar(args['eref'][2])
            self.e-=self.ef
            
    def slice_density(self,pos,**args):
        if 'dim' in args:
            dim=args['dim']
        else:
            dim=2
        
        if 'direct' in args:
            pos=np.linalg.norm(np.dot(pos,self.lv[dim]))
        
        if 'tol' in args:
            tol=round(args['tol']/np.linalg.norm(self.lv[dim])*np.shape(self.e)[dim])
        else:
            tol=0
        
        pos_dim=[]
        for i in range(3):
            if i!=dim:
                pos_dim.append(i)
        
        self.xy=np.zeros((np.shape(self.e)[pos_dim[0]],np.shape(self.e)[pos_dim[1]],2))
        for i in range(len(self.xy)):
            for j in range(len(self.xy[i])):
                self.xy[i][j]+=self.lv[pos_dim[0]][:2]*i/(len(self.xy[i])+1)+self.lv[pos_dim[1]][:2]*j/(len(self.xy[j])+1)
            
        pos=round(pos*np.shape(self.e)[dim]/np.linalg.norm(self.lv[dim]))
        z=np.zeros((np.shape(self.e)[pos_dim[0]],np.shape(self.e)[pos_dim[1]]))
        for i in range(-tol,tol+1):
            if dim==0:
                z+=self.e[pos,:,:]/(2*tol+1)
            if dim==1:
                z+=self.e[:,pos,:]/(2*tol+1)
            if dim==2:
                z+=self.e[:,:,pos]/(2*tol+1)
                
        return z,pos_dim
        
    def plot_2d_density(self,pos,cmap='jet',**args):
        plot_atoms=[]
        if 'overlay_atoms' in args:
            ranges=args['overlay_atoms']
            for i in range(sum(self.atomnums)):
                for j in range(3):
                    if self.coord[i][j] > max(ranges[j]) or self.coord[i][j] < min(ranges[j]):
                        break
                else:
                    plot_atoms.append(i)
        if 'atom_sizes' in args:
            sizes=args['atom_sizes']
        else:
            sizes=[800 for i in range(len(self.atomnums))]
            
        if 'atom_colors' in args:
            colors=args['atom_colors']
        else:
            colors=['black' for i in range(len(self.atomnums))]
            
        z,pos_dim=self.slice_density(pos)
        
        if self.normdiff:
            vmin=-1*np.max([abs(np.min(z)),abs(np.max(z))])
            vmax=np.max([abs(np.min(z)),abs(np.max(z))])
        else:
            vmin=np.min(z)
            vmax=np.max(z)
            
        plt.figure()
        plt.pcolormesh(self.xy[:,:,0],self.xy[:,:,1],z,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
        plt.colorbar()
        for i in plot_atoms:
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break
            plt.scatter(self.coord[i][pos_dim[0]],self.coord[i][pos_dim[1]],color=colors[j],s=sizes[j])
        patches=[]
        if len(plot_atoms)>0:
            for i in range(len(self.atomtypes)):
                patches.append(Patch(color=colors[i],label=self.atomtypes[i]))
                
        plt.xlabel('position / $\AA$')
        plt.ylabel('position / $\AA$')
        plt.legend(handles=patches)
        plt.axes().set_aspect('equal')
        plt.show()
