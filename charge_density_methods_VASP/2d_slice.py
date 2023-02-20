import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
import os

from lib import parse_CHGCAR, parse_LOCPOT, parse_doscar, parse_poscar

class density_data:
    def __init__(self,ifile,read_data_from_file=False,**args):
        if 'filetype' in args:
            self.filetype=args['filetype']
        else:
            self.filetype='LOCPOT'
        
        if self.filetype=='LOCPOT' and not read_data_from_file:
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_LOCPOT(ifile)
        try:
            if 'CHG' in self.filetype:
                self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        except TypeError:
            pass
        
        if self.filetype==None and not read_data_from_file:
            self.npts=1000
            os.chdir(ifile)
            try:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./POSCAR')[:4]
            except FileNotFoundError:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./CONTCAR')[:4]
            self.e=np.zeros((self.npts,self.npts,1))
            
        if read_data_from_file:
            os.chdir(ifile)
            try:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./POSCAR')[:4]
            except FileNotFoundError:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./CONTCAR')[:4]
            self.e=np.load(read_data_from_file)
            
        self.normdiff=False
        if 'ref' in args:
            for i in args['ref']:
                if self.filetype=='LOCPOT':
                    tempvar=parse_LOCPOT(i)[0]
                else:
                    tempvar=parse_CHGCAR(i)[0]
                self.e-=tempvar
            self.normdiff=True
                
        if 'eref' in args:
            self.ef=parse_doscar(args['eref'][2])
            self.e-=self.ef
            
        self.fit_params=[]
            
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
                self.xy[i][j]+=self.lv[pos_dim[0]][:2]*i/(len(self.xy)+1)+self.lv[pos_dim[1]][:2]*j/(len(self.xy[i])+1)
            
        pos=round(pos*np.shape(self.e)[dim]/np.linalg.norm(self.lv[dim]))
        z=np.zeros((np.shape(self.e)[pos_dim[0]],np.shape(self.e)[pos_dim[1]]))
        for i in range(-tol,tol+1):
            if dim==0:
                z+=self.e[pos,:,:]/(2*tol+1)
            if dim==1:
                z+=self.e[:,pos,:]/(2*tol+1)
            if dim==2:
                z+=self.e[:,:,pos]/(2*tol+1)
                
        self.z=z
                
        return z,pos_dim
    
    def write_density(self,ofile):
        np.save(ofile,self.e)
        
    def shift_coord(self,shift,z,direct=True):
        if not direct:
            shift=np.dot(shift,np.linalg.inv(self.lv))
        
        shift=np.array([round(shift[i]*np.shape(self.e)[i]) for i in range(3)])
        
        new_z=np.zeros(np.shape(self.e)[:2])
        for i in range(np.shape(self.e)[0]):
            ishift=i+shift[0]
            while ishift>1 or ishift<0:
                if ishift>1:
                    ishift-=1
                if ishift<0:
                    ishift+=1
            for j in range(np.shape(self.e)[1]):
                jshift=j+shift[1]
                while jshift>1 or jshift<0:
                    if jshift>1:
                        jshift-=1
                    if jshift<0:
                        jshift+=1
                new_z[ishift,jshift]=z[i,j]
        return new_z
        
    def plot_2d_density(self,pos,cmap='jet',center_cbar=False,shift=np.zeros(2),direct_shift=True,**args):
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
            
        if self.normdiff:
            center_cbar=True
            
        z,pos_dim=self.slice_density(pos)
        
        if np.max(abs(shift))!=0:
            z=self.shift_coord(shift,z,direct=direct_shift)
        
        if center_cbar:
            vmin=-1*np.max([abs(np.min(z)),abs(np.max(z))])
            vmax=np.max([abs(np.min(z)),abs(np.max(z))])
        else:
            vmin=np.min(z)
            vmax=np.max(z)
            
        if direct_shift:
            shift=np.dot(shift,lv[:2,:2])
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        map_data=self.ax_main.pcolormesh(self.xy[:,:,0],self.xy[:,:,1],z,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
        self.fig_main.colorbar(map_data)
        for i in plot_atoms:
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break
            self.ax_main.scatter(self.coord[i][pos_dim[0]]+shift[0],self.coord[i][pos_dim[1]]+shift[1],color=colors[j],s=sizes[j])
        patches=[]
        if len(plot_atoms)>0:
            for i in range(len(self.atomtypes)):
                patches.append(Patch(color=colors[i],label=self.atomtypes[i]))
                
        self.ax_main.set(xlabel='position / $\AA$', ylabel='position / $\AA$')
        self.fig_main.legend(handles=patches)
        self.ax_main.set_aspect('equal')
        self.fig_main.show()

    def plot_1d_slice(self,axis,pos,direct=True,fit=True,nperiods=1,print_fit_params=False,periodic_fit=True):
        if not hasattr(self,'fig_slice'):
            self.fig_slice,self.ax_slice=plt.subplots(1,1,tight_layout=True)
        def model_cosine(x,a,k,phi,y0):
            y=y0+a*np.cos(2*np.pi*k*x+phi)
            return y
            
        if direct:
            pos=round(pos*np.shape(self.e)[1-axis])
        tempx=self.xy.take(pos,axis=1-axis)
        tempx-=tempx[0]
        tempx=np.array([np.linalg.norm(i) for i in tempx])
            
        tempy=self.z.take(pos,axis=1-axis)
            
        if fit:
            bounds=[[0,0,-np.max(tempx)*2*np.pi,-np.inf],[np.inf,np.inf,np.max(tempx)*2*np.pi,np.inf]]
            p0=[np.max(tempy)-np.min(tempy),nperiods/np.max(tempx),tempx[np.argmax(tempy)],np.average(tempy)]
            
            if periodic_fit:
                nperiods*=5
                periodic_x=[tempx[i]+j*(np.max(tempx)+(tempx[1]-tempx[0])) for i in range(len(tempx)) for j in range(nperiods)]
                periodic_y=[tempy[i] for i in range(len(tempy)) for j in range(nperiods)]
            else:
                periodic_x=tempx
                periodic_y=tempy
            popt,pcov=curve_fit(model_cosine,periodic_x,periodic_y,p0=p0,bounds=bounds)
            pcov=np.sqrt(np.diag(pcov))
            fit_y=model_cosine(tempx,popt[0],popt[1],popt[2],popt[3])
            self.ax_slice.plot(tempx,fit_y)
            
            self.fit_params.append(popt)
            
            if print_fit_params:
                print('A = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[1],pcov[1],popt[2],pcov[2],popt[3],pcov[3]))
            
        self.ax_main.plot([self.xy.take(pos,axis=1-axis)[i][0] for i in [0,-1]],[self.xy.take(pos,axis=1-axis)[i][1] for i in [0,-1]])
        
        self.ax_slice.plot(tempx,tempy)
        if self.filetype=='LOCPOT':
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='eV')
        else:
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='charge density / # electrons $/AA^{-3}$')
        self.fig_slice.show()