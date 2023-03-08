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
        self.x_slices=[]
        self.z_slices=[]
            
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
                
        return z,pos_dim
    
    def write_density(self,ofile):
        np.save(ofile,self.e)
        
    def shift_coord(self,shift,z,direct=True):
        if not direct:
            shift=np.dot(shift,np.linalg.inv(self.lv)[:2,:2])
        
        shift=np.array([round(shift[i]*np.shape(z)[i]) for i in range(2)])
        
        new_z=np.zeros(np.shape(z))
        for i in range(np.shape(z)[0]):
            ishift=i+shift[0]
            while ishift>=np.shape(z)[0] or ishift<0:
                if ishift>=np.shape(z)[0]:
                    ishift-=np.shape(z)[0]
                if ishift<0:
                    ishift+=np.shape(z)[0]
            for j in range(np.shape(z)[1]):
                jshift=j+shift[1]
                while jshift>=np.shape(z)[1] or jshift<0:
                    if jshift>=np.shape(z)[1]:
                        jshift-=np.shape(z)[1]
                    if jshift<0:
                        jshift+=np.shape(z)[1]
                try:
                    new_z[ishift,jshift]=z[i,j]
                except IndexError:
                    print(ishift,jshift,i,j)
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
            shift=np.dot(shift,self.lv[:2,:2])
            
        self.z=z
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        map_data=self.ax_main.pcolormesh(self.xy[:,:,0],self.xy[:,:,1],z,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
        self.fig_main.colorbar(map_data)
        for i in plot_atoms:
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break
            tempvar=np.array([self.coord[i][pos_dim[0]]+shift[0],self.coord[i][pos_dim[1]]+shift[1]])
            tempvar=np.dot(tempvar,np.linalg.inv(self.lv[:2,:2]))
            for k in range(2):
                while tempvar[k]>1 or tempvar[k]<0:
                    if tempvar[k]>1:
                        tempvar[k]-=1
                    if tempvar[k]<0: 
                        tempvar[k]+=1
            tempvar=np.dot(tempvar,self.lv[:2,:2])
            self.ax_main.scatter(tempvar[0],tempvar[1],color=colors[j],s=sizes[j])
        patches=[]
        if len(plot_atoms)>0:
            for i in range(len(self.atomtypes)):
                patches.append(Patch(color=colors[i],label=self.atomtypes[i]))
                
        self.ax_main.set(xlabel='position / $\AA$', ylabel='position / $\AA$')
        self.fig_main.legend(handles=patches)
        self.ax_main.set_aspect('equal')
        self.fig_main.show()

    def plot_1d_slice(self,axis,pos,direct=True,fit=True,nperiods=1,nperiods_short=0,print_fit_params=False,periodic_fit=True,center_x=False):
        if not hasattr(self,'fig_slice'):
            self.fig_slice,self.ax_slice=plt.subplots(1,1,tight_layout=True)
        def model_cosine(x,a,k,phi,y0):
            y=y0+a*np.cos(2*np.pi*k*x+phi)
            return y
        
        def model_cosine_sum(x,a1,a2,k1,k2,phi1,phi2,y0):
            y=y0+a1*np.cos(2*np.pi*k1*x+phi1)+a2*np.cos(2*np.pi*k2*x+phi2)
            return y
        
        if type(axis)==int:
            if direct:
                pos=round(pos*np.shape(self.e)[1-axis])
            tempx=self.xy.take(pos,axis=1-axis)
            tempx-=tempx[0]
            tempx=np.array([np.linalg.norm(i) for i in tempx])
                
            tempy=self.z.take(pos,axis=1-axis)
            
        #for the case where 'axis' is a list of atoms to slice through
        elif axis==None:
            pos=np.array([self.coord[i,:2]for i in pos])
            tempx=np.array([np.linspace(pos[0,i],pos[1,i],np.min(np.shape(self.e))) for i in range(2)])
            tempx=np.array([[tempx[0,i],tempx[1,i]] for i in range(np.min(np.shape(self.e)))])
            for i in range(len(tempx)):
                tempx[i]=np.dot(tempx[i],np.linalg.inv(self.lv[:2,:2]))
            tempy=np.array([self.z[round(np.shape(self.xy)[0]*tempx[i,0]) ,round(np.shape(self.xy)[1]*tempx[i,1]) ] for i in range(len(tempx))])
            for i in range(len(tempx)):
                tempx[i]=np.dot(tempx[i],self.lv[:2,:2])
                tempx[i]=np.linalg.norm(tempx[i])
            tempx-=np.min(tempx)
            
        if center_x:
            tempx-=np.average(tempx)
            
        if fit:
            if fit=='simple':
                bounds=[[0,0,-np.max(tempx)*2*np.pi,-np.inf],[np.inf,np.inf,np.max(tempx)*2*np.pi,np.inf]]
                p0=[np.max(tempy)-np.min(tempy),nperiods/np.max(tempx),tempx[np.argmax(tempy)],np.average(tempy)]
            else:
                bounds=[[0,0,0,0,-np.max(tempx)*2*np.pi,-np.max(tempx)*2*np.pi,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.max(tempx)*2*np.pi,np.max(tempx)*2*np.pi,np.inf]]
                if nperiods_short==0:
                    first_peak=False
                    for i in range(5,len(tempy)-5):
                        if np.argmax(tempy[i-5:i+6])==6:
                            if first_peak:
                                spacing=tempx[i]-tempx[first_peak]
                                break
                            else:
                                first_peak=i
                else:
                    spacing=np.max(tempx)/nperiods_short
                            
                p0=[np.max(tempy)-np.min(tempy),np.max(tempy)-np.min(tempy),nperiods/np.max(tempx),1/spacing,tempx[np.argmax(tempy)],tempx[np.argmax(tempy)],np.average(tempy)]
            
            if periodic_fit:
                nperiods*=5
                periodic_x=[tempx[i]+j*(np.max(tempx)+(tempx[1]-tempx[0])) for i in range(len(tempx)) for j in range(nperiods)]
                periodic_y=[tempy[i] for i in range(len(tempy)) for j in range(nperiods)]
            else:
                periodic_x=tempx
                periodic_y=tempy
            if fit=='simple':
                popt,pcov=curve_fit(model_cosine,periodic_x,periodic_y,p0=p0,bounds=bounds)
                fit_y=model_cosine(tempx,popt[0],popt[1],popt[2],popt[3])
                self.ax_slice.plot(tempx,fit_y)
            else:
                popt,pcov=curve_fit(model_cosine_sum,periodic_x,periodic_y,p0=p0,bounds=bounds)
                fit_y=model_cosine_sum(tempx,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
            pcov=np.sqrt(np.diag(pcov))
            
            self.ax_slice.plot(tempx,fit_y)
            
            self.fit_params.append(popt)
            
            if print_fit_params and fit=='simple':
                print('A = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[1],pcov[1],popt[2],pcov[2],popt[3],pcov[3]))
            elif print_fit_params:
                print('periodic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\natomic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[2],pcov[2],popt[4],pcov[4],popt[1],pcov[1],popt[3],pcov[3],popt[5],pcov[5],popt[6],pcov[6]))
            
        self.x_slices.append(tempx)
        self.z_slices.append(tempy)
        if type(axis)==int:
            tempdata=self.ax_main.plot([self.xy.take(pos,axis=1-axis)[i][0] for i in [0,-1]],[self.xy.take(pos,axis=1-axis)[i][1] for i in [0,-1]])
        elif axis==None:
            tempdata=self.ax_main.plot([pos[0,0],pos[1,0]],[pos[0,1],pos[1,1]])
            
        color=tempdata[0].get_color()
        self.ax_slice.plot(tempx,tempy,c=color)
        
        if self.filetype=='LOCPOT':
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='eV')
        else:
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='charge density / # electrons $/AA^{-3}$')
        self.fig_slice.show()