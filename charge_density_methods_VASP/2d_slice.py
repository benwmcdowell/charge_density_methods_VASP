import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
import os
from scipy.ndimage import gaussian_filter

from lib import parse_CHGCAR, parse_LOCPOT, parse_doscar, parse_poscar

class density_data:
    def __init__(self,ifile,read_data_from_file=False,dipole_correction=False,**args):
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
        
        if dipole_correction:
            def line(x,a,b):
                return a*x+b
            
            if 'dipole_correction_range' in args:
                dipole_correction_range=args['dipole_correction_range']
            else:
                dipole_correction_range=[0,np.linalg.norm(self.lv[2])]
                
            if 'normalize_dipole' not in args:
                normalize_dipole=True
            else:
                normalize_dipole=args['normalize_dipole']
                
            dipole_correction_pts=[round(dipole_correction_range[i]/np.linalg.norm(self.lv[2])*(np.shape(self.e)[2]-1)) for i in range(2)]
                
            tempy=np.zeros(dipole_correction_pts[1]-dipole_correction_pts[0])
            for i in range(np.shape(self.e)[0]):
                for j in range(np.shape(self.e)[1]):
                    tempy+=self.e[i,j,dipole_correction_pts[0]:dipole_correction_pts[1]]/np.shape(self.e)[0]/np.shape(self.e)[1]
            
            popt,pcov=curve_fit(line,np.linspace(dipole_correction_range[0],dipole_correction_range[1],dipole_correction_pts[1]-dipole_correction_pts[0]),tempy)
            fity=line(np.linspace(0,np.linalg.norm(self.lv[2]),np.shape(self.e)[2]),popt[0],popt[1])
            for i in range(np.shape(self.e)[0]):
                for j in range(np.shape(self.e)[1]):
                    self.e[i,j,:]-=fity
            
    def slice_density(self,pos,dim=2,**args):
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
    
    def slice_density_weighted(self,fp,dim=2):
        pos_dim=[]
        for i in range(3):
            if i!=dim:
                pos_dim.append(i)
                
        if 'CHG' in fp and '.npy' in fp:
            ref=np.load(fp)
        elif 'CHG' in fp:
            ref=parse_CHGCAR(fp)
            
        weighting=np.zeros(np.shape(ref)[dim])
        
        for i in range(len(weighting)):
            weighting[i]=np.sum(ref[:,:,i])
        
        weighting/=sum(weighting)
            
        self.xy=np.zeros((np.shape(self.e)[pos_dim[0]],np.shape(self.e)[pos_dim[1]],2))
        for i in range(len(self.xy)):
            for j in range(len(self.xy[i])):
                self.xy[i][j]+=self.lv[pos_dim[0]][:2]*i/(len(self.xy)+1)+self.lv[pos_dim[1]][:2]*j/(len(self.xy[i])+1)
            
        z=np.zeros((np.shape(self.e)[pos_dim[0]],np.shape(self.e)[pos_dim[1]]))
        for i in range(len(weighting)):
            z+=self.e[:,:,i]*weighting[i]
        
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
    
    def rescale_grid(self,new_xpts,new_ypts):
        newxy=np.zeros((new_xpts,new_ypts,2))
        newz=np.zeros((new_xpts,new_ypts))
        for i in range(new_xpts):
            i2=round(i*(np.shape(self.xy)[0]-1)/(new_xpts-1))
            for j in range(new_ypts):
                j2=round(j*(np.shape(self.xy)[1]-1)/(new_ypts-1))
                newxy[i,j]=self.xy[i2,j2]
                newz[i,j]=self.z[i2,j2]
                
        self.xy=newxy
        self.z=newz
                
    def plot_2d_density(self,pos,cmap='jet',center_cbar=False,shift=np.zeros(2),direct_shift=True,slice_dim=2,eref=0,contour=[],supercell=(1,1),rescale_grid=False,dx=0,**args):
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
            
        if type(pos)!=str:
            z,pos_dim=self.slice_density(pos,dim=slice_dim)
        else:
            z,pos_dim=self.slice_density_weighted(pos,dim=slice_dim)
        
        if np.max(abs(shift))!=0:
            z=self.shift_coord(shift,z,direct=direct_shift)
        
        if direct_shift:
            shift=np.dot(shift,self.lv[:2,:2])
            
        self.z=z
        if eref=='ef':
            eref=parse_doscar('./DOSCAR')[2]
        self.z-=eref
        
        if dx!=0:
            for i in range(np.shape(self.z)[0]):
                self.z[i,:]=gaussian_filter(self.z[i,:],dx/(self.xy[1,0,0]-self.xy[0,0,0]),mode='wrap')
            for i in range(np.shape(self.z)[1]):
                self.z[:,i]=gaussian_filter(self.z[:,i],dx/(self.xy[0,1,1]-self.xy[0,1,0]),mode='wrap')
        
        if rescale_grid:
            self.rescale_grid(rescale_grid[0],rescale_grid[1])
            
        if center_cbar:
            vmin=-1*np.max([abs(np.min(z)),abs(np.max(self.z))])
            vmax=np.max([abs(np.min(z)),abs(np.max(self.z))])
        else:
            vmin=np.min(self.z)
            vmax=np.max(self.z)
            
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        for i in range(supercell[0]):
            for j in range(supercell[1]):
                map_data=self.ax_main.pcolormesh(self.xy[:,:,0]+self.lv[0,0]*i+self.lv[1,0]*j,self.xy[:,:,1]+self.lv[0,1]*i+self.lv[1,1]*j,self.z,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
        
        self.fig_main.colorbar(map_data)
        for a in range(supercell[0]):
            for b in range(supercell[1]):
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
                    tempvar=np.dot(tempvar,self.lv[:2,:2])+self.lv[0,:2]*a+self.lv[1,:2]*b
                    self.ax_main.scatter(tempvar[0],tempvar[1],color=colors[j],s=sizes[j])
        patches=[]
        if len(plot_atoms)>0:
            for i in range(len(self.atomtypes)):
                patches.append(Patch(color=colors[i],label=self.atomtypes[i]))
        for i in contour:
            self.ax_main.contour(self.xy[:,:,0],self.xy[:,:,1],self.z,i,colors='black',linestyles='dotted')
                
        self.ax_main.set(xlabel='position / $\AA$', ylabel='position / $\AA$')
        self.fig_main.legend(handles=patches)
        self.ax_main.set_aspect('equal')
        self.fig_main.show()
        
    def plot_2d_fft(self,nperiods=(1,1),scaling='linear',cmap='vivid',normalize=True,window=None,overlay_radius=[],fft_type='xy',fft_range=np.array([-np.inf,np.inf]),ifft=False):
        dim=np.shape(self.z)
        
        if fft_type=='xy':
            inv_lv=np.linalg.inv(self.lv)[:2,:2]
            max_vals=self.lv[0]*nperiods[0]+self.lv[1]*nperiods[1]
            x_periodic=np.linspace(0,max_vals[0],dim[0]*nperiods[0])
            y_periodic=np.linspace(0,max_vals[1],dim[1]*nperiods[1])
            z_periodic=np.zeros(tuple([dim[i]*nperiods[i] for i in range(2)]))
            for i in range(nperiods[0]*dim[0]):
                for j in range(nperiods[1]*dim[1]):
                    tempvar=np.array([x_periodic[i],y_periodic[j]])
                    tempvar=np.dot(tempvar,inv_lv)
                    for k in range(2):
                        tempvar[k]-=np.floor(tempvar[k])
                        tempvar[k]=round(tempvar[k]*dim[k])
                        try:
                            z_periodic[i,j]=self.z[int(tempvar[0]),int(tempvar[1])]
                        except IndexError:
                            while tempvar[k]>=dim[k] or tempvar[k]<0:
                                if tempvar[k]>=dim[k]:
                                    tempvar[k]-=dim[k]
                                elif tempvar[k]<0:
                                    tempvar[k]+=dim[k]
                        z_periodic[i,j]=self.z[int(tempvar[0]),int(tempvar[1])]            
        elif fft_type=='lv':
            x_periodic=np.linspace(0,np.linalg.norm(self.lv[0])*nperiods[0],dim[0]*nperiods[0])
            y_periodic=np.linspace(0,np.linalg.norm(self.lv[1])*nperiods[1],dim[1]*nperiods[1])
            z_periodic=np.zeros(tuple([dim[i]*nperiods[i] for i in range(2)]))
            for i in range(nperiods[0]):
                for j in range(nperiods[1]):
                    z_periodic[dim[0]*i:dim[0]*(i+1),dim[1]*j:dim[1]*(j+1)]+=self.z
                    
        elif fft_type=='test':
            x_periodic=np.linspace(0,np.linalg.norm(self.lv[0])*nperiods[0],dim[0]*nperiods[0])
            y_periodic=np.linspace(0,np.linalg.norm(self.lv[1])*nperiods[1],dim[1]*nperiods[1])
            z_periodic=np.zeros(tuple([dim[i]*nperiods[i] for i in range(2)]))
            for i in range(nperiods[0]*dim[0]):
                for j in range(nperiods[1]*dim[1]):
                    z_periodic[i,j]=np.cos(i*2*np.pi*50/dim[0])+np.cos(j*2*np.pi*50/dim[1])
                
        if window=='hann':
            wx,wy=np.meshgrid(np.hanning(len(y_periodic)),np.hann(len(x_periodic)))
            z_periodic*=(wx*wy)
        elif window=='blackman':
            wx,wy=np.meshgrid(np.blackman(len(y_periodic)),np.blackman(len(x_periodic)))
            z_periodic*=(wx*wy)
            
        self.x_periodic=x_periodic
        self.y_periodic=y_periodic
        self.z_periodic=z_periodic
            
        z_fft=np.fft.fft2(z_periodic)
        
        x_fft=np.fft.fftshift(np.fft.fftfreq(nperiods[0]*dim[0],abs(x_periodic[-1]-x_periodic[0])/(len(x_periodic)-1)))
        y_fft=np.fft.fftshift(np.fft.fftfreq(nperiods[1]*dim[1],abs(y_periodic[-1]-y_periodic[0])/(len(y_periodic)-1)))
        
        if ifft:
            self.fig_ifft,self.ax_ifft=plt.subplots(1,1,tight_layout=True)
            tempx,tempy=np.meshgrid(self.x_periodic,self.y_periodic)
            tempx=tempx.T
            tempy=tempy.T
            z_ifft=np.real(np.fft.ifft2(z_fft))
            self.ax_ifft.pcolormesh(tempx,tempy,z_ifft,cmap=cmap,shading='nearest')
            self.ax_ifft.set(xlabel='position / $\AA$', ylabel='position / $\AA$')
            self.fig_ifft.show()
            
        z_fft=abs(np.fft.fftshift(z_fft))
        
        if scaling=='log':
            z_fft=np.log(z_fft)
        if scaling=='sqrt':
            z_fft=np.sqrt(z_fft)
            
        if normalize:
            z_fft/=np.max(z_fft)
        
        self.fig_fft,self.ax_fft=plt.subplots(1,1,tight_layout=True)
        
        if np.min(abs(fft_range))==np.inf:
            self.ax_fft.pcolormesh([[x_fft[j] for i in range(len(y_fft))] for j in range(len(x_fft))],[y_fft for i in range(len(x_fft))],z_fft,cmap=cmap,shading='nearest')
        else:
            fft_range_index=np.zeros((2,2),dtype=int)
            for i in range(2):
                for j,k in zip([x_fft,y_fft],range(2)):
                    fft_range_index[k,i]=int(np.argmin(abs(j-fft_range[i])))
            self.ax_fft.pcolormesh([[x_fft[j] for i in range(fft_range_index[1,0],fft_range_index[1,1])] for j in range(fft_range_index[0,0],fft_range_index[0,1])],[y_fft[fft_range_index[1,0]:fft_range_index[1,1]] for i in range(fft_range_index[0,0],fft_range_index[0,1])],z_fft[fft_range_index[0,0]:fft_range_index[0,1],fft_range_index[1,0]:fft_range_index[1,1]],cmap=cmap,shading='nearest')
        
        self.ax_fft.plot([0,0],[np.min(y_fft),np.max(y_fft)],linestyle='dashed',c='black')
        self.ax_fft.plot([np.min(x_fft),np.max(x_fft)],[0,0],linestyle='dashed',c='black')
        
        for i in overlay_radius:
            c=plt.Circle((0,0),i,edgecolor='white',fill=None)
            self.ax_fft.add_patch(c)
        
        self.ax_fft.set(xlabel='$k_x$ / $\AA^{-1}$')
        self.ax_fft.set(ylabel='$k_y$ / $\AA^{-1}$')
        self.ax_fft.set_aspect('equal')
        self.fig_fft.show()
        
    def save_2d_slice(self,ofile):
        np.save(ofile,self.z)

    def plot_1d_slice(self,axis,pos,direct=True,fit=True,nperiods=1,nperiods_short=0,nperiods_short2=0,print_fit_params=False,periodic_fit=True,center_x=False):
        if not hasattr(self,'fig_slice'):
            self.fig_slice,self.ax_slice=plt.subplots(1,1,tight_layout=True)
        def model_cosine(x,a,k,phi,y0):
            y=y0+a*np.cos(2*np.pi*k*x+phi)
            return y
        
        def model_cosine_sum(x,a1,a2,k1,k2,phi1,phi2,y0):
            y=y0+a1*np.cos(2*np.pi*k1*x+phi1)+a2*np.cos(2*np.pi*k2*x+phi2)
            return y
        
        def model_cosine_sum_v2(x,a1,a2,a3,k1,k2,k3,phi1,phi2,phi3,y0):
            y=y0+a1*np.cos(2*np.pi*k1*x+phi1)+a2*np.cos(2*np.pi*k2*x+phi2)+a3*np.cos(2*np.pi*k3*x+phi3)
            return y
        
        if type(axis)==int:
            if pos!='average':
                if direct:
                    pos=[round(pos*np.shape(self.e)[1-axis])]
                else:
                    pos=[pos]
            else:
                pos=[i for i in range(np.shape(self.xy)[1-axis])]
            tempy=np.zeros(np.shape(self.xy)[axis])
            
            for i in range(len(pos)):
                tempx=self.xy.take(i,axis=1-axis)
                tempx-=tempx[0]
                tempx=np.array([np.linalg.norm(j) for j in tempx])
                    
                tempy+=self.z.take(pos[i],axis=1-axis)/len(pos)
            
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
            elif fit=='two cosines':
                bounds=[[0,0,0,0,0,0,-np.max(tempx)*2*np.pi,-np.max(tempx)*2*np.pi,-np.max(tempx)*2*np.pi,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.max(tempx)*2*np.pi,np.max(tempx)*2*np.pi,np.max(tempx)*2*np.pi,np.inf]]
                
                spacing=np.max(tempx)/nperiods_short
                spacing2=np.max(tempx)/nperiods_short2
                
                p0=[np.max(tempy)-np.min(tempy),(np.max(tempy)-np.min(tempy))/2,(np.max(tempy)-np.min(tempy))/2,nperiods/np.max(tempx),1/spacing,1/spacing2,tempx[np.argmax(tempy)],tempx[np.argmax(tempy)],tempx[np.argmax(tempy)],np.average(tempy)]
                
                
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
                
            elif fit=='two cosines':
                popt,pcov=curve_fit(model_cosine_sum_v2,periodic_x,periodic_y,p0=p0,bounds=bounds)
                fit_y=model_cosine_sum_v2(tempx,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9])
            
            else:
                popt,pcov=curve_fit(model_cosine_sum,periodic_x,periodic_y,p0=p0,bounds=bounds)
                fit_y=model_cosine_sum(tempx,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
            pcov=np.sqrt(np.diag(pcov))
            
            self.ax_slice.scatter(tempx,fit_y)
            
            self.fit_params.append(popt)
            
            if print_fit_params and fit=='simple':
                print('A = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[1],pcov[1],popt[2],pcov[2],popt[3],pcov[3]))
            elif print_fit_params and fit=='two cosines':
                print('periodic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\natomic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\nnatomic potential #2\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[2],pcov[2],popt[4],pcov[4],popt[1],pcov[1],popt[3],pcov[3],popt[5],pcov[5],popt[6],pcov[6]))
            elif print_fit_params:
                print('periodic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\natomic potential\nA = {} +/- {}\nk = {} +/-{}\nphi = {} +/- {}\ny0 = {} +/- {}'.format(popt[0],pcov[0],popt[2],pcov[2],popt[4],pcov[4],popt[1],pcov[1],popt[3],pcov[3],popt[5],pcov[5],popt[6],pcov[6]))
            
        self.x_slices.append(tempx)
        self.z_slices.append(tempy)
        if type(axis)==int:
            if len(pos)>1:
                pos=0
            tempdata=self.ax_main.plot([self.xy.take(pos,axis=1-axis)[i][0][0] for i in [0,-1]],[self.xy.take(pos,axis=1-axis)[i][0][1] for i in [0,-1]])
        elif axis==None:
            tempdata=self.ax_main.plot([pos[0,0],pos[1,0]],[pos[0,1],pos[1,1]])
            
        color=tempdata[0].get_color()
        self.ax_slice.plot(tempx,tempy,c=color)
        
        if self.filetype=='LOCPOT':
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='eV')
        else:
            self.ax_slice.set(xlabel='position / $\AA$', ylabel='charge density / # electrons $/AA^{-3}$')
        self.fig_slice.show()
        
    def plot_vertical_2d_slice(self,axis,pos,direct=True,center_x=False,cmap='jet',center_cbar=True,overlay_heights=False):
        if type(axis)==int:
            if direct:
                pos=round(pos*np.shape(self.e)[1-axis])
            tempx=self.xy.take(pos,axis=1-axis)
            tempx-=tempx[0]
            tempx=np.array([np.linalg.norm(i) for i in tempx])
                
            if axis==0:
                tempz=self.e[:,pos,:]
            if axis==1:
                tempz=self.e[pos,:,:]
                
            tempdata=self.ax_main.plot([self.xy.take(pos,axis=1-axis)[i][0] for i in [0,-1]],[self.xy.take(pos,axis=1-axis)[i][1] for i in [0,-1]])
            
        #for the case where 'axis' is a list of atoms to slice through
        elif axis==None:
            pos=np.array([self.coord[i,:2]for i in pos])
            tempx=np.array([np.linspace(pos[0,i],pos[1,i],np.min(np.shape(self.e))) for i in range(2)])
            tempx=np.array([[tempx[0,i],tempx[1,i]] for i in range(np.min(np.shape(self.e)))])
            for i in range(len(tempx)):
                tempx[i]=np.dot(tempx[i],np.linalg.inv(self.lv[:2,:2]))
            tempz=np.array([self.e[round(np.shape(self.xy)[0]*tempx[i,0]) ,round(np.shape(self.xy)[1]*tempx[i,1]),:] for i in range(len(tempx))])
            for i in range(len(tempx)):
                tempx[i]=np.dot(tempx[i],self.lv[:2,:2])
                tempx[i]=np.linalg.norm(tempx[i])
            tempx-=np.min(tempx)
            
            tempdata=self.ax_main.plot([pos[0,0],pos[1,0]],[pos[0,1],pos[1,1]])
            
        if center_x:
            tempx-=np.average(tempx)
            
        if center_cbar:
            vmin=-1*np.max([abs(np.min(tempz)),abs(np.max(tempz))])
            vmax=np.max([abs(np.min(tempz)),abs(np.max(tempz))])
        else:
            vmin=np.min(tempz)
            vmax=np.max(tempz)
            
        tempy=np.array([np.linalg.norm(self.lv[2])*i/(np.shape(self.e)[2]-1) for i in range(np.shape(self.e)[2])])
        
        self.fig_2d_slice,self.ax_2d_slice=plt.subplots(1,1,tight_layout=True)
        x,y=np.meshgrid(tempx,tempy)
        map_data=self.ax_2d_slice.pcolormesh(x.T,y.T,tempz,cmap=cmap,shading='nearest',vmin=vmin,vmax=vmax)
        self.fig_2d_slice.colorbar(map_data)
        self.ax_2d_slice.set(xlabel='position / $\AA$', ylabel='position / $\AA$')
        if overlay_heights:
            if type(overlay_heights)!=list:
                overlay_heights=[overlay_heights]
            for i in overlay_heights:
                self.fig_2d_slice.plot(x,[i for i in range(len(x))],linestyle='dashed',lw=2,color='black')
        self.fig_2d_slice.show()
    