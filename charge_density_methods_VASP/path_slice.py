import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import floor
from matplotlib.ticker import FormatStrFormatter
from copy import deepcopy
import os
from scipy.optimize import curve_fit

from lib import parse_CHGCAR, parse_LOCPOT, parse_poscar

#slices 3d data (charge density or electrostatic potential) along a user specified path
#the path must be a list of arrays with a shape of 3, containing the coordinates for points along the path
#the path is linearly interpolated between any specified points
#path_atom indices count from 1, as displayed in VESTA
#if specified, read_data_from_file allows the density data to be read from a .npy file
#if read_data_from_file is used, ifile should be the path to the directory of the POSCAR/CONTCAR
class slice_path():
    def __init__(self,ifile,path_atoms,read_data_from_file=False,filetype='CHGCAR',cmap='seismic',dipole_correction=False,**args):
        self.path_atoms=path_atoms
        self.filetype=filetype
        self.cmap=cmap
        if self.filetype=='LOCPOT' and not read_data_from_file:
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_LOCPOT(ifile)
        elif 'CHG' in self.filetype and not read_data_from_file:
            self.e,self.lv,self.coord,self.atomtypes,self.atomnums=parse_CHGCAR(ifile)
        elif self.filetype=='none':
            npts=1000
            self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar(ifile)[:4]
            self.e=np.zeros((npts,npts,npts))
        if read_data_from_file:
            self.e=np.load(read_data_from_file)
            os.chdir(ifile)
            try:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./CONTCAR')[:4]
            except FileNotFoundError:
                self.lv,self.coord,self.atomtypes,self.atomnums=parse_poscar('./POSCAR')[:4]
            
        dim=np.shape(self.e)
        
        if 'ref' in args:
            for i in args['ref']:
                if self.filetype=='LOCPOT':
                    tempvar=parse_LOCPOT(i)[0]
                else:
                    tempvar=parse_CHGCAR(i)[0]
                self.e-=tempvar
                
        if dipole_correction:
            def line(x,a,b):
                return a*x+b
            
            if 'dipole_correction_range' in args:
                dipole_correction_range=args['dipole_correction_range']
            else:
                dipole_correction_range=[0,np.linalg.norm(self.lv[2])]
                
            dipole_correction_pts=[round(dipole_correction_range[i]/np.linalg.norm(self.lv[2])*(np.shape(self.e)[2]-1)) for i in range(2)]
                
            tempy=np.zeros(dipole_correction_pts[1]-dipole_correction_pts[0])
            for i in range(dim[0]):
                for j in range(dim[1]):
                    tempy+=self.e[i,j,dipole_correction_pts[0]:dipole_correction_pts[1]]/dim[0]/dim[1]
            
            popt,pcov=curve_fit(line,np.linspace(dipole_correction_range[0],dipole_correction_range[1],dipole_correction_pts[1]-dipole_correction_pts[0]),tempy)
            fity=line(np.linspace(0,np.linalg.norm(self.lv[2]),np.shape(self.e)[2]),popt[0],popt[1])
            for i in range(dim[0]):
                for j in range(dim[1]):
                    self.e[i,j,:]-=fity
                
        if 'norm' in args:
            norm_mode=args['norm']
            if norm_mode not in ['none','slice','total']:
                print('unknown normalization prompt. data will not be normalized.')
        else:
            norm_mode='none'
                
        if 'gradient' in args:
            if args['gradient']==True:
                de=np.zeros(dim)
                dr=[np.linalg.norm(self.lv[i])/dim[i] for i in range(3)]
                for i in range(dim[0]):
                    for j in range(dim[1]):
                        for k in range(dim[2]):
                            for l in [-1,1]:
                                if i+l>=dim[0]:
                                    de[i][j][k]+=abs((self.e[i+l-dim[0]][j][k]-self.e[i+l-dim[0]-1][j][k])/dr[0]/2.0)
                                else:
                                    de[i][j][k]+=abs((self.e[i+l][j][k]-self.e[i+l-1][j][k])/dr[0]/2.0)
                                if j+l>=dim[1]:
                                    de[i][j][k]+=abs((self.e[i][j+l-dim[1]][k]-self.e[i][j+l-dim[1]-1][k])/dr[1]/2.0)
                                else:
                                    de[i][j][k]+=abs((self.e[i][j+l][k]-self.e[i][j+l-1][k])/dr[1]/2.0)
                                if k+l>=dim[2]:
                                    de[i][j][k]+=abs((self.e[i][j][k+l-dim[2]]-self.e[i][j][k+l-dim[2]-1])/dr[2]/2.0)
                                else:
                                    de[i][j][k]+=abs((self.e[i][j][k+l]-self.e[i][j][k+l-1])/dr[2]/2.0)
                self.e=de
                print('density gradient calculated.')
                            
        if 'zrange' in args:
            self.zrange=args['zrange']
        else:
            self.zrange=[0.0,1.0]
            
        if 'tol' in args:
            self.tol=args['tol']
        else:
            self.tol=0.0
        
        if 'colors' in args:
            self.colors=args['colors']
        else:
            self.colors=['black' for i in range(len(self.atomnums))]
            
        if 'sizes' in args:
            self.sizes=args['sizes']
        else:
            self.sizes=[100 for i in range(len(self.atomnums))]
            
        if 'direct' in args:
            for i in range(2):
                self.zrange[i]=np.dot(self.zrange[i],self.lv[2])
        
        self.path=[]
        for i in self.path_atoms:
            if len(i)>1:
                tempvar=i[1:]
            else:
                tempvar=[0,0]
            self.path.append(deepcopy(self.coord[i[0]-1,:2]))
            for j in range(2):
                self.path[-1]+=self.lv[j,:2]*float(tempvar[j])
                
        #adds tolerance to the initial and final positions specified by the path
        idiff=(self.path[1]-self.path[0])/np.linalg.norm(self.path[1]-self.path[0])
        counter=0
        while True in [np.isnan(i) for i in idiff]:
            idiff=(self.path[counter]-self.path[0])/np.linalg.norm(self.path[counter]-self.path[0])
            counter+=1
        fdiff=(self.path[-1]-self.path[-2])/np.linalg.norm(self.path[-1]-self.path[-2])
        counter=0
        while True in [np.isnan(i) for i in fdiff]:
            fdiff=(self.path[-1]-self.path[counter])/np.linalg.norm(self.path[-1]-self.path[counter])
            counter+=1
        self.path[0]-=idiff*self.tol
        self.path[-1]+=fdiff*self.tol
            
        path_length=sum([np.linalg.norm(self.path[i]-self.path[i-1]) for i in range(1,len(self.path))])
        
        if 'npts' in args:
            self.npts=args['npts']
        else:
            self.npts=path_length/min([np.linalg.norm(self.lv[j]) for j in range(3)])*min(dim)
        step_points=np.zeros(len(self.path)-1,dtype=np.int8)
        for i in range(1,len(self.path)):
            step_points[i-1]=round(np.linalg.norm(self.path[i]-self.path[i-1])/path_length*self.npts)-1
            if step_points[i-1]==1:
                step_points[i-1]+=1
        step_points[0]+=1
        self.npts=int(sum(step_points))
        path_distance=np.array([path_length*i/(self.npts-1) for i in range(self.npts)])
        self.path_coord=[self.path[0]]
        for i in range(1,len(self.path)):
            for j in range(step_points[i-1]):
                if i==1 and j==0:
                    pass
                else:
                    self.path_coord.append(self.path[i-1]+(self.path[i]-self.path[i-1])/(step_points[i-1]-1)*j)
        self.path_coord=np.array(self.path_coord)
        
        for i in range(len(self.path_coord)):
            self.path_coord[i]=np.dot(self.path_coord[i],np.linalg.inv(self.lv[:2,:2]))
            for j in range(2):
                while self.path_coord[i][j]>=1.0 or self.path_coord[i][j]<0.0:
                    if self.path_coord[i][j]>=1.0:
                        self.path_coord[i][j]-=1.0
                    if self.path_coord[i][j]<0.0:
                        self.path_coord[i][j]+=1.0
                self.path_coord[i][j]=int(floor(self.path_coord[i][j]*dim[j]))
        
        for i in range(2):
            self.zrange[i]=round(self.zrange[i]*dim[2])
        self.z=np.zeros((self.npts,self.zrange[1]-self.zrange[0]))
        for i in range(self.npts):
            self.z[i]=self.e[int(self.path_coord[i][0]),int(self.path_coord[i][1]),self.zrange[0]:self.zrange[1]]
        if norm_mode=='slice':
            self.z-=np.min(self.z)
            self.z/=np.max(self.z)
        elif norm_mode=='total':
            self.z-=np.min(self.e)
            self.z/=(np.max(self.e)-np.min(self.e))
        
        self.x=np.array([path_distance for i in range(self.zrange[1]-self.zrange[0])]).transpose()
        self.y=np.array([[(self.zrange[1]-self.zrange[0])/dim[2]*np.linalg.norm(self.lv[2])*j/dim[2] for i in range(self.npts)] for j in range(self.zrange[1]-self.zrange[0])]).transpose()
        
        self.plot_main()
        
    def plot_main(self,**args):
        self.fig_main,self.ax_main=plt.subplots(1,1,tight_layout=True)
        self.density_plot=self.ax_main.pcolormesh(self.x,self.y,self.z,cmap=self.cmap,shading='nearest')
        cbar=self.fig_main.colorbar(self.density_plot)
        if 'clim' not in args:
            max_val=max([abs(i) for i in self.z.flatten()])
            self.density_plot.set_clim(vmin=-max_val,vmax=max_val)
        else:
            self.density_plot.set_clim(vmin=args['clim'][0],vmax=args['clim'][1])
        if 'CHG' in self.filetype:
            cbar.set_label('change in electron density / electrons $\AA^{-3}$')
        if self.filetype=='LOCPOT':
            cbar.set_label('electrostatic potential / eV')
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))
        atom_pos=[self.tol]
        for i in range(1,len(self.path)):
            atom_pos.append(atom_pos[-1]+np.linalg.norm(self.path[i]-self.path[i-1]))
            if i==1:
                atom_pos[i]-=self.tol
            if i==len(self.path)-1:
                atom_pos[i]-=self.tol
        for i in range(len(self.path_atoms)):
            for j in range(len(self.atomtypes)):
                if self.path_atoms[i][0]-1 < sum(self.atomnums[:j+1]):
                    break
            self.ax_main.scatter(atom_pos[i],self.coord[self.path_atoms[i][0]-1][2]-self.zrange[0],color=self.colors[j],s=self.sizes[j])
            
        if 'contour' in args:
            if self.filetype=='LOCPOT':
                tempvar=parse_LOCPOT(args['contour'][0])[0]
            else:
                tempvar=parse_CHGCAR(args['contour'][0])[0]
            contour_data=np.zeros((self.npts,self.zrange[1]-self.zrange[0]))
            for i in range(self.npts):
                contour_data[i]+=tempvar[int(self.path_coord[i][0]),int(self.path_coord[i][1]),self.zrange[0]:self.zrange[1]]
            self.ax_main.contour(self.x,self.y,contour_data,[args['contour'][1]],colors='black',linestyles='dotted')
            
        patches=[]
        for i in range(len(self.atomtypes)):
            patches.append(Patch(color=self.colors[i],label=self.atomtypes[i]))
                
        self.ax_main.set(xlabel='position along path / $\AA$')
        self.ax_main.set(ylabel='vertical position / $\AA$')
        self.fig_main.legend(handles=patches)
        self.ax_main.set_aspect('equal')
        self.fig_main.show()

    #plots a vertical 1d slice
    #if pos is an array, the slice is taken at the given position
    #if pos is an int, the slice is taken at the specified atom number
    def get_1d_slice(self,pos,direct=True):
        if type(pos)==int:
            pos=self.coord[pos-1,:2]
            pos=np.dot(pos,np.linalg.inv(self.lv[:2,:2]))
        elif direct==False:
            pos=np.dot(pos,np.linalg.inv(self.lv[:2,:2]))
        
        for i in range(2):
            pos[i]=round(pos[i]*np.shape(self.e)[i])
            while pos[i]>=np.shape(self.e)[i] or pos[i]<0:
                if pos[i]<0:
                    pos[i]+=np.shape(self.e)[i]
                if pos[i]>=np.shape(self.e)[i]:
                    pos[i]-=np.shape(self.e)[i]
            
        return self.e[int(pos[0]),int(pos[1]),:],pos
    
    def plot_1d_slice(self,pos):
        y=[]
        colors=[]
        tempx=np.linspace(0,np.linalg.norm(self.lv[2]),np.shape(self.e)[2])
        if type(pos)!=list:
            pos=[pos]
        for i in pos:
            tempy,i=self.get_1d_slice(i)
            y.append(tempy)
            mindiff=np.max(self.x)
            for j in range(len(self.path_coord)):
                tempdiff=np.linalg.norm(self.path_coord[j]-np.array(i))
                if tempdiff<mindiff:
                    mindiff=tempdiff
                    minindex=j
            tempdata=self.ax_main.plot(self.x[minindex,:],self.y[minindex,:])
            colors.append(tempdata[0].get_color())
            self.fig_main.canvas.draw()
            
        self.fig_slice,self.ax_slice=plt.subplots(1,1,tight_layout=True)
        for i in range(len(y)):
            self.ax_slice.plot(tempx,y[i],color=colors[i])
        self.ax_slice.set(xlabel='position / $\AA$')
        if self.filetype=='LOCPOT':
            self.ax_slice.set(ylabel='electrostatic potential / eV')
        if 'CHG' in self.filetype:
            self.ax_slice.set(ylabel='electron density / electrons $\AA^{-3}$')
        self.fig_slice.show()
        
#helper function to generate path from VESTA text
#simply click the atoms you would like to slice through in the order you would like to slice them
#any duplicates will be discarded
#then copy all the text from the VESTA window and paste as a single string into the argument of this function
#the path list that can be supplied to the slice_path class is returned
def create_path_from_VESTA(text):
    path=[]
    text=text.split()
    counter=-1
    for i in text:
        if counter<0 and i=='Atom:':
            counter=0
            tempvar=[]
        elif counter==0:
            for j in path:
                if int(i)==j[0]:
                    counter=-1
                    break
            else:
                tempvar.append(int(i))
                counter+=1
        elif counter in [1,2]:
            counter+=1
        elif counter in [3,4]:
            tempvar.append(int(np.floor(float(i))))
            counter+=1
            if counter==5:
                path.append(tempvar)
                counter=-1

    return path