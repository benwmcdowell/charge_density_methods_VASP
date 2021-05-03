from numpy import zeros, shape, dot, array
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import floor
from matplotlib.ticker import FormatStrFormatter

from lib import parse_CHGCAR, parse_LOCPOT

#slices 3d data (charge density or electrostatic potential) along a user specified path
#the path must be a list of arrays with a shape of 3, containing the coordinates for points along the path
#the path is linearly interpolated between any specified points
def slice_path(ifile,path_atoms,**args):
    if 'filetype' in args:
        filetype=args['filetype']
    else:
        filetype='CHGCAR'
    
    if filetype=='LOCPOT':
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    else:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
    dim=shape(e)
    
    if 'ref' in args:
        for i in args['ref']:
            if filetype=='LOCPOT':
                tempvar=parse_LOCPOT(i)[0]
            else:
                tempvar=parse_CHGCAR(i)[0]
            e-=tempvar
        
    if 'zrange' in args:
        zrange=args['zrange']
    else:
        zrange=[0.0,1.0]
        
    if 'colors' in args:
        colors=args['colors']
    else:
        colors=['black' for i in range(len(atomnums))]
        
    if 'sizes' in args:
        sizes=args['sizes']
    else:
        sizes=[800 for i in range(len(atomnums))]
        
    if 'direct' in args:
        for i in range(2):
            zrange[i]=dot(zrange[i],lv[2])
    
    path=[]
    for i in path_atoms:
        if len(i)>1:
            tempvar=i[1:]
        else:
            tempvar=[0,0]
        path.append(coord[i[0]-1][:2])
        for j in range(2):
            path[-1]+=lv[j][:2]*tempvar[j]
    
    path_length=sum([norm(path[i]-path[i-1]) for i in range(1,len(path))])
    
    if 'npts' in args:
        npts=args['npts']
    else:
        npts=path_length/min([norm(lv[j]) for j in range(3)])*min(dim)
        
    step_points=array([round(norm(path[i]-path[i-1])/path_length*npts)-1 for i in range(1,len(path))])
    step_points[0]+=1
    npts=sum(step_points)
    path_distance=array([path_length*i/(npts-1) for i in range(npts)])
    path_coord=[path[0]]
    for i in range(1,len(path)):
        for j in range(step_points[i-1]):
            if i==0 and j==0:
                pass
            else:
                path_coord.append(path[i-1]+(path[i]-path[i-1])/(step_points[i-1]-1)*j)
    path_coord=array(path_coord)
    
    for i in range(len(path_coord)):
        path_coord[i]=dot(path_coord[i],inv(lv[:2,:2]))
        for j in range(2):
            while path_coord[i][j]>=1.0 or path_coord[i][j]<0.0:
                if path_coord[i][j]>=1.0:
                    path_coord[i][j]-=1.0
                if path_coord[i][j]<0.0:
                    path_coord[i][j]+=1.0
            path_coord[i][j]=int(floor(path_coord[i][j]*dim[j]))
    
    for i in range(2):
        zrange[i]=round(zrange[i]*dim[2])
    z=zeros((npts,zrange[1]-zrange[0]))
    for i in range(npts):
        z[i]=e[int(path_coord[i][0]),int(path_coord[i][1]),zrange[0]:zrange[1]]
        
    x=array([path_distance for i in range(zrange[1]-zrange[0])]).transpose()
    y=array([[(zrange[1]-zrange[0])/dim[2]*norm(lv[2])*j/dim[2] for i in range(npts)] for j in range(zrange[1]-zrange[0])]).transpose()
    
    plt.figure()
    plt.pcolormesh(x,y,z,cmap='jet',shading='nearest')
    cbar=plt.colorbar()
    cbar.set_label('change in electron density / electrons $\AA^{-3}$')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%+.4f'))
    atom_pos=[0.0]
    for i in range(1,len(path)):
        atom_pos.append(atom_pos[-1]+norm(path[i]-path[i-1]))
    for i in range(len(path_atoms)):
        for j in range(len(atomtypes)):
            if path_atoms[i][0]-1 < sum(atomnums[:j+1]):
                break
        plt.scatter(atom_pos[i],coord[path_atoms[i][0]-1][2]-zrange[0],color=colors[j],s=sizes[j])
        
    patches=[]
    for i in range(len(atomtypes)):
        patches.append(Patch(color=colors[i],label=atomtypes[i]))
            
    plt.xlabel('position along path / $\AA$')
    plt.ylabel('vertical position / $\AA$')
    plt.legend(handles=patches)
    plt.show()
