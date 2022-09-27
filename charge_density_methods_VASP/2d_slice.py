from numpy import zeros, shape, dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lib import parse_CHGCAR, parse_LOCPOT

def plot_2d_slice(ifile,pos,**args):
    if 'dim' in args:
        dim=args['dim']
    else:
        dim=2
        
    if 'filetype' in args:
        filetype=args['filetype']
    else:
        filetype='LOCPOT'
    
    if filetype=='LOCPOT':
        e,lv,coord,atomtypes,atomnums=parse_LOCPOT(ifile)
    else:
        e,lv,coord,atomtypes,atomnums=parse_CHGCAR(ifile)
        
    if 'ref' in args:
        for i in args['ref']:
            if filetype=='LOCPOT':
                tempvar=parse_LOCPOT(i)[0]
            else:
                tempvar=parse_CHGCAR(i)[0]
            e-=tempvar
    
    if 'direct' in args:
        pos=norm(dot(pos,lv[dim]))
    
    if 'tol' in args:
        tol=round(args['tol']/norm(lv[dim])*shape(e)[dim])
    else:
        tol=0
    
    plot_atoms=[]
    if 'overlay_atoms' in args:
        ranges=args['overlay_atoms']
        for i in range(sum(atomnums)):
            for j in range(3):
                if coord[i][j] > max(ranges[j]) or coord[i][j] < min(ranges[j]):
                    break
            else:
                plot_atoms.append(i)
    if 'atom_sizes' in args:
        sizes=args['atom_sizes']
    else:
        sizes=[800 for i in range(len(atomnums))]
        
    if 'atom_colors' in args:
        colors=args['atom_colors']
    else:
        colors=['black' for i in range(len(atomnums))]
        
    pos_dim=[]
    for i in range(3):
        if i!=dim:
            pos_dim.append(i)
    
    xy=zeros((shape(e)[pos_dim[0]],shape(e)[pos_dim[1]],2))
    for i in range(len(xy)):
        for j in range(len(xy[i])):
            xy[i][j]+=lv[pos_dim[0]][:2]*i/(len(xy)+1)+lv[pos_dim[1]][:2]*j/(len(xy[i])+1)
        
    pos=round(pos*shape(e)[dim]/norm(lv[dim]))
    z=zeros((shape(e)[pos_dim[0]],shape(e)[pos_dim[1]]))
    for i in range(-tol,tol+1):
        if dim==0:
            z+=e[pos,:,:]/(2*tol+1)
        if dim==1:
            z+=e[:,pos,:]/(2*tol+1)
        if dim==2:
            z+=e[:,:,pos]/(2*tol+1)
            
    plt.figure()
    plt.pcolormesh(xy[:,:,0],xy[:,:,1],z,shading='nearest',cmap='jet')
    plt.colorbar()
    for i in plot_atoms:
        for j in range(len(atomtypes)):
            if i < sum(atomnums[:j+1]):
                break
        plt.scatter(coord[i][pos_dim[0]],coord[i][pos_dim[1]],color=colors[j],s=sizes[j])
    patches=[]
    if len(plot_atoms)>0:
        for i in range(len(atomtypes)):
            patches.append(Patch(color=colors[i],label=atomtypes[i]))
            
    plt.xlabel('position / $\AA$')
    plt.ylabel('position / $\AA$')
    plt.legend(handles=patches)
    plt.axes().set_aspect('equal')
    plt.show()
