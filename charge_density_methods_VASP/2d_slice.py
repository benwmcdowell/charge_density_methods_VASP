from numpy import zeros, shape, dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

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
                tempvar=parse_LOCPOT(i)
            else:
                tempvar=parse_CHGCAR(ifile)
            e-=tempvar
    
    if 'direct' in args:
        pos=norm(dot(pos,lv[dim]))
    
    if 'tol' in args:
        tol=round(args['tol']/norm(lv[dim])*shape(e)[dim])
    else:
        tol=0
        
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
    plt.show()
