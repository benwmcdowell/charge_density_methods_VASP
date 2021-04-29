from numpy import array, shape, zeros
import matplotlib.pyplot as plt

from lib import parse_CHGCAR, parse_LOCPOT

def calc_plane_averaged_density(ifile,**args):
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
    
    x=array([i*lv/(shape(e)[dim]-1) for i in range(shape(e)[dim])])
    y=zeros(shape(e)[dim])
    for i in range(shape(e)[dim]):
        y[i]+=sum(e[:,:,i].flatten())
        for j in range(3):
            if j!=dim:
                y[i]/=shape(e)[dim]
                
    return x,y

def plot_plane_averaged_density(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.ylabel('electrostatice potential / eV')
    plt.xlabel('position / $\AA$')
    plt.show()
