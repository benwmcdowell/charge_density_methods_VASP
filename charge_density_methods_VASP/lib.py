from numpy import zeros, dot, cross, shape
from numpy.linalg import inv

#reads the total charge density from a CHGCAR file
def parse_CHGCAR(ifile, **args):
    if 'scale' in args:
        rescale=False
    else:
        rescale=True
    #reads atomic positions and lattice vectors
    lv=zeros((3,3))
    with open(ifile,'r') as chgcar:
        for i in range(8):
            line=chgcar.readline().split()
            if i==1:
                sf=float(line[0])
            if i>1 and i<5:
                for j in range(3):
                    lv[i-2][j]=float(line[j])*sf
            if i==5:
                atomtypes=line
            if i==6:
                atomnums=line
                for j in range(len(atomnums)):
                    atomnums[j]=int(atomnums[j])
            if i==7:
                mode=line[0]
        coord=zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        e=zeros((dim[0],dim[1],dim[2]))
        searching=True
        while searching:
            line=chgcar.readline().split()
            for i in line:
                e[x][y][z]=float(i)
                x+=1
                if x==dim[0]:
                    x=0
                    y+=1
                if y==dim[1]:
                    y=0
                    z+=1
                if z==dim[2]:
                    searching=False
                    break
    if rescale:
        print('charge density values rescaled to electrons per cubic Angstrom')
        vol=dot(cross(lv[0],lv[1]),lv[2])
        e/=vol
    
    return e, lv, coord, atomtypes, atomnums

def write_CHGCAR(filepath, e, lv, coord, atomtypes, atomnums):
    with open(filepath, 'w+') as file:
        file.write('\n1.0\n')
        for i in range(3):
            for j in range(3):
                file.write('   {}'.format(lv[i][j]))
            file.write('\n')
        for i in [atomtypes,atomnums]:
            for j in i:
                file.write('  {}'.format(j))
            file.write('\n')
        file.write('Direct\n')
        for i in range(len(coord)):
            coord[i]=dot(coord[i],inv(lv))
            for j in coord[i]:
                file.write(' {}'.format(j))
            file.write('\n')
        file.write('\n')
        dim=shape(e)
        for i in dim:
            file.write(' {}'.format(i))
        writing=True
        x=0
        y=0
        z=0
        while writing:
            file.write('\n')
            for i in range(5):
                file.write(' {:.11e}'.format(e[x][y][z]))
                x+=1
                if x==dim[0]:
                    y+=1
                    x=0
                if y==dim[1]:
                    z+=1
                    y=0
                if z==dim[2]:
                    writing=False
                    break

#reads the total charge density from a CHGCAR file
def parse_LOCPOT(ifile):
    #reads atomic positions and lattice vectors
    lv=zeros((3,3))
    with open(ifile,'r') as chgcar:
        for i in range(8):
            line=chgcar.readline().split()
            if i==1:
                sf=float(line[0])
            if i>1 and i<5:
                for j in range(3):
                    lv[i-2][j]=float(line[j])*sf
            if i==5:
                atomtypes=line
            if i==6:
                atomnums=line
                for j in range(len(atomnums)):
                    atomnums[j]=int(atomnums[j])
            if i==7:
                mode=line[0]
        coord=zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        pot=zeros((dim[0],dim[1],dim[2]))
        searching=True
        counter=0
        while searching:
            line=chgcar.readline().split()
            if not line:
                break
            for i in line:
                pot[x][y][z]+=float(i)
                x+=1
                if x==dim[0]:
                    x=0
                    y+=1
                if y==dim[1]:
                    y=0
                    z+=1
                if z==dim[2]:
                    z=0
                    counter+=1
                if counter==2:
                    searching=False
                    break
        
    if counter==2:
        pot/=2.0
    return pot, lv, coord, atomtypes, atomnums

#reads the ACF file output by Bader analysis and returns contents
def parse_bader_ACF(ifile):
    with open(ifile, 'r') as file:
        x=[]
        y=[]
        z=[]
        charge=[]
        min_dist=[]
        vol=[]
        for i in range(2):
            line=file.readline()
        while True:
            line=file.readline().split()
            try:
                x.append(float(line[1]))
                y.append(float(line[2]))
                z.append(float(line[3]))
                charge.append(float(line[4]))
                min_dist.append(float(line[5]))
                vol.append(float(line[6]))
            #stops reading the file when '--------' is reached
            except IndexError:
                break
    
    return x, y, z, charge, min_dist, vol

#reads the number of valence electrons for each atom type for the POTCAR file
def parse_potcar(ifile):
    with open(ifile, 'r') as file:
        numvalence=[]
        counter=0
        while True:
            line=file.readline()
            if not line:
                break
            if counter==1:
                numvalence.append(float(line.split()[0]))
            if 'End of Dataset' in line:
                counter=-1
            counter+=1
        
    return numvalence