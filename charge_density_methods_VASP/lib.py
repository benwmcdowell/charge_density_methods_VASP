import numpy as np

#reads the total charge density from a CHGCAR file
def parse_CHGCAR(ifile, **args):
    if 'scale' in args:
        rescale=False
    else:
        rescale=True
    #reads atomic positions and lattice vectors
    lv=np.zeros((3,3))
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
        coord=np.zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=np.dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        e=np.zeros((dim[0],dim[1],dim[2]))
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
        vol=np.dot(np.cross(lv[0],lv[1]),lv[2])
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
            coord[i]=np.dot(coord[i],inv(lv))
            for j in coord[i]:
                file.write(' {}'.format(j))
            file.write('\n')
        file.write('\n')
        dim=np.shape(e)
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
    lv=np.zeros((3,3))
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
        coord=np.zeros((sum(atomnums),3))
        for i in range(sum(atomnums)):
            line=chgcar.readline().split()
            for j in range(3):
                coord[i][j]=float(line[j])
            if mode[0]=='D':
                coord[i]=np.dot(coord[i],lv)
        line=chgcar.readline()
        #starts reading charge density info
        line=chgcar.readline().split()
        x=0
        y=0
        z=0
        dim=[int(i) for i in line]
        pot=np.zeros((dim[0],dim[1],dim[2]))
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
                            for i in range(round(sum(atomnums)/5)+1):
                                line=chgcar.readline()
        
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

def parse_poscar(ifile):
    with open(ifile, 'r') as file:
        lines=file.readlines()
        sf=float(lines[1])
        latticevectors=[float(lines[i].split()[j])*sf for i in range(2,5) for j in range(3)]
        latticevectors=np.array(latticevectors).reshape(3,3)
        atomtypes=lines[5].split()
        atomnums=[int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start=8
            mode=lines[7].split()[0]
        else:
            mode=lines[8].split()[0]
            start=9
            seldyn=[''.join(lines[i].split()[-3:]) for i in range(start,sum(atomnums)+start)]
        coord=np.array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start,sum(atomnums)+start)])
        if mode!='Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j]>1.0 or coord[i][j]<0.0:
                        if coord[i][j]>1.0:
                            coord[i][j]-=1.0
                        elif coord[i][j]<0.0:
                            coord[i][j]+=1.0
                coord[i]=np.dot(coord[i],latticevectors)
            
    #latticevectors formatted as a 3x3 array
    #coord holds the atomic coordinates with shape ()
    try:
        return latticevectors, coord, atomtypes, atomnums, seldyn
    except NameError:
        return latticevectors, coord, atomtypes, atomnums
    
#reads DOSCAR
def parse_doscar(filepath):
    with open(filepath,'r') as file:
        line=file.readline().split()
        atomnum=int(line[0])
        for i in range(5):
            line=file.readline().split()
        nedos=int(line[2])
        ef=float(line[3])
        dos=[]
        energies=[]
        for i in range(atomnum+1):
            if i!=0:
                line=file.readline()
            for j in range(nedos):
                line=file.readline().split()
                if i==0:
                    energies.append(float(line[0]))
                if j==0:
                    temp_dos=[[] for k in range(len(line)-1)]
                for k in range(len(line)-1):
                    temp_dos[k].append(float(line[k+1]))
            dos.append(temp_dos)
    energies=np.array(energies)-ef
    
    #orbitals contains the type of orbital found in each array of the site projected dos
    num_columns=np.shape(dos[1:])[1]
    if num_columns==3:
        orbitals=['s','p','d']
    elif num_columns==6:
        orbitals=['s_up','s_down','p_up','p_down','d_up','d_down']
    elif num_columns==9:
        orbitals=['s','p_y','p_z','p_x','d_xy','d_yz','d_z2','d_xz','d_x2-y2']
    elif num_columns==18:
        orbitals=['s_up','s_down','p_y_up','p_y_down','p_z_up','p_z_down','p_x_up','p_x_down','d_xy_up','d_xy_down','d_yz_up','d_yz_down','d_z2_up','d_z2_down','d_xz_up','d_xz_down','d_x2-y2_up','d_x2-y2_down']
        
    #dos is formatted as [[total dos],[atomic_projected_dos for i in range(atomnum)]]
    #total dos has a shape of (4,nedos): [[spin up],[spin down],[integrated, spin up],[integrated spin down]]
    #atomic ldos have shapes of (6,nedos): [[i,j] for j in [spin up, spin down] for i in [s,p,d]]
    #energies has shape (1,nedos) and contains the energies that each dos should be plotted against
    return dos, energies, ef, orbitals