from numpy import zeros, dot

#reads the total charge density from a CHGCAR file
def parse_CHGCAR(ifile):
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
                    
    for i in dim:
        e/=i
    
    return e, lv, coord, atomtypes, atomnums
