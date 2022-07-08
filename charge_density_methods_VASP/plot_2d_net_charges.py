import matplotlib.pyplot as plt
import numpy as np
import os
from lib import parse_potcar, parse_bader_ACF

class bader_charges_2d():
    def __init__(self,fp):
        self.fp=fp
        os.chdir(self.fp)
        self.x,self.y,self.z,self.charge=parse_bader_ACF('./ACF.dat')
        self.numvalence=parse_potcar('./POTCAR')
        
    def calc_net_charges():
        self.net_charge=[]
        for i in self.plot_atoms:
            for j in range(len(self.atomtypes)):
                if i < sum(self.atomnums[:j+1]):
                    break
            if self.atomtypes[j] in show_charges:
                charge_list.append(self.charges[i]-self.numvalence[j])
        cnorm=Normalize(vmin=min(charge_list),vmax=max(charge_list))