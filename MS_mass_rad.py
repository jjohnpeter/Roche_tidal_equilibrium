import numpy as np
import time
import shutil
import os
from scipy.interpolate import interp1d
from math import pi
from multiprocessing import Process
from para_tidal_eq import *
from dir_info import *

# Range of stellar masses for a fixed BH mass
Mstar_list = np.arange(0.15, 0.85, 0.05)  # Msun

# ---- use main-sequence star mass-radius relation
Rstar_fname = savedir + 'mass_radius_Kruns_3Gyr.iso'
data = np.loadtxt(Rstar_fname, skiprows=2)
Marr, Rarr = data[:, 2], 10**(data[:, 11])
RM_interp = interp1d(Marr, Rarr)
Rstar_list = RM_interp(Mstar_list)   # Rsun
#Rstar_simp = Mstar_list**(5/4)

LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
if not os.path.exists(savedir + LaneEmden_fname):
    # need to run LaneEmden.py to create the polytrope_profile
    os.system('python ' + pydir + 'LaneEmden.py')
data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
ximax, phimax = data[-1, 0], data[-1, 2]
#print(f'ximax = {ximax} phimax = {phimax}')

print(f'Mstar_list: {Mstar_list}')
#print(f'Rstar_list_simp: {Rstar_simp}')
print(f'Rstar_list: {Rstar_list}')
Kentr_MSlist = (4*pi)**(1./npoly)/(npoly+1)*(Mstar_list/phimax)**(1-1./npoly)*(Rstar_list/ximax)**(3./npoly-1)
print(f'Kentr_MSlist: {Kentr_MSlist}')

