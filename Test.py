import numpy as np
import time
import shutil
import os
from scipy.interpolate import interp1d
from math import pi
from multiprocessing import Process
from funcs_tidal_eq import WD_Radius
from para_tidal_eq import *
from dir_info import *

# run a large number of equilibrium solutions for different Kentr and sma
# (fixing Qbh, npoly, as specified by para_tidal_eq.py)
# Process is as follows: For desired WD mass, radius is calculated, which defines a unique entropy K
# with given npoly. A unique central density rhoc is defined with Mass, Radius, npoly. A range of sma is
# defined to sweep through for these parameters in units of [Rsun]

# note: for Ncpu=12, it takes ~10 minutes for each Kentr
Ncpu = 12   # number of processors to be used
smamin, smamax = 22, 23

# ---- manually set a list of entropies
#Kentr_list = [0.7]

# ---- consider a particular entropy given by a star of given mass and radius
Mstar = np.array([0.1, 0.2, 0.3, 0.4]) # Msun
Rstar = WD_Radius(Mstar) # WD Mass-Radius Relation


LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
if not os.path.exists(savedir + LaneEmden_fname):
    # need to run LaneEmden.py to create the polytrope_profile
    os.system('python ' + pydir + 'LaneEmden.py')
data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
ximax, phimax = data[-1, 0], data[-1, 2]

Kentr = (4*pi)**(1./npoly)/(npoly+1)*(Mstar/phimax)**(1-1./npoly)*(Rstar/ximax)**(3./npoly-1)
Kentr_list = [Kentr]

rhoc = (Mstar/phimax) / (4*pi*(Rstar/ximax)**3)
rhoc_list = [rhoc]
print(f'Mstar= {Mstar}')
print(f'Rstar= {Rstar}')
print(f'rhoc= {rhoc}')
print(f'rhoc_cgs={rhoc*5.907} [g/cm^3]')
print(f'K= {Kentr}')