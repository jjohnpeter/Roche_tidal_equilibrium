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

# run a large number of equilibrium solutions for different Kentr and rhoc
# (fixing Qbh, sma, npoly, as specified by para_tidal_eq.py)

# note: for Ncpu=12, it takes ~10 minutes for each Kentr
Ncpu = 12   # number of processors to be used
#Ncpu = 8 # Manual
rhocmin, rhocmax = 22, 23

# ---- manually set a list of entropies
Kentr_list = [0.7]

# ---- consider a particular entropy given by a star of given mass and radius
#Mstar = 0.8 # Msun
#Rstar = WD_Radius(Mstar) # WD Mass-Radius Relation

# ---- use main-sequence star mass-radius relation
# Rstar_fname = savedir + 'mass_radius_3Gyr.txt'
# data = np.loadtxt(Rstar_fname, skiprows=1)
# Marr, Rarr = data[:, 0], data[:, 1]
# RM_interp = interp1d(Marr, Rarr)
# Rstar = RM_interp(Mstar)   # Rsun

LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
if not os.path.exists(savedir + LaneEmden_fname):
    # need to run LaneEmden.py to create the polytrope_profile
    os.system('python ' + pydir + 'LaneEmden.py')
data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
ximax, phimax = data[-1, 0], data[-1, 2]

#Kentr = (4*pi)**(1./npoly)/(npoly+1)*(Mstar/phimax)**(1-1./npoly)*(Rstar/ximax)**(3./npoly-1)
# print('Mstar, Rstar, Kentr = ', Mstar, Rstar, Kentr)
#Kentr_list = [Kentr]

Nrhoc = 2*Ncpu   # each processor calculates two cases
#Nrhoc = 1*Ncpu   # Manual
rhocarr = np.linspace(rhocmin, rhocmax, Nrhoc, endpoint=True)
#rhocarr = np.array([7.043, 7.130, 7.217, 7.304, 7.391, 7.478, 7.565, 7.652, 7.739, 7.826, 7.913, 8]) # Manual
# note: for very large rhoc, the star is small,
# so we need to reduce Lmax from the default value of 2.0 to <~1


def run_tidal_eq(jlist, Kentr, s):
    # jlist is a chunk of range(Nrhoc), i for Kentr_list index, s is a random seed (not used)
    np.random.seed(s)
    for j in jlist:
        os.system('python ' + dir_main + 'tidal_equilibrium.py %.5f %.5f' % (Kentr, rhocarr[j]))


NK = len(Kentr_list)
jlist_chunks = np.array_split(range(Nrhoc), Ncpu)

# Work around for Windows being unable to remove directory while in use
def loop_shutil(path, attempts=10):
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, FileNotFoundError):
            time.sleep(1)   # Wait 1 second before trying again

# create all the relevant directories
for i in range(NK):
    Kentr = Kentr_list[i]
    Kentr_dir = savedir + 'Kentr%.5f' % Kentr
    if not os.path.exists(Kentr_dir):
        os.makedirs(Kentr_dir, exist_ok=True)
    for j in range(Nrhoc):
        rhoc = rhocarr[j]
        rhoc_dir = savedir + 'Kentr%.5f/rhoc%.3f' % (Kentr, rhoc)
        if os.path.exists(rhoc_dir):
            loop_shutil(rhoc_dir)
        os.makedirs(rhoc_dir, exist_ok=True)


if __name__ == '__main__':
    for i in range(NK):
        Kentr = Kentr_list[i]
        print('working on Kentr=', Kentr)
        procs = [Process(target=run_tidal_eq,
                         args=(jlist_chunks[n], Kentr, np.random.randint(10)))
                 for n in range(Ncpu)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
