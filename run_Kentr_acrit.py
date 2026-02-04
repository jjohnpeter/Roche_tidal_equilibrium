import numpy as np
import time
import shutil
import os
#import roche_vol.py
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
# Takes ~1.5 hours for Ncpu = 6 for 24 runs (most efficient I can get on my computer)
# For 6 runs and Ncpu = 6, takes ~30 minutes
Ncpu = 6   # number of processors to be used
smamin, smamax = 0.0572, 0.0574

# ---- manually set a list of entropies
#Kentr_list = [0.7]

# ---- consider a particular entropy given by a star of given mass and radius
# Mstar = 0.50 # Msun
# Rstar_fname = savedir + 'mass_radius_Kruns_3Gyr.iso'
# data = np.loadtxt(Rstar_fname, skiprows=2)
# Marr, Rarr = data[:, 2], 10**(data[:, 11])
# RM_interp = interp1d(Marr, Rarr)
# Rstar = RM_interp(Mstar)   # Rsun

Mstar = 0.40 # Msun
#---Since changing tidal_equilibrium_acrit.py for npoly runs, we dont need this stuff:
Rstar = WD_Radius(Mstar) # WD Mass-Radius Relation (even for the npoly != 1.5)
#Rstar = 0.0919930869  n = 1 runs

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
Mstar_list = [Mstar] # Puts different initial stellar profile masses into a list just like Kentr
#print(f'Mstar= {Mstar}')
#print(f'Rstar= {Rstar}')
#print(f'rhoc= {rhoc}')
#print(f'rhoc_cgs={rhoc*5.907} [g/cm^3]')
#print(f'K= {Kentr}')


#Nsma = 2*Ncpu   # each processor calculates two cases
Nsma = 6
smaarr = np.linspace(smamin, smamax, Nsma, endpoint=True)
#smaarr = np.array([7.043, 7.130, 7.217, 7.304, 7.391, 7.478, 7.565, 7.652, 7.739, 7.826, 7.913, 8]) # Manual

# MS case
def run_tidal_eq(jlist, Mstar, s):
    # jlist is a chunk of range(Nsma), i for Kentr_list index, s is a random seed (not used)
    np.random.seed(s)
    for j in jlist:
        # WD case
        #os.system('python ' + dir_main + 'tidal_equilibrium_acrit.py %.5f %.5f %.5f' % (Kentr, rhoc, smaarr[j]))
        # npoly case!
        os.system('python ' + dir_main + 'tidal_equilibrium_acrit.py %.2f %.5f' % (Mstar, smaarr[j]))
        # MS star case
        #os.system('python ' + dir_main + 'tidal_equilibrium_MS_K.py %.5f %.5f' % (Mstar, smaarr[j]))

NK = len(Kentr_list)
jlist_chunks = np.array_split(range(Nsma), Ncpu)

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
    # old path and MS_K runs
    #Kentr = Kentr_list[i]
    #Kentr_dir = savedir + 'Kentr%.5f' % Kentr
    # DWD runs / npoly
    rhoc = rhoc_list[i]
    Kentr_dir = savedir + 'rhoc%.3f_Qbh%.2f' % (rhoc, Qbh)
    if not os.path.exists(Kentr_dir):
        os.makedirs(Kentr_dir, exist_ok=True)
    for j in range(Nsma):
        sma = smaarr[j]
        # old path and path for new MS_K runs
        #sma_dir = savedir + 'Kentr%.5f/sma%.5f' % (Kentr, sma)
        # main DWD runs / npoly
        sma_dir = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f' % (rhoc, Qbh, sma)
        if os.path.exists(sma_dir):
            loop_shutil(sma_dir)
        os.makedirs(sma_dir, exist_ok=True)


if __name__ == '__main__':
    for i in range(NK):
        # old
        #Kentr = Kentr_list[i]
        #print('working on Kentr=', Kentr)
        Mstar = Mstar_list[i]
        print('working on MS Mstar=', Mstar)
        # WD CASE / npoly (changed Kentr --> Mstar for npoly cases)
        procs = [Process(target=run_tidal_eq,
                         args=(jlist_chunks[n], Mstar, np.random.randint(10)))
                 for n in range(Ncpu)]
        # MS Star CASE
        #procs = [Process(target=run_tidal_eq,
        #                 args=(jlist_chunks[n], Mstar, np.random.randint(10)))
        #         for n in range(Ncpu)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
