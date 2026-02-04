import numpy as np
import time
import shutil
import os
import subprocess
from math import pi
from multiprocessing import Process
from funcs_tidal_eq import WD_Radius
from para_tidal_eq import *
from dir_info import *

# run a large number of equilibrium solutions for different Kentr and sma

run_configurations = [
    (0.10, 0.20, 0.08820, 0.08840, 0.088), (0.10, 0.30, 0.09824, 0.09844, 0.1),
    (0.10, 0.50, 0.11307, 0.11327, 0.120), (0.10, 0.70, 0.12438, 0.12458, 0.15),
    (0.10, 0.90, 0.13377, 0.13397, 0.135), (0.20, 0.10, 0.05081, 0.05101, 0.051),
    (0.20, 0.20, 0.05903, 0.05923, 0.1), (0.20, 0.30, 0.06503, 0.06523, 0.07),
    (0.20, 0.50, 0.07416, 0.07436, 0.075), (0.20, 0.70, 0.08120, 0.08140, 0.084),
    (0.20, 0.90, 0.08711, 0.08731, 0.09), (0.30, 0.10, 0.04098, 0.04118, 0.06),
    (0.30, 0.20, 0.04716, 0.04736, 0.05), (0.30, 0.30, 0.05164, 0.05184, 0.055),
    (0.30, 0.50, 0.05837, 0.05857, 0.06), (0.30, 0.70, 0.06364, 0.06384, 0.065),
    (0.30, 0.90, 0.06803, 0.06823, 0.07), (0.40, 0.08, 0.03424, 0.03444, 0.04),
    (0.40, 0.10, 0.03551, 0.03571, 0.04), (0.40, 0.12, 0.03664, 0.03684, 0.04),
    (0.40, 0.20, 0.04024, 0.04044, 0.05), (0.40, 0.30, 0.04390, 0.04410, 0.05),
    (0.40, 0.50, 0.04937, 0.04957, 0.051), (0.40, 0.70, 0.05364, 0.05384, 0.055),
    (0.40, 0.90, 0.05724, 0.05744, 0.060)
]

# note: for Ncpu=12, it takes ~10 minutes for each Kentr
# Takes ~1.5 hours for Ncpu = 6 for 24 runs (most efficient I can get on my computer)
# For 6 runs and Ncpu = 6, takes ~30 minutes
Ncpu = 24   # number of processors to be used
#Nsma = 2*Ncpu   # each processor calculates two cases
Nsma = 24

# MS case
def run_tidal_eq(jlist, Mstar, Qbh, Lmax, smaarr, s):
    # jlist is a chunk of range(Nsma), i for Kentr_list index, s is a random seed (not used)
    np.random.seed(s)
    for j in jlist:
        sma = smaarr[j]

        cmd_list = [
            'python',
            dir_main + 'tidal_equilibrium_acrit.py',
            '%.2f' % Mstar,
            '%.5f' % sma,
            '%.2f' % Qbh,
            '%.3f' % Lmax
        ]
        subprocess.run(cmd_list, check=True)


# Work around for Windows being unable to remove directory while in use
def loop_shutil(path, attempts=10):
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, FileNotFoundError):
            time.sleep(1)   # Wait 1 second before trying again

if __name__ == '__main__':
    # Make sure LaneEmden prof exists
    LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
    if not os.path.exists(savedir + LaneEmden_fname):
        # need to run LaneEmden.py to create the polytrope_profile
        os.system('python ' + pydir + 'LaneEmden.py')
    data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
    ximax, phimax = data[-1, 0], data[-1, 2]

    # Loop through configs
    for i_conf, (Mstar, Qbh, smamin, smamax, current_Lmax) in enumerate(run_configurations):

        # Initial profile of star
        Rstar = WD_Radius(Mstar)  # WD Mass-Radius Relation (even for the npoly != 1.5)
        Kentr = (4 * pi) ** (1. / npoly) / (npoly + 1) * (Mstar / phimax) ** (1 - 1. / npoly) * (Rstar / ximax) ** (
                    3. / npoly - 1)
        rhoc = (Mstar / phimax) / (4 * pi * (Rstar / ximax) ** 3)

        # Sma/Lmax
        smaarr = np.linspace(smamin, smamax, Nsma, endpoint=True)
        #current_Lmax = smamax * 1.005 # Uncomment after this n=2.0 run
        print(f'Config: M={Mstar}, Q={Qbh}. Lmax is {current_Lmax:.3f}')

        # create all the relevant directories
        Kentr_dir = savedir + 'rhoc%.3f_Qbh%.2f' % (rhoc, Qbh)
        if not os.path.exists(Kentr_dir):
            os.makedirs(Kentr_dir, exist_ok=True)
        for sma in smaarr:
            sma_dir = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f' % (rhoc, Qbh, sma)
            if os.path.exists(sma_dir):
                loop_shutil(sma_dir)
            os.makedirs(sma_dir, exist_ok=True)

        # Multiprocessing
        jlist_chunks = np.array_split(range(Nsma), Ncpu)
        procs=[]
        for n in range(Ncpu):
            p = Process(
                target=run_tidal_eq,
                args=(jlist_chunks[n], Mstar, Qbh, current_Lmax, smaarr, np.random.randint(10))
            )
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

        print(f'Finished M={Mstar}, Q={Qbh}')
