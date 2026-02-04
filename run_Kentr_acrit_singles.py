import numpy as np
import time
import shutil
import os
import subprocess
from math import pi
from multiprocessing import Pool
from funcs_tidal_eq import WD_Radius
from para_tidal_eq import *
from dir_info import *

# Runs multiple runs with different M + Qbh combos at once (usually for reduced grid error by
#   doing an extra run for each M + Qbh
# Format: (Mstar, Qbh, sma_manual, Lmax_manual)
run_configurations = [
    (0.10, 0.20, 0.08830, 0.1),
    (0.10, 0.30, 0.09834, 0.1),
    (0.10, 0.50, 0.11317, 0.120),
    (0.10, 0.70, 0.12444, 0.15),
    (0.10, 0.90, 0.13395, 0.135),
    (0.20, 0.10, 0.05091, 0.051),
    (0.20, 0.20, 0.05913, 0.1),
    (0.20, 0.30, 0.06521, 0.07),
    (0.20, 0.50, 0.07430, 0.075),
    (0.20, 0.70, 0.08134, 0.084),
    (0.20, 0.90, 0.08718, 0.09),
    (0.30, 0.10, 0.04116, 0.06),
    (0.30, 0.20, 0.04726, 0.05),
    (0.30, 0.30, 0.05171, 0.055),
    (0.30, 0.50, 0.05847, 0.06),
    (0.30, 0.70, 0.06374, 0.065),
    (0.30, 0.90, 0.06813, 0.07),
    (0.40, 0.08, 0.03438, 0.04),
    (0.40, 0.10, 0.03558, 0.04),
    (0.40, 0.12, 0.03671, 0.04),
    (0.40, 0.20, 0.04038, 0.05),
    (0.40, 0.30, 0.04400, 0.05),
    (0.40, 0.50, 0.04951, 0.051),
    (0.40, 0.70, 0.05382, 0.055),
    (0.40, 0.90, 0.05738, 0.060)
]

Ncpu = 6


def run_single_simulation(args):
    Mstar, Qbh, Lmax, sma = args

    cmd_list = [
        'python',
        dir_main + 'tidal_equilibrium_acrit.py',
        '%.2f' % Mstar,
        '%.5f' % sma,
        '%.2f' % Qbh,
        '%.3f' % Lmax
    ]

    try:
        subprocess.run(cmd_list, check=True)
        print(f"DONE: M={Mstar}, Q={Qbh}, sma={sma:.5f}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR in run: M={Mstar}, Q={Qbh}, sma={sma:.5f}")


def loop_shutil(path, attempts=10):
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, FileNotFoundError):
            time.sleep(1)


if __name__ == '__main__':
    LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
    if not os.path.exists(savedir + LaneEmden_fname):
        os.system('python ' + pydir + 'LaneEmden.py')
    data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
    ximax, phimax = data[-1, 0], data[-1, 2]

    all_tasks = []

    print("Preparing directories and generating task list...")

    for i_conf, (Mstar, Qbh, sma_manual, current_Lmax) in enumerate(run_configurations):

        Rstar = WD_Radius(Mstar)
        Kentr = (4 * pi) ** (1. / npoly) / (npoly + 1) * (Mstar / phimax) ** (1 - 1. / npoly) * (Rstar / ximax) ** (
                    3. / npoly - 1)
        rhoc = (Mstar / phimax) / (4 * pi * (Rstar / ximax) ** 3)

        Kentr_dir = savedir + 'rhoc%.3f_Qbh%.2f' % (rhoc, Qbh)
        if not os.path.exists(Kentr_dir):
            os.makedirs(Kentr_dir, exist_ok=True)

        sma = sma_manual
        sma_dir = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f' % (rhoc, Qbh, sma)
        if os.path.exists(sma_dir):
            loop_shutil(sma_dir)
        os.makedirs(sma_dir, exist_ok=True)

        all_tasks.append((Mstar, Qbh, current_Lmax, sma))

    total_jobs = len(all_tasks)
    print(f"Setup Complete. Starting Pool with {Ncpu} workers.")
    print(f"Total simulations to run: {total_jobs}")

    with Pool(processes=Ncpu) as pool:
        pool.map(run_single_simulation, all_tasks)

    print("All simulations finished.")