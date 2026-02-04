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

# n=1.5 runs: Accidentally deleted (first) det case for M=0.2, Qbh=0.10 (sma=0.08136). Note
#             the next det was 0.08139, not that it should matter. Need Nresz=150 runs anyway
# n=1.6 runs: Runs that are done:
# M=0.2, Q=0.1: completely done (0.00002); M=0.2, Q=0.2;
#

# Transition points for n=1.8 and n=2.0 (e.g)
# these current n_low and n_high are each for diff npoly
# n_low = np.array([
#     0.08793, 0.09778, 0.11258, 0.12396, 0.13336,
#     0.05076, 0.05892, 0.06491, 0.07394, 0.08093, 0.08677,
#     0.04107, 0.04704, 0.05147, 0.05820, 0.06345, 0.06781,
#     0.03551, 0.04028, 0.04384, 0.04929, 0.05354, 0.05709
#     ])
#
# n_high = np.array([
#     0.08796, 0.09788, 0.11266, 0.12406, 0.13348,
#     0.05078, 0.05896, 0.06494, 0.074, 0.08098, 0.0868,
#     0.04106, 0.04706, 0.0515, 0.05824, 0.06346, 0.06786,
#     0.03552, 0.0403, 0.04386, 0.0493, 0.05356, 0.05712
# ])

# previous npoly (last) overflow points (so if we are running n=2.1, these are for n=2)
n_overf = np.array([0.08734, 0.11346])
#n_overf = np.array([0.08734, 0.11346, 0.12502, 0.13440])
#M=0.2: Qbh=0.1, M=0.3: Qbh=0.2,0.3, M=0.4: Qbh=0.1, 0.2, 0.3, 0.5, 0.7
# Define the M and Q order to match the arrays

mq_order = [(0.20, 0.90), (0.10, 0.50)]
#mq_order = [(0.20, 0.90), (0.10, 0.50), (0.10, 0.70), (0.10, 0.90)]
# Manual list of Lmax
# Must have 23 numbers, matching the order of mq_order above

Lmax_list = [0.087, 0.113]
#Lmax_list = [0.087, 0.113, 0.125, 0.134]

# All runs
# mq_order = [
#     (0.10,  0.20), (0.10, 0.30), (0.10, 0.50), (0.10, 0.70), (0.10, 0.90),
#     (0.20, 0.10), (0.20, 0.20), (0.20, 0.30), (0.20, 0.50), (0.20, 0.70), (0.20, 0.90),
#     (0.30, 0.10), (0.30, 0.20), (0.30, 0.30), (0.30, 0.50), (0.30, 0.70), (0.30, 0.90),
#     (0.40, 0.10), (0.40, 0.20), (0.40, 0.30), (0.40, 0.50), (0.40, 0.70), (0.40, 0.90)
# ]

# Manual list of Lmax
# Must have 23 numbers, matching the order of mq_order above
# Lmax_list = [
#     0.088, 0.099, 0.113, 0.125, 0.134,  # M=0.1 runs
#     0.051, 0.059, 0.065, 0.074, 0.081, 0.087,  # M=0.2 runs
#     0.041, 0.047, 0.052, 0.059, 0.064, 0.068,  # M=0.3 runs
#     0.036, 0.041, 0.044, 0.049, 0.053, 0.058  # M=0.4 runs
# ]

# # Calculate midpts and ranges
# n_avg = (n_low + n_high) / 2
# n_run_min = n_avg - 0.00005
# n_run_max = n_avg + 0.00005

# Lower and upper bounds
n_run_min = n_overf
n_run_max = n_overf

# Build the run_configurations list automatically
run_configurations = []

# Check if list lengths match
if len(Lmax_list) != len(mq_order):
    print(f"WARNING: You have {len(mq_order)} configs but {len(Lmax_list)} Lmax values!")

for i in range(len(mq_order)):
    M, Q = mq_order[i]
    # Create 6(variable) points centered on the average
    sma_sweep = np.linspace(n_run_min[i], n_run_max[i], 1, endpoint=True)
    # Use the manual Lmax from your list
    current_Lmax = Lmax_list[i]
    run_configurations.append((M, Q, sma_sweep, current_Lmax))

Ncpu = 2

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

    # Iterate through the automatically generated configurations
    for i_conf, (Mstar, Qbh, sma_manual, current_Lmax) in enumerate(run_configurations):

        Rstar = WD_Radius(Mstar)
        Kentr = (4 * pi) ** (1. / npoly) / (npoly + 1) * (Mstar / phimax) ** (1 - 1. / npoly) * (Rstar / ximax) ** (
                    3. / npoly - 1)
        rhoc = (Mstar / phimax) / (4 * pi * (Rstar / ximax) ** 3)

        Kentr_dir = savedir + 'rhoc%.3f_Qbh%.2f' % (rhoc, Qbh)
        if not os.path.exists(Kentr_dir):
            os.makedirs(Kentr_dir, exist_ok=True)

        for sma in sma_manual:
            sma_dir = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f' % (rhoc, Qbh, sma)
            if os.path.exists(sma_dir):
                loop_shutil(sma_dir)
            os.makedirs(sma_dir, exist_ok=True)

            all_tasks.append((Mstar, Qbh, current_Lmax, sma))

    total_jobs = len(all_tasks)
    print(f"Setup Complete. Starting Pool with {Ncpu} processors.")
    print(f"Total sims to run: {total_jobs}")

    with Pool(processes=Ncpu) as pool:
        pool.map(run_single_simulation, all_tasks)

    print("All sims finished.")