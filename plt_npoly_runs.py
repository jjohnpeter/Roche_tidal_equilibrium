import sys
from funcs_tidal_eq import *
from para_tidal_eq import *
from dir_info import *
from scipy.ndimage import label
import numpy as np
import pylab as pl
from matplotlib.ticker import AutoMinorLocator
import os
import subprocess
from scipy.interpolate import interp1d
import scipy.optimize as opt
from funcs_tidal_eq import WD_Radius
from math import pi, log
import matplotlib.pyplot as plt
from glob import glob
import re

# For all npoly runs, we use the same fit function (Eggleton's), providing the "A" and "B" values as a
# function of npoly. We make separate plots of these fitted constants, A and B, as a function of n

# Both file names and npoly values have to match
npoly_files = ['npoly_1.7/', 'npoly_1.8/', 'npoly_1.9/', 'npoly_2.0/', 'npoly_2.1/', 'npoly_2.2/', 'npoly_2.3/', 'npoly_2.4/', 'npoly_2.5/']
npoly_vals = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
# Mstar and Qbh combinations for ALL runs (must use the same for each npoly file, but is okay if
# some files have EXTRA Mstar-Qbh runs, as long as the runs below are included in all files

Mstar = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                  0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
Qbh_list = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
                     0.1, 0.2, 0.3, 0.5, 0.7, 0.9])        # also has to match the order
Rstar = WD_Radius(Mstar) # used in both DWD and npoly cases; sets initial conditions

# Create empty lists of A,B to append our values for each npoly into
A_list = []
A_list_err = []
B_list = []
B_list_err = []
for val in range(len(npoly_files)):
    filename = npoly_files[val]
    npoly = npoly_vals[val]

    # Overrides dir_main and save_dir from dir_info.py (in fact, we don't even import dir_info.py_
    dir_main = '/Users/JovanJohnPeter/Roche_tidal_equilibrium/'
    savedir = dir_main + filename
    labels = ['qstar', 'xL1', 'PhitotL1+1.5Q/a', 'xsurf', 'Phitotsurf+1.5Q/a', 'rhoc']

    # Load ximax and phimax from LaneEmden solution of given npoly
    LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
    if not os.path.exists(savedir + LaneEmden_fname):
        # need to run LaneEmden.py to create the polytrope_profile
        os.system('python ' + pydir + 'LaneEmden.py')
    data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
    ximax, phimax = data[-1, 0], data[-1, 2]

    # For DWD case, we manually set rhoc and it was slightly slightly off(negligibly s.t. actual results
    # weren't affected, but file names were, so just for n=1.5, we explicitly list rhoc_list until we
    # get to time to redo DWD sims (ideally with Nresz=150 instead of Nresz=100)
    if npoly == 1.5:
        rhoc_list = [6981.850, 6981.850, 6981.850, 6981.850, 6981.850,
             27927.401, 27927.401, 27927.401, 27927.401, 27927.401, 27927.401,
             62836.652, 62836.652, 62836.652, 62836.652, 62836.652, 62836.652,
             111709.604, 111709.604, 111709.604, 111709.604, 111709.604, 111709.604
            ]
    else:
        rhoc_list = (Mstar / phimax) / (4 * pi * (Rstar / ximax) ** 3)
        rhoc_list = np.around(rhoc_list, 3)
        #print(rhoc_list)
    NK = len(rhoc_list)
    Nswept=24
    masksma = np.zeros((NK, Nswept), dtype=bool)  # mask unused simulations

    last_overf = []
    last_overf_q = []
    first_det = []
    first_det_q = []
    for i in range(NK):
        rhoc = rhoc_list[i]
        Qbh = Qbh_list[i]
        dir_list = glob(savedir + 'rhoc%.3f_Qbh%.2f/*/' % (rhoc, Qbh))
        list_overf = []
        list_overf_q = []
        list_det = []
        list_det_q = []
        for j in range(len(dir_list)):
            # savedir = dir_list[j]
            # fname = savedir + 'output.txt'
            rundir = dir_list[j]
            fname = rundir + 'output.txt'
            with open(fname, 'r') as f:
                # print(f.read())
                if 'converged' not in f.read():  # this simulation isn't useful
                    masksma[i, j] = True
                    continue
            with open(fname, 'r') as f:
                output_file = f.read()
                if 'overflowing' in output_file:
                    f.seek(0)
                    row_new = f.readline()
                    sma_overf, eq_overf = '', ''
                    while len(row_new) > 0:
                        if 'sma' in row_new:
                            sma_overf = row_new
                            sma_overf = float(sma_overf.replace('sma= ', '').strip())
                            print(i, sma_overf)
                        if 'equilibrium result: ' in row_new:
                            eq_overf = row_new
                            # print(eq_overf)
                        row_new = f.readline()

                    eq_overf = eq_overf.replace('equilibrium result: ', '')
                    for lab in labels:
                        eq_overf = eq_overf.replace(lab + '=', '').replace('', '').strip()
                    values = [float(item) for item in eq_overf.split(',')]  # 5 properties of eq sol
                    qstar = values[0]

                    # Add to overflowing lists
                    list_overf.append(sma_overf)
                    list_overf_q.append(qstar)
                if 'detached' in output_file:
                    f.seek(0)
                    row_new = f.readline()
                    sma_det, eq_det = '', ''
                    while len(row_new) > 0:
                        if 'sma' in row_new:
                            if 'sma' in row_new:
                                sma_det = row_new
                                sma_det = float(sma_det.replace('sma= ', '').strip())
                                # print(sma_det)
                        if 'equilibrium result: ' in row_new:
                            eq_det = row_new
                            # print(eq_det)
                        row_new = f.readline()

                    eq_det = eq_det.replace('equilibrium result: ', '')
                    for lab in labels:
                        eq_det = eq_det.replace(lab + '=', '').replace('', '').strip()
                    values = [float(item) for item in eq_det.split(',')]
                    qstar = values[0]

                    # Add to detached lists
                    list_det.append(sma_det)
                    list_det_q.append(qstar)

        # Extract only the last overflow point and the first detached point as the range where overflow occurs
        if len(list_overf) == 0 or len(list_det) == 0:
            print(f"EMPTY CASE: rhoc{rhoc:.3f}_Qbh{Qbh:.2f}")
        last_overf.append(list_overf[-1])
        last_overf_q.append(list_overf_q[-1])
        first_det.append(list_det[0])
        first_det_q.append(list_det_q[0])

    # Convert lists to numpy arrays for simple math between lists
    last_overf = np.array(last_overf)
    last_overf_q = np.array(last_overf_q)
    first_det = np.array(first_det)
    first_det_q = np.array(first_det_q)

    # Define range of possible overflow and associated mass ratio range
    overf_mdpt = (last_overf + first_det) / 2
    overf_left = overf_mdpt - last_overf
    overf_right = first_det - overf_mdpt
    overf_range = [overf_left, overf_right]

    qstar_mdpt = (first_det_q + last_overf_q) / 2
    qstar_left = qstar_mdpt - first_det_q
    qstar_right = last_overf_q - qstar_mdpt

    # Ensures x errors cant be negative when plotting
    qstar_left = np.maximum(qstar_left, 0)
    qstar_right = np.maximum(qstar_right, 0)

    qstar_range = [qstar_left, qstar_right]

    # Extract the specific potential & rho file from critical sma file (determine Niter for the subprocess)
    Niter_list = []
    for i in range(len(overf_mdpt)):
        curr_sma = last_overf[i]
        curr_rhoc = rhoc_list[i]
        curr_Qbh = Qbh_list[i]
        #sma_fold = dir_main + 'rhoc%.3f_Qbh%.2f/sma%.5f/' % (curr_rhoc, curr_Qbh, curr_sma)
        sma_fold = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f/' % (curr_rhoc, curr_Qbh, curr_sma)
        # print(curr_sma)

        # Get list of rho_.txt files (assume Niter in potential_.txt file is same as rho file)
        # Note that the * represents the "deviation" in the pattern matching of the glob function
        rho_files = glob(os.path.join(sma_fold, 'rho*.txt'))
        Niter_opt = []
        for f in rho_files:
            match = re.search(r'rho(\d+)\.txt', f)  # either true or false
            if match:
                Niter_opt.append(int(match.group(1)))  # group 1 extracts first parentheses value (\d+)
        Niter_list.append(max(Niter_opt))

    # Get K values for runs to use to get spherically equiv radius of star
    Kentr_list = ((4 * pi) ** (1. / npoly) / (npoly + 1) * (Mstar / phimax) ** (1 - 1. / npoly) *
                  (Rstar / ximax) ** (3. / npoly - 1))

    # Calculate Rstar associated with converged mass (unpertubed --> stuff this mass into spherical star)
    # Rstar_pert = WD_Radius(last_overf_q)
    Rstar_pert = ((Kentr_list * (npoly + 1)) ** npoly / (4 * pi) *
                  (last_overf_q / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
    RL_over_a = Rstar_pert / last_overf

    # Last_overf_q is the converged mass, stuff it into spherically symmetric star for unperturbed R
    Rstar_unpert = WD_Radius(Mstar)
    print(f'Rstar_unpert={Rstar_unpert}')
    print(f'Rstar_unpert_a = {Rstar_unpert / last_overf}')

    ## Eggleton's approximation
    #RLeff = 0.49 / (0.6 + (Qbh_list / last_overf_q) ** (2. / 3) * np.log(
    #    1 + (Qbh_list / last_overf_q) ** (-1. / 3)))  # perturbed, sma units

    # --- FITS: All of this work to get the following A,B coefficients for all npoly
    # Fitting a Functional Form of Simulations
    Q = Qbh_list / last_overf_q
    Y = Rstar_pert / last_overf

    def RL_fit(Q, A, B):
        return A / (B + (Q ** (2. / 3)) * np.log(1 + (Q ** (-1. / 3))))

    Ainit, Binit = [0.49, 0.6]  # Eggleton fit values as a guess
    start_pars = [Ainit, Binit]
    pars, cov = opt.curve_fit(RL_fit, Q, Y, p0=start_pars)
    [A, B] = pars
    pars_err = np.sqrt(np.diag(cov))
    print('Fit parameters:')
    print(f'A = {A:.5f} +/- {pars_err[0]:.5f}')
    print(f'B = {B:.5f} +/- {pars_err[1]:.5f}')

    # Add fit params to list
    A_list.append(A)
    A_list_err.append(pars_err[0])
    B_list.append(B)
    B_list_err.append(pars_err[1])

# Plot A and B as a function of n
# Eggleton values for A and B (to plot as reference lines)
A_eggl = 0.49
B_eggl = 0.60

# Create figure with 2 vertical panels (same x-axis)
fig, (axA, axB) = plt.subplots(2, 1, figsize=(6,8), sharex=True)

# First subplot (A)
axA.errorbar(npoly_vals, A_list, yerr=A_list_err, fmt = 'o', linewidth=1.2, capsize=3)
axA.axhline(A_eggl, linestyle='--', linewidth=1.0, color = 'gray') # reference line for A

axA.set_ylabel(r'Fitted constant $A$')

# Second subplot (B)
axB.errorbar(npoly_vals, B_list, yerr=B_list_err, fmt = 'o', linewidth=1.2, capsize=3)
axB.axhline(B_eggl, linestyle='--', linewidth=1.0, color='gray') # reference line for B

axB.set_xlabel(r'Polytropic index $n$') # only need to do this for one subplot since both plots share x-axis
axB.set_ylabel(r'Fitted constant $B$')

plt.tight_layout()

save_path = dir_main + 'npoly_FINAL'
file_path = save_path + '/' + 'npoly_A_B_coeffs'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
plt.savefig(file_path, dpi=300)
plt.show()