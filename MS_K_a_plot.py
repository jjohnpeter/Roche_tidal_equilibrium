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
from math import pi, log
import matplotlib.pyplot as plt
from glob import glob
import re


# this script extracts the effective roche radius for the MS_K case

# Manually set what WD mass used to get radius; used to calculate Eggleton's acrit (not used in RL case)
Mstar = np.arange(0.15, 0.85, 0.05)      # has to match what Mstar was used in rhoc list below
Qbh_list = Mstar * 4                     # fixed mass ratio
Rstar = MS_Radius(Mstar)

# Number of runs per K
Nswept = 24

# ---- ximax and phimax only used to estimate Rstar for spherical star
npoly, ximax, phimax = 1.5, 3.6537479466, 2.7140896970   # obtained from LaneEmden.py

# Which K to extract acrit
Kentr_list = (4*pi)**(1./npoly)/(npoly+1)*(Mstar/phimax)**(1-1./npoly)*(Rstar/ximax)**(3./npoly-1)

# Which Iteration Profile for Every Run in Order
Niter_list = []

dir_main = '/Users/JovanJohnPeter/Roche_tidal_equilibrium/MS_K_runs/'
labels = ['qstar', 'xL1', 'PhitotL1+1.5Q/a', 'xsurf', 'Phitotsurf+1.5Q/a', 'rhoc']
kplt_list = 0   # =0: plots critical sma   =1: plots Roche Radius; against mass ratio Q/q

#old
NK = len(Kentr_list)
masksma = np.zeros((NK, Nswept), dtype=bool)   # mask unused simulations

last_overf = []
last_overf_q = []
first_det = []
first_det_q = []
for i in range(NK):
    Kentr = Kentr_list[i]
    dir_list = glob(dir_main + 'Kentr%.5f/*/' % Kentr, recursive=True)
    Qbh = Qbh_list[i]

    list_overf = []
    list_overf_q = []
    list_det = []
    list_det_q = []
    for j in range(len(dir_list)):
        savedir = dir_list[j]
        fname = savedir + 'output.txt'
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
                        if kplt_list == 0 and 'sma' in row_new:
                            sma_overf = row_new
                            sma_overf = float(sma_overf.replace('sma= ', '').strip())
                            # print(sma_overf)
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
                    if kplt_list == 0:
                        list_overf.append(sma_overf)
                    if kplt_list == 1:
                        Rstar_conv = ((Kentr * (npoly + 1)) ** npoly / (4 * pi) *
                                      (qstar / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
                        list_overf.append(Rstar_conv)
                    list_overf_q.append(qstar)

                if 'detached' in output_file:
                    f.seek(0)
                    row_new = f.readline()
                    sma_det, eq_det = '', ''
                    while len(row_new) > 0:
                        if kplt_list == 0 and 'sma' in row_new:
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
                    if kplt_list == 0:
                        list_det.append(sma_det)
                    if kplt_list == 1:
                        Rstar_conv = ((Kentr * (npoly + 1)) ** npoly / (4 * pi) *
                                      (qstar / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
                        list_det.append(Rstar_conv)
                    list_det_q.append(qstar)

    #print(list_overf)
    #print(list_overf_q)
    #print(list_det)
    #print(list_det_q)

    # Extract only the last overflow point and the first detached point as the range where overflow occurs
    #print(f'Finished Kentr={Kentr:.5f}, overflow_count={len(list_overf)}, det_count={len(list_det)}')
    last_overf.append(list_overf[-1])
    last_overf_q.append(list_overf_q[-1])
    first_det.append(list_det[0])
    first_det_q.append(list_det_q[0])

# Convert lists to numpy arrays for simple math between lists
last_overf = np.array(last_overf)
last_overf_q = np.array(last_overf_q)
first_det = np.array(first_det)
first_det_q = np.array(first_det_q)
#print(last_overf)
#print(last_overf_q)
#print(first_det)
#print(first_det_q)

# Define range of possible overflow and associated mass ratio range
overf_mdpt = (last_overf + first_det) / 2
overf_left = overf_mdpt - last_overf
overf_right = first_det - overf_mdpt
overf_range = [overf_left, overf_right]


qstar_mdpt = (first_det_q + last_overf_q) / 2
qstar_left = qstar_mdpt - first_det_q
qstar_right = last_overf_q - qstar_mdpt
qstar_range = [qstar_left, qstar_right]


# SMA case
if kplt_list == 0:
    # Extract the specific potential & rho file from critical sma file (determine Niter for the subprocess)
    Niter_list = []
    for i in range(len(overf_mdpt)):
        curr_sma = last_overf[i]
        #curr_rhoc = rhoc_list[i]
        curr_Kentr = Kentr_list[i]
        curr_Qbh = Qbh_list[i]
        sma_fold = dir_main + 'Kentr%.5f/sma%.5f/' % (curr_Kentr, curr_sma)
        #print(curr_sma)

    # Get list of rho_.txt files (assume Niter in potential_.txt file is same as rho file)
    # Note that the * represents the "deviation" in the pattern matching of the glob function
        rho_files = glob(os.path.join(sma_fold, 'rho*.txt'))
        Niter_opt = []
        for f in rho_files:
            match = re.search(r'rho(\d+)\.txt', f) # either true or false
            if match:
                Niter_opt.append(int(match.group(1))) # group 1 extracts first parentheses value (\d+)
        Niter_list.append(max(Niter_opt))


    # Calculate Rstar associated with converged mass (not unpertubed!)
    #Rstar_pert = MS_Radius(last_overf_q) # this is wrong
    Rstar_pert = ((Kentr_list * (npoly + 1)) ** npoly / (4 * pi) *
                 (last_overf_q / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
    RL_over_a = Rstar_pert / last_overf
    #print(f'RL_over_a = {RL_over_a}')


    #Rstar_unpert = MS_Radius(Mstar)
    #print(f'Rstar_unpert={Rstar_unpert}')
    #print(f'Rstar_unpert_a = {Rstar_unpert/last_overf}')

    # Eggleton's approximation
    RLeff = 0.49 / (0.6+(Qbh_list/last_overf_q) ** (2./3) * np.log(1+(Qbh_list/last_overf_q) ** (-1./3)))  # perturbed, sma units
    #RLeff = 0.49 / (0.6 + (Qbh_list / Mstar) ** (2. / 3) * np.log(1 + (Qbh_list / Mstar) ** (-1. / 3))) # unperturbed


    plt.scatter(Kentr_list, RLeff, s=4, label=r'Eggletons $R_{L}/a$', color='red')  #unperturbed


    # Fitting a Functional Form of Simulations
    #Q = Qbh_list / Mstar
    #Y = Rstar_unpert / last_overf

    #def RL_fit(Q, A, B):
    #    return A / (B + (Q ** (2. / 3)) * np.log(1 + (Q ** (-1. / 3))))

    #Ainit, Binit = [0.49, 0.6]     # Eggleton fit values as a guess
    #start_pars = [Ainit, Binit]
    #pars, cov = opt.curve_fit(RL_fit, Q, Y, p0=start_pars)
    #[A, B] = pars
    #pars_err = np.sqrt(np.diag(cov))
    #print('Fit parameters:')
    #print(f'A = {A:.5f} +/- {pars_err[0]:.5f}')
    #print(f'B = {B:.5f} +/- {pars_err[1]:.5f}')

    # Curve fit plot
    #Q_range = np.linspace(min(Q)*0.9, max(Q)*1.1, 300)
    #plt.plot(Q_range, RL_fit(Q_range, A, B), linestyle = '--', linewidth = 2,
    #         color='orange', label = fr'Fit: $A={A:.3f}\pm{pars_err[0]:.3f}$, $B={B:.3f}\pm{pars_err[1]:.3f}$')

    # plt.errorbar(Kentr_list, Rstar_unpert / last_overf,
    #             xerr=[qstar_range[0], qstar_range[1]],
    #             yerr=[overf_range[0], overf_range[1]],
    #             fmt='none',
    #             markersize=4,
    #             capsize=3, linewidth=0.5, label=r'Simulated $R_{L}/a$')
    plt.errorbar(Kentr_list, Rstar_pert / last_overf,
                 xerr=[qstar_range[0], qstar_range[1]],
                 yerr=[overf_range[0], overf_range[1]],
                 fmt='none',
                 markersize=4,
                 capsize=3, linewidth=0.5, label=r'Simulated $R_{L}/a$')


    plt.xlabel("Entropy K")
    plt.ylabel(r"$R_{*}/a_{crit}$")
    plt.title("Roche Limit vs Mass Ratio: MS_K case, M=0.15,...,0.80, Qbh=4*M")
    plt.legend()

    save_path = dir_main + 'vol_roche_rad_MS_K'
    filename = save_path + '/' + 'lobe_eff_rad_MS_K.png'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()

# Compare Eggleton curve with simulated curve and calculate percent difference
#pct_err_overf = 100 * ((Rstar_unpert / last_overf) - RLeff) / RLeff #unperturbed (last overf case)
#pct_err = 100 * (RL_over_a  - RLeff) / RLeff                 #perturbed
#pct_err_det = 100 * ((Rstar_unpert / first_det) - RLeff) / RLeff # (first det case)
#print(f'pct_err_overf = {pct_err_overf}')
#print(f'pct_err_det = {pct_err_det}')

# Proper: use converged mass --> put into spherically symmetric system
pct_err_overf = 100 * ((Rstar_pert / last_overf) - RLeff) / RLeff # (last overf case)
pct_err_det = 100 * ((Rstar_pert / first_det) - RLeff) / RLeff # (first det case)

# Midpoint between the last overf and first det
pct_err_mid = 0.5 * (pct_err_overf + pct_err_det)

# Asymmetric y errors
yerr_lower = pct_err_mid - np.minimum(pct_err_overf, pct_err_det)
yerr_upper = np.maximum(pct_err_overf, pct_err_det) - pct_err_mid
yerr_asym = np.vstack([yerr_lower, yerr_upper])

# Create a Separate Plot
plt.figure()
#plt.axhline(0, color = 'gray', linewidth = 0.8, linestyle = '--')

# Colorcoding each unique primary WD mass (for simulated data point graph)


#plt.scatter(Kentr_list, pct_err, s=30, edgecolors='k', color='gray')
plt.errorbar(Kentr_list, pct_err_mid,
             yerr=yerr_asym,
             fmt='o', markersize=5, capsize=3,
             linewidth=0.6, color='gray', ecolor='black',
             markeredgecolor='k')


#plt.scatter (Q, pct_err, s=30, color = 'royalblue', edgecolors='k')
plt.legend()
plt.xlabel('Entropy K')
plt.ylabel('Percent Difference from Eggleton [%]')
plt.title('Deviation of Simulated Radii from Eggletons Radii: MS_K case')

plt.grid(alpha=0.3)
plt.tight_layout()

filename_res = save_path + '/' + "res_plot_MS_K.png"
plt.savefig(filename_res, dpi=300)
#plt.show()