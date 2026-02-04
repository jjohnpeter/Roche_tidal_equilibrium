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

#from roche_vol_runs import RL_over_a

# this script extracts the effective roche radius for a list of runs

# fixed parameters
#Qbh = 1
#sma = 1.5 # Fix only for RL case

# Manually set what WD mass used to get radius; used to calculate Eggleton's acrit (not used in RL case)
#DWD Case
#Mstar = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
#                  0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])      # has to match what Mstar was used in rhoc list below
#Qbh_list = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
#                     0.08, 0.1, 0.12, 0.2, 0.3, 0.5, 0.7, 0.9])        # also has to match the order

# npoly
Mstar = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                  0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
Qbh_list = np.array([0.2, 0.3, 0.1, 0.2, 0.3, 0.5, 0.7, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
                     0.1, 0.2, 0.3, 0.5, 0.7, 0.9])        # also has to match the order
Rstar = WD_Radius(Mstar) # used in both DWD and npoly cases; sets initial conditions

# Which K to extract acrit/RL from (cases with diff Kentr, old naming convention)
#Kentr_list = [0.00120, 0.00201, 0.00257, 0.00301, 0.00339, 0.00372, 0.00401, 0.00427, 0.00450, 0.00471,\
#              0.00490, 0.00507, 0.00521, 0.00533]
#Kentr_list = [0.00401, 0.00427, 0.00450, 0.00471, 0.00490, 0.00507, 0.00521, 0.00533]

#Kentr_list = [0.11766, 0.15684, 0.20349, 0.24846, 0.3, 0.35, 0.41147, 0.5, 0.6, 0.7, 0.77783]


# Kentr for these runs
#Kentr = 0.00538754
Nswept = 24

# ---- ximax and phimax only used to estimate Rstar for spherical star
#DWD case
npoly, ximax, phimax = 1.5, 3.6537479466, 2.7140896970   # obtained from LaneEmden.py
#npoly case
#npoly, ximax, phimax = 1.6, 3.7758963108, 2.6454196311 # obtained from LaneEmden.py
#npoly, ximax, phimax = 1.8, 4.0449958741, 2.5208856501 # obtained from LaneEmden.py
#npoly, ximax, phimax = 2.3, 4.9067732274, 2.2691207881

# Rhoc acrit/RL extractions
rhoc_list = [6981.850, 6981.850,
             27927.401, 27927.401, 27927.401, 27927.401, 27927.401,
             62836.652, 62836.652, 62836.652, 62836.652, 62836.652, 62836.652,
             111709.604, 111709.604, 111709.604, 111709.604, 111709.604, 111709.604,
            ]

# rhoc_list = [10199.528, 10199.528, 10199.528, 10199.528, 10199.528,
#              40798.112, 40798.112, 40798.112, 40798.112, 40798.112, 40798.112,
#              91795.752, 91795.752, 91795.752, 91795.752, 91795.752, 91795.752,
#              163192.449, 163192.449, 163192.449, 163192.449, 163192.449, 163192.449
#             ]

# rhoc_list = [7905.777, 7905.777, 7905.777, 7905.777, 7905.777,
#              31623.109, 31623.109, 31623.109, 31623.109, 31623.109, 31623.109,
#              71151.996, 71151.996, 71151.996, 71151.996, 71151.996, 71151.996,
#              126492.438, 126492.438, 126492.438, 126492.438, 126492.438, 126492.438
#             ]

# rhoc_list = [20225.968, 20225.968, 20225.968, 20225.968, 20225.968,
#              80903.873, 80903.873, 80903.873, 80903.873, 80903.873, 80903.873,
#              182033.714, 182033.714, 182033.714, 182033.714, 182033.714, 182033.714,
#              323615.491, 323615.491, 323615.491, 323615.491, 323615.491, 323615.491
#             ]

## Which Iteration Profile for Every Run in Order
#Niter_list = []

dir_main = '/Users/JovanJohnPeter/Roche_tidal_equilibrium/DWD/'
#dir_main = '/Users/JovanJohnPeter/Roche_tidal_equilibrium/npoly_2.3/'
labels = ['qstar', 'xL1', 'PhitotL1+1.5Q/a', 'xsurf', 'Phitotsurf+1.5Q/a', 'rhoc']
kplt_list = 0   # =0: plots critical sma   =1: plots Roche Radius; against mass ratio Q/q

#old
#NK = len(Kentr_list)
NK = len(rhoc_list)
masksma = np.zeros((NK, Nswept), dtype=bool)   # mask unused simulations

last_overf = []
last_overf_q = []
first_det = []
first_det_q = []
for i in range(NK):
    # old
    #Kentr = Kentr_list[i]
    #dir_list = glob(dir_main + 'Kentr%.5f/*/' % Kentr, recursive=True)
    rhoc = rhoc_list[i]
    #print(rhoc)
    Qbh = Qbh_list[i]
    #print(Qbh)
    dir_list = glob(dir_main + 'rhoc%.3f_Qbh%.2f/*/' % (rhoc, Qbh) , recursive=True)
    #print(dir_list)
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
    last_overf.append(list_overf[-1])
    last_overf_q.append(list_overf_q[-1])
    first_det.append(list_det[0])
    first_det_q.append(list_det_q[0])

# Convert lists to numpy arrays for simple math between lists
last_overf = np.array(last_overf)
print(last_overf)
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

# Ensures x errors cant be negative when plotting
qstar_left = np.maximum(qstar_left, 0)
qstar_right = np.maximum(qstar_right, 0)

qstar_range = [qstar_left, qstar_right]


# SMA case
if kplt_list == 0:
    # Extract the specific potential & rho file from critical sma file (determine Niter for the subprocess)
    Niter_list = []
    for i in range(len(overf_mdpt)):
        curr_sma = last_overf[i]
        curr_rhoc = rhoc_list[i]
        curr_Qbh = Qbh_list[i]
        sma_fold = dir_main + 'rhoc%.3f_Qbh%.2f/sma%.5f/' % (curr_rhoc, curr_Qbh, curr_sma)
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

    # Get K values for runs to use to get spherically equiv radius of star
    Kentr_list = (4 * pi) ** (1. / npoly) / (npoly + 1) * (Mstar / phimax) ** (1 - 1. / npoly) * (Rstar / ximax) ** (
                3. / npoly - 1)

    # Calculate Rstar associated with converged mass (unpertubed --> stuff this mass into spherical star)
    #Rstar_pert = WD_Radius(last_overf_q)
    Rstar_pert = ((Kentr_list * (npoly + 1)) ** npoly / (4 * pi) *
                  (last_overf_q / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
    RL_over_a = Rstar_pert / last_overf
    #print(f'RL_over_a = {RL_over_a}')


    # Last_overf_q is the converged mass, stuff it into spherically symmetric star for unperturbed R
    #Rstar_unpert = WD_Radius(Mstar)
    #print(f'Rstar_unpert={Rstar_unpert}')
    print(f'Rstar_unpert = {Rstar_pert}')
    print(f'Rstar_unpert_a = {RL_over_a}')

    # Eggleton's approximation
    RLeff = 0.49 / (0.6+(Qbh_list/last_overf_q) ** (2./3) * np.log(1+(Qbh_list/last_overf_q) ** (-1./3)))  # perturbed, sma units
    #RLeff = 0.49 / (0.6 + (Qbh_list / Mstar) ** (2. / 3) * np.log(1 + (Qbh_list / Mstar) ** (-1. / 3))) # unperturbed


    #plt.errorbar(Qbh_list/last_overf_q, Rstar_WD/last_overf, xerr=qstar_range, yerr=overf_range, fmt='none', ecolor='red',
    #             capsize=3, linewidth=0.5, label=r'Simulated Volumetric $R_{L}$/a')
    #plt.errorbar(Qbh_list / last_overf_q, Rstar_unpert/last_overf, xerr=qstar_range, yerr=overf_range,
    #             fmt='none',
    #             ecolor='red',
    #             capsize=3, linewidth=0.5, label=r'Simulated $R_{*}$/a')
    plt.scatter(Qbh_list/last_overf_q, RLeff, s=4, label=r'Eggletons $R_{L}/a$')  #perturbed
    #plt.scatter(Qbh_list/Mstar, RLeff, s=4, label=r'Eggletons $R_{L}/a$')          #unperturbed
    # plt.scatter(last_overf_q, last_overf, c='r', marker='o', label='lower bound')
    # plt.scatter(first_det_q, first_det, c='b', marker='o', label='upper bound')

    # Plotting acrit error of Eggleton's approximation vs polytrope results
    # plt.plot(qstar_left, last_overf / acrit_Eggl, color = 'red')
    # plt.plot(qstar_right, first_det / acrit_Eggl, color = 'blue' )

    # Fitting a Functional Form of Simulations
    Q = Qbh_list / last_overf_q
    Y = Rstar_pert / last_overf
    #Q = Qbh_list / Mstar
    #Y = Rstar_pert / last_overf

    def RL_fit(Q, A, B):
        return A / (B + (Q ** (2. / 3)) * np.log(1 + (Q ** (-1. / 3))))

    Ainit, Binit = [0.49, 0.6]     # Eggleton fit values as a guess
    start_pars = [Ainit, Binit]
    pars, cov = opt.curve_fit(RL_fit, Q, Y, p0=start_pars)
    [A, B] = pars
    pars_err = np.sqrt(np.diag(cov))
    print('Fit parameters:')
    print(f'A = {A:.5f} +/- {pars_err[0]:.5f}')
    print(f'B = {B:.5f} +/- {pars_err[1]:.5f}')

    # Curve fit plot
    Q_range = np.linspace(min(Q)*0.9, max(Q)*1.1, 300)
    plt.plot(Q_range, RL_fit(Q_range, A, B), linestyle = '--', linewidth = 2,
             color='orange', label = fr'Fit: $A={A:.5f}\pm{pars_err[0]:.5f}$, $B={B:.5f}\pm{pars_err[1]:.5f}$')

    # Colorcoding each unique primary WD mass (for simulated data point graph)
    colors = {0.1: 'green', 0.2: 'blue', 0.3: 'black', 0.4: 'purple'}
    uniq_mass = np.unique(Mstar)
    for m in uniq_mass:
        mask = Mstar == m
        #plt.errorbar(Qbh_list[mask] / Mstar[mask], Rstar_unpert[mask] / last_overf[mask],
        #             xerr=[qstar_range[0][mask], qstar_range[1][mask]],
        #             yerr=[overf_range[0][mask], overf_range[1][mask]],
        #             fmt='none',
        #             markersize=4,
        #             color=colors[m], ecolor=colors[m],
        #             capsize=3, linewidth=0.5, label=fr'q = {m}$M_{{\odot}}$')
        plt.errorbar(Qbh_list[mask] / last_overf_q[mask], Rstar_pert[mask] / last_overf[mask],
                     xerr=[qstar_range[0][mask], qstar_range[1][mask]],
                     yerr=[overf_range[0][mask], overf_range[1][mask]],
                     fmt='none',
                     markersize=4,
                     color=colors[m], ecolor=colors[m],
                     capsize=3, linewidth=0.5, label=fr'q = {m}$M_{{\odot}}$')

    plt.xlabel("Mass Ratio Q/q")
    plt.ylabel(r"$R_{*}/a_{crit}$")
    #plt.title("Roche Limit vs Mass Ratio: n = 1.5 Polytrope") # WD case
    plt.title(f"Roche Limit vs Mass Ratio: n = {npoly} Polytrope")
    plt.legend()

    save_path = dir_main + 'vol_roche_rad'
    filename = save_path + '/' + 'lobe_eff_rad.png'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()

# Compare Eggleton curve with simulated curve and calculate percent difference
# New, "perturbed mass" --> into a spherically symmetric unperturbed star
pct_err_overf = 100 * ((Rstar_pert / last_overf) - RLeff) / RLeff #unperturbed (last overf case)
pct_err_det = 100 * ((Rstar_pert / first_det) - RLeff) / RLeff # first detached case

# Midpoint between first det and last overf
pct_err_mid = 0.5 * (pct_err_overf + pct_err_det)

# Asymmetric y errors
yerr_lower = pct_err_mid - np.minimum(pct_err_overf, pct_err_det)
yerr_upper = np.maximum(pct_err_overf, pct_err_det) - pct_err_mid
yerr_asym = np.vstack([yerr_lower, yerr_upper])

# Create a Separate Plot
plt.figure()
#plt.axhline(0, color = 'gray', linewidth = 0.8, linestyle = '--')

Q = Qbh_list / last_overf_q

# Colorcoding each unique primary WD mass (for simulated data point graph)
colors = {0.1: 'green', 0.2: 'blue', 0.3: 'black', 0.4: 'purple'}
uniq_mass = np.unique(Mstar)

for m in uniq_mass:
    mask = Mstar == m
    #plt.scatter(Q[mask], pct_err[mask], s=30, color = colors[m],
    #            edgecolors='k', label=fr'$q={m}\,M_{{\odot}}$')
    plt.errorbar(Q[mask], pct_err_mid[mask],
                 yerr = [yerr_lower[mask], yerr_upper[mask]],
                 fmt='o', markersize=4, capsize=3, linewidth=0.6,
                 color=colors[m], ecolor=colors[m],
                 markeredgecolor='k',
                 label=fr'$q={m}\,M_{{\odot}}$')
#plt.scatter (Q, pct_err, s=30, color = 'royalblue', edgecolors='k')
plt.legend()
plt.xlabel('Mass ratio Q/q')
plt.ylabel('Percent Difference from Eggleton [%]')
plt.title(f'Deviation of Simulated Radii from Eggletons Radii: n = {npoly}')

plt.grid(alpha=0.3)
plt.tight_layout()

filename = save_path + '/' + "res_plot.png"
plt.savefig(filename, dpi=300)

#--------------PLOT residuals (how much each simulated data pt deviates from our fit. For
# context, Eggleton's fit is accurate to within 1% of true lobe vol-equiv radius)
plt.figure()
for m in uniq_mass:
    mask = Mstar == m
    resid = (RL_fit(Q[mask], A, B) - (Rstar_pert[mask] / last_overf[mask]))/(Rstar_pert[mask] / last_overf[mask])
    resid = resid * 100 # percent residuals
    plt.errorbar(Q[mask], resid,
                 yerr=[yerr_lower[mask], yerr_upper[mask]],
                 fmt='o', markersize=4, capsize=3, linewidth=0.6,
                 color=colors[m], ecolor=colors[m],
                 markeredgecolor='k',
                 label=fr'$q={m}\,M_{{\odot}}$')
plt.legend()
plt.xlabel('Mass ratio Q/q')
plt.ylabel('Fit Residuals (Percent Deviation) [%]')
plt.title(f'n = {npoly}')

plt.grid(alpha=0.3)
plt.tight_layout()

filename = save_path + '/' + "fit_residuals.png"
plt.savefig(filename, dpi=300)

#--------------PLOT exact lobe vs simulated radii for this particular run-------------
if __name__ == '__main__':
    lobe_radii = []

    for n in range(NK):
        try:
            procs = subprocess.run(
                ['python', 'eggl_exact_radius.py',
                 str(rhoc_list[n]), str(last_overf[n]),
                 str(Niter_list[n]), str(Qbh_list[n])],
                capture_output=True,
                text=True,
                check=True
            )

            # If successful, get radius value
            R_Lobe = float(procs.stdout.strip())
            print(f"Run {n} Success: {R_Lobe}")
            lobe_radii.append(R_Lobe)

        except subprocess.CalledProcessError as e:
            # If script crashes print error
            print(f"Run {n} FAILED")
            print("Error Output:")
            print(e.stderr)  # actual Python error from script
            print(e.stdout)  # anything that happened before the crash

            # Append NaN so the arrays stay the same length, preventing later crashes
            lobe_radii.append(np.nan)
    lobe_radii = np.array(lobe_radii)
print(f'roche_radii_exact_lobe: {lobe_radii / last_overf}')

# Curve fit plot - Simulated vs Exact Lobe
plt.figure()
# 1. Plotting the fit for simulated data pts
Q_range = np.linspace(min(Q)*0.9, max(Q)*1.1, 300)
plt.plot(Q_range, RL_fit(Q_range, A, B), linestyle = '--', linewidth = 2,
    color='orange', label = fr'Fit: $A={A:.5f}\pm{pars_err[0]:.5f}$, $B={B:.5f}\pm{pars_err[1]:.5f}$')

# 1b. Plotting the actual data pts (over the fit)
# Colorcoding each unique primary mass (for simulated data point graph)
colors = {0.1: 'green', 0.2: 'blue', 0.3: 'black', 0.4: 'purple'}
uniq_mass = np.unique(Mstar)
for m in uniq_mass:
    mask = Mstar == m
    plt.errorbar(Qbh_list[mask] / last_overf_q[mask], Rstar_pert[mask] / last_overf[mask],
                 xerr=[qstar_range[0][mask], qstar_range[1][mask]],
                 yerr=[overf_range[0][mask], overf_range[1][mask]],
                 fmt='none',
                 markersize=4,
                 color=colors[m], ecolor=colors[m],
                 capsize=3, linewidth=0.5, label=fr'q = {m}$M_{{\odot}}$')

# 2. Overlaying associated (point source) exact lobe vol equiv rad case
plt.scatter(Qbh_list / last_overf_q, lobe_radii / last_overf,
             marker='o', s=15, color='gray', edgecolor='black')

plt.xlabel("Mass Ratio Q/q")
plt.ylabel(r"$R_{*}/a_{crit}$")
plt.legend()

filename = save_path + '/' + 'sims_vs_pt_source_lobe_exact.png'
plt.savefig(filename, dpi=300)

#-------ERROR plot for sims vs exact lobe graph
# Compare Eggleton curve with simulated curve and calculate percent difference
# New, "perturbed mass" --> into a spherically symmetric unperturbed star
RL_exact_lobe = lobe_radii / last_overf
pct_err_overf = 100 * ((Rstar_pert / last_overf) - RL_exact_lobe) / RL_exact_lobe #unperturbed (last overf case)
pct_err_det = 100 * ((Rstar_pert / first_det) - RL_exact_lobe) / RL_exact_lobe # first detached case

# Midpoint between first det and last overf
pct_err_mid = 0.5 * (pct_err_overf + pct_err_det)

# Asymmetric y errors
yerr_lower = pct_err_mid - np.minimum(pct_err_overf, pct_err_det)
yerr_upper = np.maximum(pct_err_overf, pct_err_det) - pct_err_mid
yerr_asym = np.vstack([yerr_lower, yerr_upper])

# Create a Separate Plot
plt.figure()

# Colorcoding each unique primary mass (for simulated data point graph)
colors = {0.1: 'green', 0.2: 'blue', 0.3: 'black', 0.4: 'purple'}
uniq_mass = np.unique(Mstar)

for m in uniq_mass:
    mask = Mstar == m
    plt.errorbar(Q[mask], pct_err_mid[mask],
                 yerr = [yerr_lower[mask], yerr_upper[mask]],
                 fmt='o', markersize=4, capsize=3, linewidth=0.6,
                 color=colors[m], ecolor=colors[m],
                 markeredgecolor='k',
                 label=fr'$q={m}\,M_{{\odot}}$')
plt.legend()
plt.xlabel('Mass ratio Q/q')
plt.ylabel('Percent Difference from Exact Lobe [%]')
plt.title(f'Deviation of Simulated Radii from Exact Lobe Radii: n = {npoly}')

plt.grid(alpha=0.3)
plt.tight_layout()

filename = save_path + '/' + "sims_vs_pt_source_lobe_exact_err.png"
plt.savefig(filename, dpi=300)