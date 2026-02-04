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

# Manually set what WD mass used to get radius; used to calculate Eggleton's acrit (not used in RL case)
# has to match what Mstar was used in rhoc list below
Mstar = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                  0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
Qbh_list = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
                     0.08, 0.1, 0.12, 0.2, 0.3, 0.5, 0.7, 0.9])        # also has to match the order
Rstar = WD_Radius(Mstar)

# Kentr for these runs
Kentr = 0.00538754
Nswept = 24

# ---- ximax and phimax only used to estimate Rstar for spherical star
npoly, ximax, phimax = 1.5, 3.65375, 2.71409   # obtained from LaneEmden.py

# Rhoc acrit/RL extractions
rhoc_list = [6981.850, 6981.850, 6981.850, 6981.850, 6981.850,
             27927.401, 27927.401, 27927.401, 27927.401, 27927.401, 27927.401,
             62836.652, 62836.652, 62836.652, 62836.652, 62836.652, 62836.652,
             111709.604, 111709.604, 111709.604, 111709.604, 111709.604, 111709.604,
             111709.604, 111709.604]

# Which Iteration Profile for Every Run in Order
Niter_list = []

dir_main = '/Users/JovanJohnPeter/Roche_tidal_equilibrium/DWD/'
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


    # Calculate Rstar associated with converged mass
    # Okay to use WD Mass-Radius relation for this only for n=1.5 but for other n use polytropic
    # mass radius formula (more general)
    Rstar_WD = WD_Radius(last_overf_q)
    # Rstar_pert = ((Kentr_list * (npoly + 1)) ** npoly / (4 * pi) *
    #               (last_overf_q / phimax) ** (1. - npoly)) ** (1 / (3. - npoly)) * ximax
    RL_over_a = Rstar_WD / last_overf

    if __name__ == '__main__':
        roche_radii = []

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
                roche_radii.append(R_Lobe)

            except subprocess.CalledProcessError as e:
                # If script crashes print error
                print(f"Run {n} FAILED!")
                print("Error Output:")
                print(e.stderr)  # actual Python error from script
                print(e.stdout)  # anything that happened before the crash

                # Append NaN so the arrays stay the same length, preventing later crashes
                roche_radii.append(np.nan)
        roche_radii = np.array(roche_radii)
    print(f'roche_radii: {roche_radii/last_overf}')

    # Eggleton's approximation (don't use/plot for this script)
    RLeff = 0.49 / (0.6+(Qbh_list/last_overf_q) ** (2./3) * np.log(1+(Qbh_list/last_overf_q) ** (-1./3)))  # sma units
    print(f'roche_radii_eggl: {RLeff}')

    # Colorcoding each unique primary WD mass
    colors = {0.1: 'green', 0.2: 'blue', 0.3: 'orange', 0.4: 'purple'}
    uniq_mass = np.unique(Mstar)
    for m in uniq_mass:
        mask = Mstar == m
        # Plots true lobe vol equiv rad (pt source case)
        plt.errorbar(Qbh_list[mask] / last_overf_q[mask], roche_radii[mask] / last_overf[mask],
                     xerr=[qstar_range[0][mask], qstar_range[1][mask]],
                     yerr=[overf_range[0][mask], overf_range[1][mask]],
                     fmt='s',
                     fillstyle='none',
                     markersize=4,
                     color='gray', ecolor='gray',
                     capsize=3, label=None)
        # Plots simulated vol equiv rad (polytropic case)
        plt.errorbar(Qbh_list[mask] / last_overf_q[mask], Rstar_WD[mask] / last_overf[mask],
                     xerr=[qstar_range[0][mask], qstar_range[1][mask]],
                     yerr=[overf_range[0][mask], overf_range[1][mask]],
                     fmt='o',
                     markersize=4,
                     color=colors[m], ecolor=colors[m],
                     capsize=3, label=fr'q = {m}$M_{{\odot}}$')
    # To add labels to define what shape is for what graph, we plot invisible points
    plt.plot([],[], color='k', marker='s', fillstyle='none', linestyle='None', label='True Eggleton')
    plt.plot([],[], color='k', marker='o', linestyle='none', label='Simulated')
    # Plots Eggleton's approximation (his equation)
    #plt.scatter(Qbh_list/last_overf_q, RLeff, s=4, label=r'Eggletons $R_{L}/a$')


    plt.xlabel("Mass Ratio Q/q")
    plt.ylabel(r"$R_{L}$/a")
    #plt.title("Roche Limit vs Mass Ratio")
    plt.legend()

    save_path = dir_main + 'simulated_vs_exact_lobe'
    filename = save_path + '/' + 'RLa.png'
    if not os.path.exists(save_path):
         os.makedirs(save_path, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()

# Compare our simulated values with true lobe (percent errors)
# RL/a values (true eggleton vs simulated)
RLa_true_eggl = roche_radii / last_overf
RLa_sim = Rstar_WD / last_overf
pct_err_mid = 100 * (RLa_sim - RLa_true_eggl) / RLa_true_eggl
yerr_lower = np.zeros_like(pct_err_mid)
yerr_upper = np.zeros_like(pct_err_mid)

# Create a Separate Plot
plt.figure()
#plt.axhline(0, color = 'gray', linewidth = 0.8, linestyle = '--')

# Colorcoding each unique primary WD mass (for simulated data point graph)
colors = {0.1: 'green', 0.2: 'blue', 0.3: 'black', 0.4: 'purple'}
uniq_mass = np.unique(Mstar)

for m in uniq_mass:
    mask = Mstar == m
    Q = Qbh_list / last_overf_q # mass ratio
    plt.errorbar(Q[mask], pct_err_mid[mask],
                 yerr = [yerr_lower[mask], yerr_upper[mask]],
                 fmt='o', markersize=4, capsize=3, linewidth=0.6,
                 color=colors[m], ecolor=colors[m],
                 markeredgecolor='k',
                 label=fr'$q={m}\,M_{{\odot}}$')
#plt.scatter (Q, pct_err, s=30, color = 'royalblue', edgecolors='k')
plt.legend()
plt.xlabel('Mass ratio Q/q')
plt.ylabel('Percent Difference: simulated vs true lobe [%]')

plt.grid(alpha=0.3)
plt.tight_layout()

filename_res = save_path + '/' + "res_plot.png"
plt.savefig(filename_res, dpi=300)
#plt.show()