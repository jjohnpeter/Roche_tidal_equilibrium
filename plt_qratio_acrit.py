import numpy as np
import pylab as pl
from matplotlib.ticker import AutoMinorLocator
import os
from scipy.interpolate import interp1d
import scipy.optimize as opt
from funcs_tidal_eq import WD_Radius
from math import pi, log
import matplotlib.pyplot as plt
from glob import glob

# fixed parameters
#Qbh = 1
sma = 1.5 # Fix only for RL case


# Manually set what WD mass used to get radius; used to calculate Eggleton's acrit (not used in RL case)
Mstar = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                  0.4, 0.4, 0.4, 0.4, 0.4, 0.4])      # has to match what Mstar was used in rhoc list below
Qbh_list = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
                     0.1, 0.2, 0.3, 0.5, 0.7, 0.9])        # also has to match the order
#Rstar = WD_Radius(Mstar)

# Which K to extract acrit/RL from (cases with diff Kentr, old naming convention)
#Kentr_list = [0.00120, 0.00201, 0.00257, 0.00301, 0.00339, 0.00372, 0.00401, 0.00427, 0.00450, 0.00471,\
#              0.00490, 0.00507, 0.00521, 0.00533]
#Kentr_list = [0.00401, 0.00427, 0.00450, 0.00471, 0.00490, 0.00507, 0.00521, 0.00533]

#Kentr_list = [0.11766, 0.15684, 0.20349, 0.24846, 0.3, 0.35, 0.41147, 0.5, 0.6, 0.7, 0.77783]


# Kentr for these runs
Kentr = 0.00538754


Nswept = 24

# ---- ximax and phimax only used to estimate Rstar for spherical star
npoly, ximax, phimax = 1.5, 3.65375, 2.71409   # obtained from LaneEmden.py

# Rhoc acrit/RL extractions
rhoc_list = [6981.850, 6981.850, 6981.850, 6981.850, 6981.850,
             27927.401, 27927.401, 27927.401, 27927.401, 27927.401, 27927.401,
             62836.652, 62836.652, 62836.652, 62836.652, 62836.652, 62836.652,
             111709.604, 111709.604, 111709.604, 111709.604, 111709.604, 111709.604]

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

    # Calculate Rstar associated with converged mass
    Rstar = WD_Radius(qstar_mdpt)
    RL_over_a = Rstar / overf_mdpt

    # Eggleton's approximation
    RLeff = 0.49 / (0.6+(Qbh_list/qstar_mdpt) ** (2./3) * np.log(1+(Qbh_list/qstar_mdpt) ** (-1./3)))  # sma units
    acrit_Eggl = Rstar / RLeff

    # Fitting (using midpoints instead of the whole range just for a general fit not extremely precise)


    plt.errorbar(Qbh_list/qstar_mdpt, overf_mdpt, xerr=qstar_range, yerr=overf_range, fmt='none', ecolor='red',
                 capsize=3, linewidth=0.5, label=r'Simulated $a_{crit}$')
    plt.scatter(Qbh_list/qstar_mdpt, acrit_Eggl, s=4, label=r'Eggletons $a_{crit}$')
    # plt.scatter(last_overf_q, last_overf, c='r', marker='o', label='lower bound')
    # plt.scatter(first_det_q, first_det, c='b', marker='o', label='upper bound')

    # Plotting acrit error of Eggleton's approximation vs polytrope results
    # plt.plot(qstar_left, last_overf / acrit_Eggl, color = 'red')
    # plt.plot(qstar_right, first_det / acrit_Eggl, color = 'blue' )


    plt.xlabel("Mass Ratio Q/q")
    plt.ylabel(r"$a_{crit}$ ($R_{\odot}$)")
    plt.title("Critical SMA vs Mass Ratio")
    plt.legend()

    save_path = dir_main + 'qratio_acrit'
    filename = save_path + '/' + 'acrit.png'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()

# RL Case
if kplt_list == 1:

    # Eggleton's approximation
    RLeff = 0.49*sma / (0.6+(Qbh_list/qstar_mdpt) ** (2./3) * np.log(1+(Qbh_list/qstar_mdpt) ** (-1. / 3))) # [Rsun]
    #RLeff_right = 0.49*sma / (0.6+(Qbh/qstar_right) ** (2./3) * np.log(1+(Qbh/qstar_right) ** (-1. / 3)))
    #RLeff_left = 0.49*sma / (0.6+(Qbh/qstar_left) ** (2./3) * np.log(1+(Qbh/qstar_left) ** (-1. / 3)))

    # Fitting
    Q = Qbh_list/qstar_mdpt
    def RL_fit(Q, A, B):
        return A*sma / (B+(Q) ** (2./3) * np.log(1+(Q) ** (-1. / 3)))

    Ainit, Binit = [0.49, 0.6]
    start_pars = [Ainit, Binit]
    pars, cov = opt.curve_fit(RL_fit, Q, overf_mdpt, p0=start_pars)
    [A,B] = pars
    pars_err = np.sqrt(np.diag(cov))
    print(np.transpose([pars, pars_err]))

    # Curve fit plot
    Q_range = np.linspace(0.05, 3.3)
    plt.plot(Q_range, RL_fit(Q_range, A, B), color='orange')

    # Actual plots, sims and Eggleton's
    plt.errorbar(Qbh_list/qstar_mdpt, overf_mdpt, xerr=np.abs(qstar_range), yerr = np.abs(overf_range), fmt='none',
                 ecolor='red', capsize=3, linewidth=2, label=r'Simulated $R_{L}$')
    plt.scatter(Qbh_list/qstar_mdpt, RLeff, s=4, label=r'Eggletons $R_{L}$')


    # plt.scatter(Qbh/last_overf_q, last_overf, c='r', marker='o', label='lower bound')
    # plt.scatter(Qbh/first_det_q, first_det, c='b', marker='o', label='upper bound')

    # To plot error of Eggleton's approximation based on simulation results
    #plt.scatter(Qbh/qstar_mdpt, overf_mdpt / RLeff, color = 'red')
    #plt.scatter(Qbh/qstar_right, first_det / RLeff_right, color = 'blue')
    #plt.scatter(qstar_left, last_overf / RLeff_left, color = 'orange')

    plt.xlabel("Mass Ratio Q/q")
    plt.ylabel(r"$R_{L}$ ($R_{\odot}$)")
    plt.title("Roche Limit vs Mass Ratio")
    plt.legend()

    save_path = dir_main + 'qratio_RL'
    filename = save_path + '/' + 'RocheLim.png'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()