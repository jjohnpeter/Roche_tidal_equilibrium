import shutil
import os
import sys
import subprocess
from funcs_tidal_eq import *
from para_tidal_eq import *
from dir_info import *


# example run:OLD
# python tidal_equilibrium_acrit.py 0.578 2.043 2.0 (for Kentr, rhoc, sma not for Mstar sma)

# input parameters (old)
#Kentr = float(sys.argv[1])    # entropy constant [in units such that G=Msun=Rsun=1]
#rhoc = float(sys.argv[2])     # peak density [Msun/Rsun^3]
#sma = float(sys.argv[3])      # Only to record sma

# input parameters (new): for npoly runs (could also do for DWD runs)
Mstar = float(sys.argv[1])     # MS star mass
sma = float(sys.argv[2])       # Only to record sma
Qbh = float(sys.argv[3])
Lmax = float(sys.argv[4])

# Get Rstar from WD mass-rad relation
Rstar = WD_Radius(Mstar)

# Get rhoc, make sure LaneEmden file is already mine
LaneEmden_fname = 'polytrope_profile_npoly%.5f' % npoly + '.txt'
if not os.path.exists(savedir + LaneEmden_fname):
    # need to run LaneEmden.py to create the polytrope_profile
    os.system('python ' + pydir + 'LaneEmden.py')
data = np.loadtxt(savedir + LaneEmden_fname, skiprows=1)
ximax, phimax = data[-1, 0], data[-1, 2]
rhoc = (Mstar/phimax) / (4*pi*(Rstar/ximax)**3)

# Compute Kentr, given ximax, phimax, Mstar, Rstar
Kentr = (4*pi)**(1./npoly)/(npoly+1)*(Mstar/phimax)**(1-1./npoly)*(Rstar/ximax)**(3./npoly-1)

Niter = 0    # start from this iteration number (if >= 1, make sure potential%d and rho%d exist)

# Old paths / MS_K runs path
#savedir_Krhoc = savedir + 'Kentr%.5f/sma%.5f/' % (Kentr, sma)   # where data and prints are saved
#savedir_Krhoc = savedir + 'rhoc%.3f/sma%.5f/' % (rhoc, sma)
# Main DWD path / npoly
savedir_Krhoc = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f/' % (rhoc, Qbh, sma)


if not os.path.exists(savedir_Krhoc):
    os.makedirs(savedir_Krhoc, exist_ok=True)   # '-p' allows a multi-layer directory to be created
log_file = open(savedir_Krhoc + "output.txt", "w")
sys.stdout = log_file
potname_C_output = savedir_Krhoc + 'potential.txt'   # keep this the same for all iterations

rhoname = savedir_Krhoc + 'rho%d.txt' % Niter
Nx, Ny, Nz, xarr, yarr, zarr = set_grid(Lmax, Nresz)

if Niter == 0:   # start from scratch
    # initial density profile.
    #rhoname_initial = dir_main + 'npoly_1.6/polytrope_profile_npoly%.5f.txt' % npoly
    rhoname_initial = dir_main + 'DWD/polytrope_profile_npoly%.5f.txt' % npoly
    rhoarr = map_LaneEmden(rhoname_initial, Nx, Ny, Nz, xarr, yarr, zarr, npoly, rhoc, Kentr)
    #rhoarr, rhoc = map_LaneEmden_x0(rhoname_initial, Nx, Ny, Nz, xarr, yarr, zarr, npoly, x0surf, Kentr)
    print(f'sma= {sma}')
    print('max(rho)=', np.amax(rhoarr))
    write_rho(rhoname, rhoarr, Nx, Ny, Nz)
else:
    rhoarr = read_rho(rhoname, Nx, Ny, Nz)

qstar, xcom = stellar_mass(rhoarr, Nx, Ny, Nz, xarr, yarr, zarr)
print('qstar(%d)=%.5f' % (Niter, qstar))

# compile the C code
#compile_command = 'gcc -o run main.c boundary_functions.c parameters_and_boundary_types.c ../lib/libutil.a'
compile_command = 'gcc -std=gnu99 -D_USE_MATH_DEFINES -o run main.c boundary_functions.c parameters_and_boundary_types.c ../lib/libutil.a -lm'
os.chdir(srcdir)

#os.system(compile_command)

# run the C code
run_command = srcdir + 'run %d %.3f %.3f %d %.3f %.5f %.5f %.2f' % (Niter, npoly, Lmax, Nresz, rhoc, Kentr, sma, Qbh)
#run_command = srcdir + 'run %d %.3f %.3f %d %.3f %.5f' % (Niter, npoly, Lmax, Nresz, rhoc, Kentr)
print("Running command:", run_command)
#os.system(run_command)
subprocess.run(run_command.split(), check=True)

# print(run_command)
print('finished iteration %d' % Niter)
# read dimensionless potential Phi
Phiarr = read_Phi(potname_C_output, Nx, Ny, Nz)
potname = savedir_Krhoc + 'potential%d.txt' % Niter
shutil.move(potname_C_output, potname)

# update the density profile (including tidal potential)
rhoarr = update_rho_sma(Phiarr, Nx, Ny, Nz, xarr, yarr, zarr, xcom, npoly, rhoc, Kentr, Qbh, qstar, sma)
#rhoarr = update_rho_x0(Phiarr, Nx, Ny, Nz, xarr, yarr, zarr, xcom, npoly, x0surf, Kentr, Qbh, qstar)
# print('max(rho)=', np.amax(rhoarr))

qstar_old = qstar
qstar, xcom = stellar_mass(rhoarr, Nx, Ny, Nz, xarr, yarr, zarr)
frac_delta_q = (qstar - qstar_old)/qstar
print('qstar(%d)=%.5f, frac_delta_q=%.3e' % (Niter, qstar, frac_delta_q))

# ----- check convergence based on total stellar mass
while abs(qstar - qstar_old)/qstar > rtol:
    Niter += 1
    if Niter > Niter_max:
        break
    rhoname_old = rhoname
    if OnlySaveLast:  # remove rhoname_old file
        if os.path.exists(rhoname_old):
            os.remove(rhoname_old)
    rhoname = savedir_Krhoc + 'rho%d.txt' % Niter
    write_rho(rhoname, rhoarr, Nx, Ny, Nz)
    #run_command = srcdir + 'run %d %.3f %.3f %d %.3f %.5f' % (Niter, npoly, Lmax, Nresz, rhoc, Kentr)
    run_command = srcdir + 'run %d %.3f %.3f %d %.3f %.5f %.5f %.2f' % (Niter, npoly, Lmax, Nresz, rhoc, Kentr, sma, Qbh)
    #os.system(run_command)
    subprocess.run(run_command.split(), check=True)
    print('finished iteration %d' % Niter)
    Phiarr = read_Phi(potname_C_output, Nx, Ny, Nz)  # always read the freshly generated C_output
    potname_old = potname
    if OnlySaveLast:  # remove files generated in intermediate interation steps
        if os.path.exists(potname_old):
            os.remove(potname_old)
    potname = savedir_Krhoc + 'potential%d.txt' % Niter
    shutil.move(potname_C_output, potname)
    rhoarr = update_rho_sma(Phiarr, Nx, Ny, Nz, xarr, yarr, zarr, xcom, npoly, rhoc, Kentr, Qbh, qstar, sma)
    # rhoarr = update_rho(Phiarr, Nx, Ny, Nz, xarr, yarr, zarr, npoly, rhoc, Kentr, Qbh, qstar)  # old units
    # rhoarr = update_rho_x0(Phiarr, Nx, Ny, Nz, xarr, yarr, zarr, xcom, npoly, x0surf, Kentr, Qbh, qstar)  # bad!
    qstar_old, frac_delta_q_old = qstar, frac_delta_q
    qstar, xcom = stellar_mass(rhoarr, Nx, Ny, Nz, xarr, yarr, zarr)
    frac_delta_q = (qstar - qstar_old)/qstar
    print('qstar(%d)=%.5f, frac_delta_q=%.3e' % (Niter, qstar, frac_delta_q))

if Niter > Niter_max:
    print('maximum number of iterations (%d) reached, and the solution does not converge!' % Niter_max)
else:
    print('solution has converged because Mstar(%d)=Mstar(%d)!' % (Niter+1, Niter))

# exit()

# calculate the position of the L1 point
# potential closest to the x-axis (for iy = 0, iz = 0, but not exactly on x-axis)
Phixarr = [Phiarr[i, 0, 0] + Phitidal_sma(xarr[i], yarr[0], zarr[0], Qbh, qstar, sma) for i in range(Nx)]
# orbital spin change:
xL1, xL2, PhiL1, PhiL2 = findL1L2(Phixarr, Nx, xarr)

# determine the position of stellar surface along x-axis
ixc = np.searchsorted(xarr, 0.)
isurf = ixc + 1
eps_small = 1e-10   # a small number equivalent to 0
while rhoarr[isurf, 0, 0] > eps_small:
    isurf += 1
xsurf = xarr[isurf-1]  # last point of non-zero density
Phitotsurf = Phixarr[isurf-1]
print('equilibrium result: qstar=%.5f, xL1=%.5f, PhitotL1+1.5Q/a=%.8f, xsurf=%.5f, Phitotsurf+1.5Q/a=%.8f, rhoc=%.5f'
      % (qstar, xL1, PhiL1, xsurf, Phitotsurf, rhoc))

iL1 = np.searchsorted(xarr, xL1)
# note: xarr[iL1-1] < xL1 <= xarr[iL1]
if xsurf < xarr[iL1-1]:
    print('FINAL: detached')
else:
    print('FINAL: overflowing')

# remove grid file (not used)
gridfile = savedir_Krhoc + 'GridX.txt'
if os.path.exists(gridfile):
    os.remove(gridfile)

log_file.close()
