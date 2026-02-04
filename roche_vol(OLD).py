import sys
from funcs_tidal_eq import *
from para_tidal_eq import *
from dir_info import *
import numpy as np
from scipy.ndimage import label

# this script extracts the Roche Lobe volume of an existing run, then converts to effective radius RL
# note that Qbh was specified in para_tidal_eq.py but new iteration of this file has QBh as argument
# so, remember to comment out Qbh in para_tidal_eq.py before running this file

# Example: "python roche_vol.py 27927.401 0.08717 6 0.90"

# input parameters
#Kentr = float(sys.argv[1])    #  # entropy constant [in units such that G=Msun=Rsun=1]
rhoc = float(sys.argv[1])     # peak density [Msun/Rsun^3]
sma = float(sys.argv[2])      ## For file name
Niter = float(sys.argv[3])    # which iteration profile
Qbh = float(sys.argv[4])      # companion (BH) mass [Msun]

# We extract the Lmax value used in the run, recorded within the output.txt file
# Overwrites Lmax from dir_info.py
savedir_Krhoc = savedir + 'rhoc%.3f_Qbh%.2f/sma%.5f/' % (rhoc, Qbh, sma)
fname = savedir_Krhoc + 'output.txt'

labels = ['qstar', 'xL1', 'PhitotL1+1.5Q/a', 'xsurf', 'Phitotsurf+1.5Q/a', 'rhoc']

with open(fname, 'r') as f:
    output_file = f.read()
    f.seek(0)
    row_new = f.readline()
    while len(row_new) > 0:
        if 'Running command: /Users/JovanJohnPeter/Roche_tidal_equilibrium/poisson3D/src/run ' in row_new:
            c_args = row_new
        if 'equilibrium result: ' in row_new:
            eq_res = row_new
        row_new = f.readline()

    eq_res = eq_res.replace('equilibrium result: ', '')
    for lab in labels:
        eq_res = eq_res.replace(lab + '=', '').replace('', '').strip()
    values = [float(item) for item in eq_res.split(',')]  # 5 properties of eq sol
    qstar = values[0]
    xL1 = values[1]
    phiL1 = values[2]
    c_args = c_args.replace('Running command: /Users/JovanJohnPeter/Roche_tidal_equilibrium/poisson3D/src/run ', '').split()
    c_args_arr = np.array(c_args, dtype=float)
    Lmax = c_args_arr[2]
    #print(Lmax)

# Reading the rho file
rhoname = savedir_Krhoc + 'rho%d.txt' % Niter
Nx, Ny, Nz, xarr, yarr, zarr = set_grid(Lmax, Nresz)
rhoarr = read_rho(rhoname, Nx, Ny, Nz)
#print(rhoarr)

# Reading the pot file
potname = savedir_Krhoc + 'potential%d.txt' % Niter
Phiarr = read_Phi(potname, Nx, Ny, Nz)
qstar, xcom = stellar_mass(rhoarr, Nx, Ny, Nz, xarr, yarr, zarr)
rhoc = np.amax(rhoarr)

# get rid of all the zeros in density profile
eps_small = 1e-10
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            rhoarr[i, j, k] = max(eps_small, rhoarr[i, j, k])


# add tidal potential
Phitotarr = np.zeros_like(Phiarr)
for i in range(Nx):
    x = xarr[i]
    for j in range(Ny):
        y = yarr[j]
        for k in range(Nz):
            z = zarr[k]
            Phitotarr[i, j, k] = Phiarr[i, j, k] + Phitidal_sma(x, y, z, Qbh, qstar, sma)

#print(np.min(Phitotarr), phiL1, np.max(Phitotarr))
# Potential decreases as you go inside the star system (relative to the lobe's surface equipotential)
# Condition that the potential within the star is less than the potential at the lagrange point
# When using scipy.ndimage.label, we treat the inside of the primary roche lobe, the companion roche
# lobe, and outside both as their own feature (0, 1, 2)

# Method 1: "Label"

# Since we use the roche lobe contour (phiL1) as the condition, need to make sure not to count the
# part of the lobe that surrounds the companion.
# Ridding of all potentials to the right of the L1 point
Phixarr = Phitotarr[:, 0, 0]
xL1, xL2, PhiL1, PhiL2 = findL1L2(Phixarr, Nx, xarr)
# Find index in xarr that is right after the L1 point
iL1_cut = np.searchsorted(xarr, xL1)
# Cut array past this value
Phitotarr = Phitotarr[:iL1_cut+1,:, :]


pot_cond = (Phitotarr <= phiL1)
Phi_length = len(Phitotarr[pot_cond])
#print(f'length of valid Phi: {Phi_length}') #test
labeled_pot, num_pot_features = label(pot_cond)
#print(num_pot_regions)

# Identify the "feature number" (label) associated with inside the stellar roche lobe
ixc, iyc, izc = np.searchsorted(xarr, 0.), np.searchsorted(yarr, 0), np.searchsorted(zarr, 0.)
#print(ixc, iyc, izc)
center_label = labeled_pot[ixc, iyc, izc]
phi_prim_roche = (labeled_pot == center_label) # Labels the grid cells inside primary lobe as True

dx = xarr[1] - xarr[0]
dy = yarr[1] - yarr[0]
dz = zarr[1] - zarr[0]
cell_vol = dx*dy*dz

# # test
# V_lobe_test = Phi_length * cell_vol
# R_lobe_test =  ((3*V_lobe_test)/(4*np.pi))**(1/3)
# print(f'Lobe Radius test: {R_lobe_test}')

V_lobe = phi_prim_roche.sum() * cell_vol # Number of "True" cells times volume of 1 cell
V_lobe = V_lobe * 2 * 2 # Each factor of 2 accounts for grid symmetry along y,z
#print(f'Length of valid phi_label: {len(phi_prim_roche)}')
R_Lobe = ((3*V_lobe)/(4*np.pi))**(1/3)
#print(f'Lobe Radius via label: {R_Lobe}')
print(R_Lobe)


# star_mask = (rhoarr > 0)
# V_star = star_mask.sum() * cell_vol
# V_star = V_star * 2 * 2 # Each factor of 2 accounts for grid symmetry along y,z
# R_star_true = ((3*V_star)/(4*np.pi))**(1/3)

#print(R_Lobe)



# # Method 2: Meshes
# from skimage.measure import marching_cubes
# from trimesh import Trimesh
# verts, faces, normals, values = marching_cubes(Phitotarr.transpose(), level=phiL1, spacing=(dx,dy,dz))
# # Computing volume of mesh
# mesh = Trimesh(vertices=verts, faces=faces)
# V_lobe = mesh.volume
# R_Lobe = ((3*V_lobe)/(4*np.pi))**(1/3)
# print(R_Lobe)
