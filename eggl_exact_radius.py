import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from math import pi
from funcs_tidal_eq import *
from para_tidal_eq import *
from dir_info import *
from scipy.ndimage import label

# For the point source case (Eggleton), this script calculates the volume-equivalent radius of the Lobe
# We start by creating our grid. Then, set our converged mass as the point source mass of the primary.
# Use the roche potential (without the stellar gas potential term) to create the roche lobe.

# input parameters
rhoc = float(sys.argv[1])     # peak density [Msun/Rsun^3]
sma = float(sys.argv[2])      ## For file name
Niter = float(sys.argv[3])    # which iteration profile (only need this to retrieve our file values)
Qbh = float(sys.argv[4])      # companion (BH) mass [Msun]

# Constructing our point source configuration (recreating what Eggleton did)

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


# ----Extracting the radii!!
method = 2 # 1 for method 1, 2 for method 2

# ---METHOD 1: sum each grid point (3D). Create a condition to count the number of terms in the potential
# array that satisfy that potential < lobe_potential. Multiply that number by the volume of each grid
# "cube" which gives you the total volume which you can  stuff into a sphere to get the desired radius
# This method most likely has a slightly higher error than method 2 b/c it will overcount the volume
# along the surface of the star (at the surface, the corresponding grid "cubes" aren't fully filled,
# yet we calculate as so. However, hopefully this should be accurate to ~ a 1-2 tenths of a percent

if method == 1:
    # Create grid
    Nresz_pt_source = 400
    Nx, Ny, Nz, xarr, yarr, zarr = set_grid(Lmax, Nresz_pt_source)
    dx = xarr[1] - xarr[0]  # grid res (used for bisection in method 2)

    # Build classical Roche potential
    Phi_classic = np.zeros((Nx, Ny, Nz), dtype=float)
    for i in range(Nx):
        x = xarr[i]
        for j in range(Ny):
            y = yarr[j]
            for k in range(Nz):
                z = zarr[k]
                # Calculate self gravity of point mass (primary)
                Phi_self = -qstar / np.sqrt(x ** 2 + y ** 2 + z ** 2)
                Phi_tidal = Phitidal_sma(x, y, z, Qbh, qstar, sma)
                Phi_classic[i, j, k] = Phi_self + Phi_tidal

    # calculate the position of the L1 point
    # potential closest to the x-axis (for iy = 0, iz = 0, but not exactly on x-axis)
    Phixarr_classic = [Phi_classic[i, 0, 0] for i in range(Nx)]
    xL1_classic, xL2_classic, PhiL1_classic, PhiL2_classic = findL1L2(Phixarr_classic, Nx, xarr)

    # Find index in xarr that is right after the L1 point
    #### CHECK THIS: DONT KNOW IF IT SHOW BE xL1_classic+1 or no +1 or do the +1 in the Phi_classic_count
    iL1_cut = np.searchsorted(xarr, xL1_classic)
    # Cut array past this value
    Phi_classic_count = Phi_classic[:iL1_cut,:, :]

    pot_cond = (Phi_classic_count <= PhiL1_classic)
    Phi_length = len(Phi_classic_count[pot_cond])
    labeled_pot, num_pot_features = label(pot_cond)

    # Identify the "feature number" (label) associated with inside the roche lobe equipotential
    ixc, iyc, izc = np.searchsorted(xarr, 0.), np.searchsorted(yarr, 0), np.searchsorted(zarr, 0.)
    #print(ixc, iyc, izc)
    center_label = labeled_pot[ixc, iyc, izc]
    phi_prim_roche = (labeled_pot == center_label) # Labels the grid cells inside primary lobe as True

    dx = xarr[1] - xarr[0]
    dy = yarr[1] - yarr[0]
    dz = zarr[1] - zarr[0]
    cell_vol = dx*dy*dz

    # comment out the 4 lines below if method 2 is to be used instead of this method
    V_lobe_grid = phi_prim_roche.sum() * cell_vol # Number of "True" cells times volume of 1 cell
    V_lobe_grid = V_lobe_grid * 2 * 2 # Each factor of 2 accounts for grid symmetry along y,z
    R_Lobe_grid = ((3*V_lobe_grid)/(4*np.pi))**(1/3)
    print(R_Lobe_grid)


# ---Method 2: shoot rays/cones to make the 3D problem a 2D one. Imagine you're standing in the center of
# the star and shooting rays that eventually reach the surface of the lobe. You create cones with these
# "rays" and get the volume of each ray and sum up. This method is much more precise but more complex.
# Always best to use different methods to confirm accuracy, so we use both. To increase accuracy of
# this method, we interpolate between grid points (the more interpolations, the more cones we can
# use, hence a more accurate volume(imagine that at infinitely many infinite cones are just straight
# lines from the center to the lobe surface that cover every single atom of the star)).

else:

    mu = Qbh / (Qbh + qstar)

    # convert from spherical coordinates to Cartesian: Φ(r, θ, ϕ) to  Φ(x,y,z)
    def get_potential(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            return -1e30 # infinite potential at center

        # Calculate exact potential again (in terms of r)
        phi_self_r = -qstar / r
        phi_tidal_r = Phitidal_sma(x, abs(y), abs(z), Qbh, qstar, sma)
        phi_tot_r = phi_self_r + phi_tidal_r
        return phi_tot_r

    # Find L1 point (saddle pt: dPhi/dx = 0)
    def dphi_dx(x):
        h = 1e-5 * sma
        phiL1_rays = (get_potential(x+h,0,0) - get_potential(x-h,0,0)) / (2*h)
        return phiL1_rays

    # bisection to get xL1, use some small number as tolerance
    bis_tol = 1e-7
    xL1_rays = bisec(dphi_dx, 0.01*sma, 0.99*sma, 1e-7*sma) # has to be between 1% and 99% of sma
    PhiL1_rays = get_potential(xL1_rays, 0, 0)

    # Integrate volume
    # Create angular grid
    Ntheta = 300
    Nphi = 600
    dtheta = pi/Ntheta
    dphi = 2*pi/Nphi

    Vlobe = 0
    # Loop through all angles and add to the total volume
    for i in range(Ntheta):
        theta = (i + 0.5) * dtheta

        for j in range(Nphi):
            phi = (j + 0.5) * dphi

            # unit vect (direction of the particular ray)
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)

            # function to get when phi(r) = phi at L1
            def root_funct(r):
                root = get_potential(r*nx, r*ny, r*nz) - PhiL1_rays
                return(root)

            # Set a reasonable range to do bisection from
            rmin = 1e-6
            rmax = xL1_rays

            # Bracket for the root
            fmin = root_funct(rmin)
            fmax = root_funct(rmax)

            # fmax < 0 means the root is essentially at rmax (x position of L1 point)
            if fmax < 0:
                #print(f'Couldnt bracket root for theta: {theta:.3f}, phi: {phi:.3f}')
                r_surf = rmax
            else:
                r_surf = bisec(root_funct, rmin, rmax, 1e-7)

            # Use bisection get at what radius r the potential is equal to L1 potential
            # arbitrary tolerance using some low number, we use 1e-5 here
            #r_surf = bisec(root_funct, 1e-6, xL1_rays * 1.5, 1e-7)
            #r_surf = bisec(root_funct, rmin, rmax, 1e-7)

            # Add cone volume
            Vlobe = Vlobe + ((1/3) * (r_surf**3)* np.sin(theta) * dtheta * dphi)

    # Finally, calculate effective radius
    R_lobe_rays = ((3 * Vlobe) / (4 * np.pi)) ** (1 / 3)
    print(R_lobe_rays)

        # Bisection arbitray
        # fmin = get_root(rmin, th, ph)
        # fmax = get_root(rmax, th, ph)
        #
        # # Rid of potentials that are below the L1 potential (constrained by our box bounds)
        # if fmax < 0:
        #     return 0
        #
        # # Bisection: We want to find an rmin, rmax where the roots change sign; if you keep bisecting
        # # you eventually converge to the root, which is at 0
        # for i in range(40):
        #     rmid = 0.5 * (rmin + rmax)
        #     fmid = get_root(rmid, th, ph)
        #     if fmid > 0: # this is outside lobe
        #         rmax = rmid
        #     else:
        #         rmin = rmid
        # return 0.5 * (rmin + rmax)


