#This script contains functions for making common measurements
#on trajectory data.

import h5py
import numpy as np
import numpy.linalg as la
import numba



###################
#Particle functions
###################

@numba.jit(nopython=True)
def get_min_disp(r1, r2, edges):

    """
    Compute displacement respecting the minimum image convention.

    INPUT: Two distance vectors (numpy arrays) and array of periodic box 
           dimensions.
    OUTPUT: Displacement vector (numpy array.)
    """

    arr1 = edges/2.0
    arr2 = -edges/2.0
    rdiff = r1-r2
    rdiff = np.where(rdiff>arr1, rdiff-edges, rdiff)
    rdiff = np.where(rdiff<arr2, rdiff+edges, rdiff)
    return rdiff

@numba.jit(nopython=True) 
def get_min_dist(r1, r2, edges):

    """
    Compute distance respecting the minimum image convention.

    INPUT: Two distance vectors (numpy arrays) and array of periodic box 
           dimensions.
    OUTPUT: Distance (float.)
    """

    rdiff = get_min_disp(r1,r2,edges)
    return la.norm(rdiff)

@numba.jit(nopython=True)
def apply_min_image(disp_r, edges):

    """
    Apply the minimum image convention to a displacement vector.
    
    INPUT: Displacement vector (numpy array) and array of periodic box
           dimensions.
    OUTPUT: Displacement vector (numpy array.)
    """

    new_disp = np.zeros((disp_r.shape))
    for i in range(new_disp.shape[0]):
        new_disp[i,:] = get_min_disp(disp_r[i,:],np.zeros(disp_r.shape[0]),edges)
    return new_disp

@numba.jit(nopython=True)
def apply_pbc(pos, edges):

    """
    Apply the pbc to all particle positions.
    
    INPUT: Vector of positions (numpy array) and array of periodic box
           dimensions.
    OUTPUT: Vector of positions (numpy array.)
    """

    arr1 = edges/2.0
    arr2 = -edges/2.0

    new_pos = np.zeros((pos.shape))
    for i in range(pos.shape[0]):
        new_pos[i,:] = np.where(pos[i,:]>=arr1, pos[i,:]-edges, pos[i,:])
        new_pos[i,:] = np.where(new_pos[i,:]<arr2, new_pos[i,:]+edges, new_pos[i,:])
    return new_pos

#This function takes in a position trajectory and an array of bonds (fixed through trajectory)
@numba.jit(nopython=True)
def get_strain_bonds(pos, bonds, edges, leq):

    """
    Compute the strain in each bond in a network.
    
    INPUT: Positions of all particles during trajectory, array of bonds
           (containing indices of 2 participating atoms), array of periodic box
           dimensions, and equilibrium bond length.
    OUTPUT: Strain of each bond (numpy array.)
    """

    #Check whether connectivity changes
    if len(bonds.shape)==2: #Fixed connectivity

        nframes = pos.shape[0]
        N = pos.shape[1]
        nbonds = bonds.shape[0]
        strain_arr = np.zeros((nframes, nbonds))

        for t in range(nframes):
            for i in range(nbonds):
                b = bonds[i,:]
                min_disp_vec = get_min_disp(pos[t,b[0],:], pos[t,b[1],:], edges)
                pos1, pos2 = pos[t,b[0],:], pos[t,b[0],:]-min_disp_vec
                if la.norm(pos1-pos2)>2:
                    print('Error!!')
                strain = la.norm(min_disp_vec)-leq
                strain_arr[t][i] = strain
        return strain_arr

    elif len(bonds.shape)==3: #Connectivity changes from frame to frame
        
        nframes = pos.shape[0]
        N = pos.shape[1]
        nbonds = bonds.shape[1]
        strain_arr = np.zeros((nframes, nbonds))

        for t in range(nframes):
            for i in range(nbonds):
                b = bonds[t,i,:]
                min_disp_vec = get_min_disp(pos[t,b[0],:], pos[t,b[1],:], edges)
                pos1, pos2 = pos[t,b[0],:], pos[t,b[0],:]-min_disp_vec
                if la.norm(pos1-pos2)>2:
                    print('Error!!')
                strain = la.norm(min_disp_vec)-leq
                if la.norm(pos1-pos2)<1e-8: # (0,0 entries are given a "flag" to ignore)
                    strain = 1e4
                strain_arr[t][i] = strain
        return strain_arr

    else:
        raise TypeError
    
###################
#Noise functions
###################
    
def get_rms_noise(noise):

    """
    Compute root mean square fluctuations in a noise trajectory.
    
    INPUT: Noise trajectory ((d+2)-dimensional numpy array, with
           first dimension time (nframes),
           next d dimensions {n_mu, mu=1,2,...,d},
           and last dimension
           d (components of field))
    OUTPUT: RMS noise fluctuations for a single trajectory (scalar)
    """

    if len(noise.shape)==2:
        dim = 1
    else:
        dim = noise.shape[-1]
    rms = np.sqrt(np.mean(noise**2)*dim)

    return rms

