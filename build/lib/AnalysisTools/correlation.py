#Compute correlation functions

import numpy as np
import h5py
import sys
import numba

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import particle_io
from . import measurement_tools

@numba.jit(nopython=True)
def get_single_particle_time_corr(obs, times, tmax=5.0):

    """
    Compute single-particle time correlation function of some observable over a
    trajectory.
    
    INPUT: Trajectory observable (nframes x N x (1 or d) numpy array),
    either scalar or vector
    OUTPUT: Time correlation function (1d numpy array)
    """
    
    if times.shape[0]<2:
        print('Error: not enough time to compute time correlations!')
        raise TypeError
    if len(obs.shape)!=3:
        print('Error: need 3D numpy array (nsteps, N, size of observable.)')
        raise TypeError

    dt = times[1]-times[0]
    N = obs.shape[1]
    dim = obs.shape[2]
    fmax = int(tmax/dt) #max frame
    corr = np.zeros(fmax)
    
    for t in range(fmax):
        for i in range(N):
            for mu in range(dim):
                corr[t] += np.mean(obs[:-fmax,i,mu]*obs[t:(-fmax+t),i,mu])
        corr[t] /= (N*dim)

    return corr

@numba.jit(nopython=True)
def get_single_particle_radial_corr(obs, pos, edges, rmax=10.0, nbins=30):

    """
    Compute single-particle position correlation function along r
    (assuming radial symmetry) of some observable over a trajectory.
    
    INPUT: Trajectory observable (nframes x N x (1 or d) numpy array) 
    (either scalar or vector), positions (nframes x N x d numpy array), edges (1d numpy array)
    OUTPUT: Radial position correlation function (1d numpy array)
    """
    
    if len(obs.shape)!=3:
        print('Error: need 3D numpy array (nsteps, N, size of observable.)')
        raise TypeError

    N = obs.shape[1]
    dim = obs.shape[2]
    fmax = obs.shape[0] #max frame
    corr = np.zeros(nbins)
    counts = np.zeros(nbins)
    dr = rmax/nbins
    
    #TODO: update to use neighbor list
    for t in range(fmax):
        for i in range(N-1):
            for j in range(i+1,N):
                r = measurement_tools.get_min_dist(pos[t,i,:],pos[t,j,:],edges)
                if r<rmax:
                    index = int(r/dr)
                    counts[index] += 1
                    corr[index] += np.dot(obs[t,i,:],obs[t,j,:])
    
    for i in range(nbins):
        corr[i] /= counts[i]

    return corr