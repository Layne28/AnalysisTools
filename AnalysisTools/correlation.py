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