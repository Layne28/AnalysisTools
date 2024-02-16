#Compute mean-squared displacement

import numpy as np
import h5py
import sys
import numba

from . import particle_io
from . import measurement_tools

@numba.jit(nopython=True)
def get_msd(pos, times, tmax=5.0):

    """
    Compute mean-squared displacement (MSD) of particles over a trajectory.
    
    INPUT: Positions (nframes x N x d numpy array),
           times (nframes numpy array),
           tmax (optional, max time to which to compute MSD)
    OUTPUT: Times and MSD vs time (2d numpy array)
    """
    
    if times.shape[0]<2:
        print('Error: not enough time to compute MSD!')
        raise TypeError
    if len(pos.shape)!=3:
        print('Error: position must be 3D numpy array (nsteps, N, d.)')
        raise TypeError

    dt = times[1]-times[0]
    N = pos.shape[1]
    dim = pos.shape[2]
    fmax = int(tmax/dt) #max frame
    if fmax>times.shape[0]:
        fmax = times.shape[0]-1
    msd = np.zeros((fmax,2))
    
    for t in range(fmax):
        msd[t,0] = dt*t
        for i in range(N):
            for mu in range(dim):
                msd[t,1] += np.mean((pos[:-fmax,i,mu] - pos[t:(-fmax+t),i,mu])**2)
        msd[t,1] /= N

    return msd
