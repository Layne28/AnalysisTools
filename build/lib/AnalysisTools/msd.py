#Compute mean-squared displacement

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_traj(myfile) #Extract data
    nchunks = int(sys.argv[2])
    tmax = 100.0
    if len(sys.argv)>3:
        tmax = float(sys.argv[3])

    #Split trajectory into chunks
    msd_dict = {}
    msd_dict['nchunks'] = nchunks
    seglen = traj['pos'].shape[0]//nchunks
    for n in range(nchunks):
        pos = traj['pos'][(n*seglen):((n+1)*seglen),:,:]
        image = traj['image'][(n*seglen):((n+1)*seglen),:,:]
        times = traj['times'][(n*seglen):((n+1)*seglen)]
        msd = get_msd(pos,image,traj['edges'],times, tmax)
        msd_dict['msd_%d' % n] = msd[:,1]
        msd_dict['times'] = msd[:,0]

    #### Output MSD to file in same directory as input h5 file ####        
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/msd.npz'
    np.savez(outfile, **msd_dict)

@numba.jit(nopython=True)
def get_msd(pos, image, edges, times, tmax=5.0):

    """
    Compute mean-squared displacement (MSD) of particles over a trajectory.
    
    INPUT: Positions (nframes x N x d numpy array),
           Periodic image index array (nframes x N x d numpy array),
           box edges (d numpy array)
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

    #unwrap trajectory
    upos = unwrap(pos, image, edges, dim)
    
    for t in range(fmax):
        msd[t,0] = dt*t
        for i in range(N):
            for mu in range(dim):
                msd[t,1] += np.mean((upos[:-fmax,i,mu] - upos[t:(-fmax+t),i,mu])**2)
        msd[t,1] /= N

    return msd

@numba.jit(nopython=True)
def unwrap(pos, image, edges, dim):

    """
    Unwrap a trajectory with periodic boundary conditions.
    
    INPUT: Positions (nframes x N x d numpy array),
           Periodic image index array (nframes x N x d numpy array),
           box edges (d numpy array),
           dimensions (scalar)
    OUTPUT: Unwrapped positions (nframes x N x d numpy array)
    """

    upos = np.zeros(pos.shape)
    for t in range(pos.shape[0]):
        for d in range(dim):
            upos[t][0][d] = pos[t][0][d] + image[t][0][d]*edges[d]

    return upos

if __name__ == '__main__':
    main()