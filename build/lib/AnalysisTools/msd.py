#Compute mean-squared displacement

import numpy as np
import h5py
import sys
import numba
import argparse
import scipy

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools

def main():
    
    parser = argparse.ArgumentParser(description='Compute MSD.')
    parser.add_argument('myfile', help='Trajectory file (.h5 or .gsd.)')
    parser.add_argument('--tmax', default=100.0, help='Max. time to take MSD out to (must be smaller than time length of trajectory.)')
    parser.add_argument('--nchunks', default=5, help='Number of chunks to divide trajectory into.')
    parser.add_argument('--use_one_only', default=0, help='Select only one particle to follow for computing MSD')
    
    
    args = parser.parse_args()
    myfile = args.myfile
    tmax = float(args.tmax)
    nchunks = int(args.nchunks)
    use_one_only = int(args.use_one_only)

    ### Load data ####
    traj = particle_io.load_traj(myfile) #Extract data

    #Split trajectory into chunks
    msd_dict = {}
    msd_dict['nchunks'] = nchunks
    seglen = traj['pos'].shape[0]//nchunks
    for n in range(nchunks):
        if use_one_only==1:
            pos = traj['pos'][(n*seglen):((n+1)*seglen),0:1,:]
            image = traj['image'][(n*seglen):((n+1)*seglen),0:1,:]
            print(pos.shape)
        else:
            pos = traj['pos'][(n*seglen):((n+1)*seglen),:,:]
            image = traj['image'][(n*seglen):((n+1)*seglen),:,:]
            print(pos.shape)
        times = traj['times'][(n*seglen):((n+1)*seglen)]
        msd = get_msd(pos,image,traj['edges'],times, tmax)
        msd_dict['msd_%d' % n] = msd[:,1]
        msd_dict['times'] = msd[:,0]

    #### Output MSD to file in same directory as input h5 file ####        
    if use_one_only==1:
        numstring = 'one'
    else:
        numstring = 'all'
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/msd_tmax=%f_nchunks=%d_%s.npz' % (tmax, nchunks, numstring)
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

def fit_aoup(times, msd, dim):

    """
    Fit MSD to theoretical expression for self-propelled particle (AOUP)

    INPUT: Times (numpy array),
           MSD (numpy array),
           spatial dimensionality (int)

    OUTPUT: AOUP parameters (D, tau_p)
    """

    #discard t=0 to prevent NaN in log fit
    if times[0]==0.0:
        times=times[1:]
        msd = msd[1:]

    #Get AOUP functional form for given dimension
    aoup_msd = aoup_msd_func_log(dim)

    #Perform fit 
    fit_params, pcov = scipy.optimize.curve_fit(aoup_msd, times, np.log(msd))

    return fit_params

def aoup_msd_func(dim):

    """
    Define theoretical AOUP function for a given spatial dimension

    INPUT: dimensionality (int)
    OUTPUT: AOUP function in dim-d
    """

    def aoup_msd(t, D, tau):
        return 2*dim*D * (t + tau*(np.exp(-t/tau) - 1))

    return aoup_msd

def aoup_msd_func_log(dim):

    """
    Define theoretical AOUP function in log space for a given spatial dimension

    INPUT: dimensionality (int)
    OUTPUT: AOUP function in dim-d
    """

    def aoup_msd_log(t, D, tau):
        return np.log(2*dim*D * (t + tau*(np.exp(-t/tau) - 1)))

    return aoup_msd_log

if __name__ == '__main__':
    main()