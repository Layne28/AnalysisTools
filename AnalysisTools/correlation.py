#Compute correlation functions

import numpy as np
import h5py
import sys
import numba

import matplotlib as mpl
import matplotlib.pyplot as plt

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools
import AnalysisTools.cell_list as cl

@numba.jit(nopython=True)
def get_single_particle_time_corr(obs, times, tmax=5.0):

    """
    Compute single-particle time correlation function of some observable over a
    trajectory.
    
    INPUT: Trajectory observable (nframes x N x (1 or d) numpy array),
    either scalar or vector
    OUTPUT: Times and time correlation function (2d numpy array)
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
    if fmax>times.shape[0]:
        fmax = times.shape[0]-1
    corr = np.zeros((fmax,2))
    
    for t in range(fmax):
        corr[t,0] = dt*t
        for i in range(N):
            for mu in range(dim):
                corr[t,1] += np.mean(obs[:-fmax,i,mu]*obs[t:(-fmax+t),i,mu])
        corr[t,1] /= (N*dim)

    return corr

def get_single_particle_time_corr_chunked(obs, times, tmax=5.0, nchunks=10, nskip=0):

    """
    Compute single-particle time correlation function of some observable over a
    trajectory. Get average and std. error by chunking trajectory.
    
    INPUT: Trajectory observable (nframes x N x (1 or d) numpy array),
    either scalar or vector
    OUTPUT: Dictionary containing time correlation functions (1d numpy array)
    and times (1d numpy array) for each chunk, plus average and std. error.
    """
    
    if times.shape[0]<2:
        print('Error: not enough time to compute time correlations!')
        raise TypeError
    if len(obs.shape)!=3:
        print('Error: need 3D numpy array (nsteps, N, size of observable.)')
        raise TypeError

    values = []
    the_dict = {}
    the_dict['nchunks'] = nchunks
    seglen = obs.shape[0]//nchunks
    print('seglen: ',seglen)
    segtimes = times[:seglen]

    for n in range(nchunks):
        corr = get_single_particle_time_corr(obs[(n*seglen):((n+1)*seglen),:,:],segtimes, tmax)

        the_dict['times_%d' % n] = corr[:,0]
        the_dict['corr_%d' % n] = corr[:,1]

    the_dict['times'] = corr[:,0]
    the_dict['avg_corr'] = get_corr_avg(the_dict, nskip=nskip)
    the_dict['stddev_corr'] = get_corr_stddev(the_dict, nskip=nskip)
    the_dict['nskipped'] = nskip

    return the_dict

@numba.jit(nopython=True)
def get_single_particle_radial_corr(obs, pos, edges, dim, rmax=3.0, nbins=50, use_cell_list=1):

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
    fmax = obs.shape[0] #max frame
    corr = np.zeros((nbins,2))
    counts = np.zeros(nbins)
    dr = rmax/nbins
    print(dr)
    for i in range(nbins):
        corr[i,0] = i*dr
    
    if use_cell_list==1:
        print('Using cell list')
        ncell_arr, cellsize_arr, cell_neigh = cl.init_cell_list(edges, rmax, dim)
        for t in range(fmax):
            if t%10==0:
                print(t)
            #Create cell list for locating pairs of particles
            head, cell_list, cell_index = cl.create_cell_list(pos[t,:,:], edges, ncell_arr, cellsize_arr, dim)
            for i in range(N):
                pos1 = pos[t,i,:]
                icell = int(cell_index[i])
                for nc in range(cell_neigh[icell][0]):
                    jcell = int(cell_neigh[icell][nc+1])
                    j = int(head[jcell])
                    while j != -1:
                        pos2 = pos[t,j,:]
                        rij = measurement_tools.get_min_dist(pos1,pos2,edges)
                        if rij<rmax:
                            index = int(rij/dr)
                            counts[index] += 1
                            corr[index,1] += np.dot(obs[t,i,:],obs[t,j,:])
                        j = int(cell_list[j])

    else:
        for t in range(fmax):
            if t%10==0:
                print(t)
            for i in range(N-1):
                for j in range(i+1,N):
                    rij = measurement_tools.get_min_dist(pos[t,i,:],pos[t,j,:],edges)
                    if rij<rmax:
                        index = int(rij/dr)
                        counts[index] += 1
                        corr[index,1] += np.dot(obs[t,i,:],obs[t,j,:])
        
    for i in range(nbins):
        if counts[i]>0:
            corr[i,1] /= counts[i]

    return corr

def get_single_particle_radial_corr_chunked(obs, pos, edges, dim, rmax=3.0, nbins=50, use_cell_list=1, nchunks=10, nskip=0):

    """
    Compute single-particle radial correlation function of some observable over a
    trajectory. Get average and std. error by chunking trajectory.
    
    INPUT: Trajectory observable (nframes x N x (1 or d) numpy array),
    either scalar or vector
    OUTPUT: Dictionary containing time correlation functions (1d numpy array)
    and times (1d numpy array) for each chunk, plus average and std. error.
    """
    
    if len(obs.shape)!=3:
        print('Error: need 3D numpy array (nsteps, N, size of observable.)')
        raise TypeError

    values = []
    the_dict = {}
    the_dict['nchunks'] = nchunks
    seglen = obs.shape[0]//nchunks
    print('seglen: ',seglen)

    for n in range(nchunks):
        corr = get_single_particle_radial_corr(obs[(n*seglen):((n+1)*seglen),:,:],pos,edges,dim,rmax=rmax, nbins=nbins, use_cell_list=use_cell_list)

        the_dict['distances_%d' % n] = corr[:,0]
        the_dict['corr_%d' % n] = corr[:,1]

    the_dict['distances'] = corr[:,0]
    the_dict['avg_corr'] = get_corr_avg(the_dict, nskip=nskip)
    the_dict['stddev_corr'] = get_corr_stddev(the_dict, nskip=nskip)
    the_dict['nskipped'] = nskip

    return the_dict

def get_corr_avg(data, nskip=0):

    """
    Compute correlation function average over chunks.

    INPUT: Dictionary containing chunked correlation functions.
    OUTPUT: Average correlation function (numpy array.)
    """

    nchunks = data['nchunks']

    avg = np.zeros(data['corr_0'].shape[0])

    for n in range(nskip, nchunks): #Skip first nskip chunks
        avg += data['corr_%d' % n]

    avg /= (nchunks-nskip)
    return avg

def get_corr_stddev(data, nskip=0):

    """
    Compute correlation function standard deviation over chunks.

    INPUT: Dictionary containing chunked correlation functions.
    OUTPUT: Standard deviation of chunk correlation functions (numpy array.)
    """

    if 'avg_corr' in data.keys():
        avg = data['avg_corr']
    else:
        avg = get_corr_avg(data, nskip=nskip)

    nchunks = data['nchunks']

    stddev = np.zeros(data['corr_0'].shape[0])

    for n in range(nskip, nchunks): #Skip first nskip chunks
        stddev += (data['corr_%d' % n]-avg)**2

    stddev /= (nchunks-nskip)
    stddev = np.sqrt(stddev)

    return stddev
