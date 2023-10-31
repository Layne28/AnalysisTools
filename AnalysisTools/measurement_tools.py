#This script contains functions for making common measurements
#on trajectory data.

import h5py
import numpy as np
import numpy.linalg as la
import numba

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

def get_histogram(data, nskip=0, llim=np.nan, ulim=np.nan, nbins=50, nchunks=10):

    """
    Compute histogram of a scalar observable taken over a trajectory.

    INPUT: Time-ordered numpy array of observable values along a trajectory,
           parameters specifying bins, and number of chunks to divide
           into (for computing error bars.)
    OUTPUT: Dictionary containing histograms for each chunk, as well as average
            histogram and error bars.
    """

    if np.isnan(llim):
        llim = np.min(data)
    if np.isnan(ulim):
        ulim = np.max(data)
    mybins = np.linspace(llim,ulim,num=nbins)

    values = []
    the_dict = {}
    the_dict['nchunks'] = nchunks
    seglen = data.shape[0]//nchunks

    if (len(data.shape)==1): #one value per configuration
        for n in range(nchunks):
            values.append(data[(n*seglen):((n+1)*seglen)])
    elif (len(data.shape)==2): #one value per particle
        for n in range(nchunks):
            values.append(data[(n*seglen):((n+1)*seglen),:])
    else:
        print('Error: array has more than two dimensions.')
        raise(TypeError)
    
    for n in range(nchunks):

        hist, bin_edges = np.histogram(values[n], bins=mybins, density=True)
        bins = (bin_edges[:-1]+bin_edges[1:])/2

        the_dict['bins_%d' % n] = bins
        the_dict['hist_%d' % n] = hist

    the_dict['avg_hist'] = get_hist_avg(the_dict, nskip=nskip)
    the_dict['bins'] = bins
    the_dict['stddev_hist'] = get_hist_stddev(the_dict, nskip=nskip)
    the_dict['nskipped'] = nskip

    return the_dict

    
def get_hist_avg(data, nskip=0):

    """
    Compute histogram average over chunks.

    INPUT: Dictionary containing chunked histograms.
    OUTPUT: Average histogram (numpy array.)
    """

    nchunks = data['nchunks']

    avg = np.zeros(data['hist_0'].shape[0])

    for n in range(nskip, nchunks): #Skip first nskip chunks
        avg += data['hist_%d' % n]

    avg /= (nchunks-nskip)
    return avg

def get_hist_stddev(data, nskip=0):

    """
    Compute histogram standard deviation over chunks.

    INPUT: Dictionary containing chunked histograms.
    OUTPUT: Standard deviation of chunk histograms (numpy array.)
    """

    if 'avg_hist' in data.keys():
        avg = data['avg_hist']
    else:
        avg = get_hist_avg(data, nskip=nskip)

    nchunks = data['nchunks']

    stddev = np.zeros(data['hist_0'].shape[0])

    for n in range(nskip, nchunks): #Skip first nskip chunks
        stddev += (data['hist_%d' % n]-avg)**2

    stddev /= (nchunks-nskip)
    stddev = np.sqrt(stddev)

    return stddev

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
    if (bonds.shape).size==2: #Fixed connectivity

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

    elif (bonds.shape).size==3: #Connectivity changes from frame to frame
        
        nframes = pos.shape[0]
        N = pos.shape[1]

        for t in range(nframes):
            nbonds = bonds.shape[1]
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