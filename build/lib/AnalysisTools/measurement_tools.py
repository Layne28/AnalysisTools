#This script contains functions for making common measurements
#on trajectory data.

import h5py
import numpy as np
import numba

@numba.jit(nopython=True)
def get_min_disp(r1, r2, edges):

    """
    Compute displacement respecting the minimum image convention.

    INPUT: Two distance vectors (numpy arrays) and array of periodic box 
           dimensions.
    OUTPUT: Displacement vector (numpy array)
    """

    arr1 = edges/2.0
    arr2 = -edges/2.0
    rdiff = r1-r2
    rdiff = np.where(rdiff>arr1, rdiff-edges, rdiff)
    rdiff = np.where(rdiff<arr2, rdiff+edges, rdiff)
    return rdiff

def get_histogram(data, llim=np.nan, ulim=np.nan, nbins=50, nchunks=10):

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

    the_dict['avg_hist'] = get_hist_avg(the_dict)
    the_dict['stddev_hist'] = get_hist_stddev(the_dict)

    return the_dict

    
def get_hist_avg(data, nskip=0):

    """
    Compute histogram average over chunks

    INPUT: Dictionary containing chunked histograms.
    OUTPUT: Average histogram (numpy array)
    """

    nchunks = data['nchunks']

    avg = np.zeros(data['hist_0'].shape[0])

    for n in range(nskip, nchunks): #Skip first nskip chunks
        avg += data['hist_%d' % n]

    avg /= (nchunks-nskip)
    return avg

def get_hist_stddev(data, nskip=0):

    """
    Compute histogram standard deviation over chunks

    INPUT: Dictionary containing chunked histograms.
    OUTPUT: Standard deviation of chunk histograms (numpy array)
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