#Compute histogram of specified quantity

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as particle_io

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_traj(myfile) #Extract data
    quantity = sys.argv[2]
    if quantity=='vel' or quantity=='velocity':
        vel = traj['vel']
        vx_hist = get_histogram(vel[:,:,0], nskip=0, nchunks=1)
        if traj['dim']==2 or traj['dim']==3:
            vy_hist = get_histogram(vel[:,:,1], nskip=0, nchunks=1)
        if traj['dim']==3:
            vz_hist = get_histogram(vel[:,:,2], nskip=0, nchunks=1)
        speed_hist = get_histogram(np.linalg.norm(vel, axis=2), nskip=0, nchunks=1)

        #### Output velocity histograms to file in same directory as input h5 file ####        
        folder = '/'.join((myfile.split('/'))[:-1])
        np.savez(folder + '/vx_hist.npz', **vx_hist)
        if traj['dim']==2 or traj['dim']==3:
            np.savez(folder + '/vy_hist.npz', **vy_hist)
        if traj['dim']==3:
            np.savez(folder + '/vz_hist.npz', **vz_hist)
        np.savez(folder + '/speed_hist.npz', **speed_hist)

    else:
        print('Quantity not yet supported.')
        exit()

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

if __name__ == '__main__':
    main()