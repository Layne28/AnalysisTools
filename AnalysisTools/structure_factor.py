#Compute and output the static structure factor given an input trajectory
#The static structure factor is defined as:
#S(q) = (1/N) * <\sum_{j,k} [exp(i * q * (rj - rk))]>
#     = (1/N) * <|\sum_j exp(i * q * rj)|^2>
#where rj is the position of particle j
#and each sum is over all N particles.

#Input: Trajectory in h5md format (.h5)
#Output: S(q) vs allowed wavevectors q in npz format

import numpy as np
import h5py
import sys
import numba

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools
import AnalysisTools.trajectory_stats as stats

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_traj(myfile) #Extract data
    nchunks = int(sys.argv[2])
    #eq_frac = float(sys.argv[2]) #cut off first eq_frac*100% of data (equilibration)
 
    #Compute S(q)
    print('Computing S(q)...')
    sq = get_sq(traj, nchunks=nchunks, qmax=np.pi)
    print('Computed S(q).')

    #### Output S(q) to file in same directory as input h5 file ####        
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/sq.npz'
    print(outfile)
    np.savez(outfile, **sq)
    
    # Compute S(q)(t)
    print('Computing S(q) trajectory...')
    sq_traj = get_sq_traj(traj, qmax=1.0)
    print('Computed S(q) traj.')
    #### Output S(q) traj to file in same directory as input h5 file #### 
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/sq_traj.npz'
    np.savez(outfile, **sq_traj)

#### Methods ####

def rebin_sq(base_folder, nbins=-1, max_num_traj=100):

    """Rebin S(q) to produce 1d with specified nbins"""

    if nbins==-1:
        #search base_folder path for system size
        subdirs = base_folder.split('/')
        substring = [item for item in subdirs if item.startswith('Lx=')][0]
        substring = substring.split('_')[0]
        L = float(substring.split('=')[-1])
        nbins = int(2.0/(2*np.pi/L))
        print(nbins)
    
    print('rebinning...')
    data = stats.get_trajectory_data(base_folder, 'sq.npz', dataset='sq', subfolder='prod', max_num_traj=max_num_traj)
    #print(data)
    for d in data:
        for n in range(d['nchunks']):
            sq = d['sq_vals_%d' % n]
            q = d['qvals']
            counts = np.zeros(nbins)
            sq_new = np.zeros(nbins)
            qnorm = np.linalg.norm(q, axis=1)
            q1d = np.linspace(0,np.max(qnorm),num=nbins+1)
            for j in range(qnorm.shape[0]):
                index = int(np.floor(qnorm[j]/np.max(qnorm)*nbins))
                if index==nbins:
                    index = nbins-1
                counts[index] += 1.0
                sq_new[index] += sq[j]
            sq_new = np.divide(sq_new,counts)
            q1d = q1d[1:-1]
            sq_new = sq_new[1:]
            q1d = (q1d)[~np.isnan(sq_new)]
            sq_new = sq_new[~np.isnan(sq_new)]
            d['sq_vals_1d_%d' % n] = sq_new
            d['qvals_1d'] = q1d

    return data

    #mystats = stats.get_postprocessed_stats(data)
    #np.savez(base_folder + '/sq_avg.npz', **mystats)

@numba.jit(nopython=True)
def get_allowed_q(qmax, dq, dim):

    """Generate a grid of wavevectors (w/ lattice constant dq),
    then select those within a sphere of radius qmax."""

    qvals = np.arange(0, qmax+dq, dq)

    if dim==3:
        qlist = np.zeros((qvals.shape[0]**3,3))
        cnt = 0
        for kx in range(qvals.shape[0]):
            for ky in range(qvals.shape[0]):
                for kz in range(qvals.shape[0]):
                    qvec = np.array([qvals[kx], qvals[ky], qvals[kz]])
                    if np.linalg.norm(qvec)<=qmax:
                        qlist[cnt,:] = qvec
                        cnt += 1
    elif dim==2:
        qlist = np.zeros((qvals.shape[0]**2,2))
        cnt = 0
        for kx in range(qvals.shape[0]):
            for ky in range(qvals.shape[0]):
                qvec = np.array([qvals[kx], qvals[ky]])
                if np.linalg.norm(qvec)<=qmax:
                    qlist[cnt,:] = qvec
                    cnt += 1
    elif dim==1:
        qlist = np.zeros((qvals.shape[0],1))
        cnt = 0
        for kx in range(qvals.shape[0]):
            qvec = np.array([qvals[kx]])
            if np.linalg.norm(qvec)<=qmax:
                qlist[cnt,:] = qvec
                cnt += 1

    else:
        print('Error: dim must be 1, 2, or 3.')
        raise ValueError

    return qlist[:cnt,:]

def get_sqt(traj, nchunks=5, spacing=0.0, qmax=2*np.pi, tmax=100.0):

    """
    Compute dynamic structure factor.

    INPUT: Particle trajectory (dictionary),
           number of chunks to divide trajectory into,
           spacing in q space,
           max q value,
           max time value
    OUTPUT: Dictionary containing S(q,t) for each chunk
    """

    the_dict = {}
    the_dict['nchunks'] = nchunks
    seglen = traj['pos'].shape[0]//nchunks

    #### Chunk positions ####
    pos_chunks = []
    for n in range(nchunks):
        pos_chunks.append(traj['pos'][(n*seglen):((n+1)*seglen),:,:])

    #### Compute allowed wavevectors ####
    dim=traj['dim']
    if spacing==0.0:
        spacing = 2*np.pi/(np.max(traj['edges'])) #spacing
        
    qvals = get_allowed_q(qmax, spacing, dim)

    #### Compute S(q) for each wavevector ####
    for n in range(nchunks):
        print('chunk', n)
        q1d, sqavg, sqvals = get_sq_range(pos_chunks[n], traj['dim'], traj['edges'], qvals)
        the_dict['sq_vals_%d' % n] = sqvals
        the_dict['sq_vals_1d_%d' % n] = sqavg
    the_dict['qvals'] = qvals
    the_dict['qvals_1d'] = q1d
    the_dict['qmag'] = np.linalg.norm(qvals, axis=1)

    return the_dict

def get_sq_traj(traj, spacing=0.0, qmax=15.0):

    """
    Compute instantaneous structure factor vs time.

    INPUT: Particle trajectory (dictionary),
           spacing in q space,
           max q value,
    OUTPUT: Dictionary containing S(q)(t)
    """

    the_dict = {}

    #### Compute allowed wavevectors ####
    dim=traj['dim']
    if spacing==0.0:
        spacing = 2*np.pi/(np.max(traj['edges'])) #spacing
        
    qvals = get_allowed_q(qmax, spacing, dim)

    #### Compute S(q) for each wavevector and time ####
    sqavg_list = []
    sqvals_list = []
    for t in range(traj['pos'].shape[0]):
        post = traj['pos'][t,:,:]
        q1d, sqavg, sqvals = get_sq_range(post[np.newaxis,:,:], traj['dim'], traj['edges'], qvals)
        sqavg_list.append(sqavg)
        sqvals_list.append(sqvals)
    the_dict['sq_vals'] = np.array(sqvals_list)
    the_dict['sq_vals_1d'] = np.array(sqavg_list)
    the_dict['qvals'] = qvals
    the_dict['qvals_1d'] = q1d
    the_dict['qmag'] = np.linalg.norm(qvals, axis=1)
    the_dict['times'] = traj['times']

    return the_dict

def get_sq(traj, nchunks=5, nlast=3, spacing=0.0, qmax=np.pi):

    """
    Compute static structure factor.

    INPUT: Particle trajectory (dictionary),
           number of chunks to divide trajectory into,
           spacing in q space,
           max q value
    OUTPUT: Dictionary containing S(q) for each chunk,
            also S(q) averaged over last nlast chunks
    """

    the_dict = {}
    the_dict['nchunks'] = nchunks
    the_dict['nlast'] = nlast
    seglen = traj['pos'].shape[0]//nchunks

    #### Chunk positions ####
    pos_chunks = []
    print(traj['pos'].shape)
    for n in range(nchunks):
        pos_chunks.append(traj['pos'][(n*seglen):((n+1)*seglen),:,:])

    #### Compute allowed wavevectors ####
    dim=traj['dim']
    if spacing==0.0:
        spacing = 2*np.pi/(np.max(traj['edges'])) #spacing
        
    qvals = get_allowed_q(qmax, spacing, dim)

    #### Compute S(q) for each wavevector ####
    for n in range(nchunks):
        print('chunk', n)
        q1d, sqavg, sqvals = get_sq_range(pos_chunks[n], traj['dim'], traj['edges'], qvals)
        the_dict['sq_vals_%d' % n] = sqvals
        the_dict['sq_vals_1d_%d' % n] = sqavg
    the_dict['sq_vals_nlast'] = sum([the_dict['sq_vals_%d' % n] for n in range(nchunks-nlast, nchunks)])/nlast
    the_dict['sq_vals_1d_nlast'] = sum([the_dict['sq_vals_1d_%d' % n] for n in range(nchunks-nlast, nchunks)])/nlast
    the_dict['qvals'] = qvals
    the_dict['qvals_1d'] = q1d
    the_dict['qmag'] = np.linalg.norm(qvals, axis=1)

    return the_dict

@numba.jit(nopython=True)
def get_sq_range(pos, dim, edges, qvals):

    sqvals = np.zeros(qvals.shape[0],dtype=numba.float64)
    for i in range(qvals.shape[0]):
        #print(qvals[i,:])
        sqvals[i] = get_single_point_sq(pos, dim, edges, qvals[i,:])

    #Get "isotropic" S(q) by histogramming
    #make big enough to only group values with same |q|
    nbins = 1000#int(2.0/(2*np.pi/edges[0]))
    print('num bins:', nbins)
    qnorm = np.zeros(qvals.shape)
    for i in range(qvals.shape[0]):
        qnorm[i] = np.linalg.norm(qvals[i])
    q1d = np.linspace(0,np.max(qnorm)*(1+1.0/nbins),num=nbins)
    counts = np.zeros(nbins,dtype=numba.float64)
    sqavg = np.zeros(nbins, dtype=numba.float64)
    for i in range(qnorm.shape[0]):
        index = int(np.floor(qnorm[i]/np.max(q1d)*nbins)[0])
        counts[index] += 1.0
        sqavg[index] += sqvals[i]
    sqavg = np.divide(sqavg,counts)
    q1d = q1d[1:]
    sqavg = sqavg[1:]
    q1d = (q1d)[~np.isnan(sqavg)]
    sqavg = sqavg[~np.isnan(sqavg)]

    return q1d, sqavg, sqvals

@numba.jit(nopython=True)
def get_single_point_sq(pos, dim, edges, q):

    sq = 0
    N = pos.shape[1]
    traj_len = pos.shape[0]
    #print('q:', q)
    for t in range(traj_len):
        #rho = 0. + 0.j
        rho_real = 0
        rho_imag = 0
        for i in range(N):
            mypos = pos[t,i,:dim]
            mypos[0] += edges[0]/2.0
            mypos[1] += edges[1]/2.0
            #rho_real += np.cos(np.dot(q,mypos))
            #rho_imag += np.sin(np.dot(q,mypos))
            rho_real += np.cos(q[0]*mypos[0]+q[1]*mypos[1])
            rho_imag += np.sin(q[0]*mypos[0]+q[1]*mypos[1])
            #rho += np.exp(-1j*np.dot(q, pos[t,i,:dim]))
        sq += rho_real**2 + rho_imag**2
    sq = sq*(1.0/(N*(traj_len)))

    return sq

if __name__ == '__main__':
    main()
