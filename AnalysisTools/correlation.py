#Compute correlation functions

import numpy as np
import h5py
import sys
import numba
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools
import AnalysisTools.cell_list as cl

def main():
    
    ### Load data ####
    parser = argparse.ArgumentParser(description='Compute correlation function.')
    parser.add_argument('myfile', help='Input trajectory file.')
    parser.add_argument('--quantity', default='velocity', help='velocity, stress, strain...')
    parser.add_argument('--corrtype', default='time', help='"time" or "space"')
    parser.add_argument('--nchunks', default=5, help='No. of trajectory chunks')
    parser.add_argument('--nlast', default=3, help='No. of chunks to lump together for "equilibrated" average')
    parser.add_argument('--tmax', default=50.0, help='Max time to compute time correlations')
    parser.add_argument('--rmax', default=20.0, help='Max distance to compute spatial correlations')
    
    args = parser.parse_args()
    myfile = args.myfile
    traj = particle_io.load_traj(myfile)
    nchunks = int(args.nchunks)
    nlast = int(args.nlast)
    tmax = float(args.tmax)
    rmax = float(args.rmax)
    corrtype = args.corrtype
    quantity = args.quantity
    
    ### Compute correlation function ###
    
    if quantity=='strain':
        obs = measurement_tools.get_strain_bonds(traj['pos'], traj['bonds'], traj['edges'], 1.0)
    elif quantity=='stress':
        if traj['dim']==2:
            obs = -(traj['virial'][:,:,0]+traj['virial'][:,:,3])/2.0
        else:
            obs = -(traj['virial'][:,:,0]+traj['virial'][:,:,3]+traj['virial'][:,:,5])/3.0
    else:
        if quantity=='velocity':
            obs = traj['vel']
        else:
            obs = traj[quantity]
    
    nlaststeps = int(obs.shape[0]*nlast/nchunks)
    obs = obs[-nlaststeps:,:,:]
    
    if corrtype == 'time':
        corr = get_single_particle_time_corr(obs, traj['times'][-nlaststeps:], tmax)
        outfile = '/'.join((myfile.split('/'))[:-1]) + ('/%s_time_corr.npz' % quantity)
        print(outfile)
        the_dict = {}
        the_dict['times'] = corr[:,0]
        the_dict['corr'] = corr[:,1]
        the_dict['nchunks'] = nchunks
        the_dict['nlast'] = nlast
        np.savez(outfile, **the_dict)
    elif corrtype == 'space':
        corr = get_single_particle_radial_corr(obs, traj['pos'], traj['edges'], traj['dim'], rmax=rmax)
        outfile = '/'.join((myfile.split('/'))[:-1]) + ('/%s_spatial_corr.npz' % quantity)
        print(outfile)
        the_dict = {}
        the_dict['distances'] = corr[:,0]
        the_dict['corr'] = corr[:,1]
        the_dict['nchunks'] = nchunks
        the_dict['nlast'] = nlast
        np.savez(outfile, **the_dict)
    else:
        print('Error: corrtype not recognized.')
        exit()
    
    
#### Methods ####    

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
    
    #TODO: need to handle obs=strain separately

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
            if t%1==0:
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

###################################
#Noise functions
###################################

@numba.jit(nopython=True)
def get_time_corr_noise(noise, times, tmax=100):
    
    """
    Compute time correlation of noise trajectory.
    
    INPUT: Noise trajectory ((d+2)-dimensional numpy array, with
           first dimension time (nframes),
           next d dimensions {n_mu, mu=1,2,...,d},
           and last dimension
           d (components of field))
    OUTPUT: Time correlation function of noise field (numpy array)
    """
    
    if len(noise.shape)==2:
        dim = 1
    else:
        dim = noise.shape[-1]
        
    if dim>3:
        print('Error: dimension cannot be greater than 3!')
        raise TypeError
    if times.shape[0]<2:
        print('Warning: not enough time to compute time correlations! Returning zero.')
        return np.zeros((1, 2))
        #raise TypeError

    dt = times[1]-times[0]
    #N = obs.shape[1]
    print(dt)
    fmax = int(tmax/dt) #max frame
    if fmax>times.shape[0]:
        fmax = times.shape[0]-1
    print('fmax:', fmax)
    corr = np.zeros((fmax,2))
    
    for t in range(fmax):
        corr[t,0] = dt*t
        corr[t,1] += np.mean(noise[:-fmax,...]*noise[t:(-fmax+t),...])

    return corr

def get_space_corr_noise(noise, spacing, rmax=20):
    
    """
    Compute spatial correlation of noise trajectory.
    
    INPUT: Noise trajectory ((d+2)-dimensional numpy array, with
           first dimension time (nframes),
           next d dimensions {n_mu, mu=1,2,...,d},
           and last dimension
           d (components of field))
    OUTPUT: Spatial correlation function of noise field (numpy array)
    """
    
    dim = noise.shape[-1]
    nsteps = noise.shape[0]
        
    if dim>3:
        print('Error: dimension cannot be greater than 3!')
        raise TypeError
    
    if dim==1:
        Narr = np.array([noise.shape[1]])
        fourier_corr = np.zeros(Narr[0]//2+1, dtype=np.complex64)
        real_corr = np.zeros(Narr[0])
    elif dim==2:
        Narr = np.array([noise.shape[1],noise.shape[2]])
        fourier_corr = np.zeros((Narr[0],Narr[1]//2+1), dtype=np.complex64)
        real_corr = np.zeros((Narr[0], Narr[1]))
    else:
        Narr = np.array([noise.shape[1],noise.shape[2],noise.shape[3]])
        fourier_corr = np.zeros((Narr[0],Narr[1],Narr[2]//2+1), dtype=np.complex64)
        real_corr = np.zeros((Narr[0], Narr[1], Narr[2]))

    for t in range(nsteps):
        print(t)
        for d in range(dim):
            #Get fourier components of field
            if dim==1:
                fourier_noise = np.fft.rfft(noise[t,:], axis=0)
                fourier_corr += fourier_noise.real**2 + fourier_noise.imag**2
            elif dim==2:
                fourier_noise = np.fft.rfft2(noise[t,...,d], axes=(0,1))
                fourier_corr += fourier_noise.real**2 + fourier_noise.imag**2
            else:
                fourier_noise = np.fft.rfftn(noise[t,...,d], axes=(0,1,2))
                fourier_corr += fourier_noise.real**2 + fourier_noise.imag**2
        if dim==1:
            real_corr += np.fft.irfft(fourier_corr, axis=0)
        elif dim==2:
            real_corr += np.fft.irfft2(fourier_corr, axes=(0,1))
        else:
            real_corr += np.fft.irfftn(fourier_corr, axes=(0,1,2))

    real_corr /= (dim*nsteps)
    
    if dim==1:
        x = np.linspace(0,noise.shape[-1]*spacing[0], noise.shape[-1])
        n = x.shape[0]
        x[n//2 + 1:] = -np.flip(x[1:n//2])
        r = np.abs(x)
    elif dim==2:
        x = np.linspace(0,noise.shape[1]*spacing[0], noise.shape[1])
        nx = x.shape[0]
        x[nx//2 + 1:] = -np.flip(x[1:nx//2])
        y = np.linspace(0,noise.shape[2]*spacing[1], noise.shape[2])
        ny = y.shape[0]
        y[ny//2 + 1:] = -np.flip(y[1:ny//2])
        xv, yv = np.meshgrid(x, y)
        r = np.sqrt(xv**2+yv**2)
    else:
        x = np.linspace(0,noise.shape[1]*spacing[0], noise.shape[1])
        nx = x.shape[0]
        x[nx//2 + 1:] = -np.flip(x[1:nx//2])
        y = np.linspace(0,noise.shape[2]*spacing[1], noise.shape[2])
        ny = y.shape[0]
        y[ny//2 + 1:] = -np.flip(y[1:ny//2])
        z = np.linspace(0,noise.shape[3]*spacing[2], noise.shape[3])
        nz = z.shape[0]
        z[nz//2 + 1:] = -np.flip(z[1:nz//2])
        xv, yv, zv = np.meshgrid(x, y, z)
        r = np.sqrt(xv**2+yv**2+zv**2)
    
    rvals = np.unique(r.round(decimals=6))
    radial_corr = np.zeros(rvals.shape)
    for i in range(rvals.shape[0]):
        radial_corr[i] = np.mean(real_corr[np.abs(r-rvals[i])<1e-6])

    return real_corr, r, radial_corr, rvals, spacing

if __name__ == '__main__':
    main()