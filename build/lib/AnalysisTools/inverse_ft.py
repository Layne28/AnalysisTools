#Compute and output the real-space correlation function
#given a fourier space power spectrum

#Input: Fourier space function in npz format
#Output: Correlation function vs r in npz format

import numpy as np
import h5py
import sys
import numba
import argparse

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as measurement_tools
import AnalysisTools.trajectory_stats as stats
import AnalysisTools.structure_factor as structure_factor

def main():

    ### Load data ####
    parser = argparse.ArgumentParser(description='Compute statistics over trajectories (either avg+stderr OR histogram OR CSD).')
    parser.add_argument('myfile', help='Input trajectory file.')
    parser.add_argument('--spacing', default='0.5', help='spacing in real space')
    parser.add_argument('--rmax', default='30.0', help='max distance to compute correlation')

    args = parser.parse_args()
    myfile = args.myfile #Expects .npz input file
    data = np.load(myfile) #Extract data
    print(data)
    spacing = float(args.spacing)
    rmax = float(args.rmax)
    
    sqvals = 0.0
    if 'avg' in myfile:
        if '/sq' in myfile:
            sqvals = data['sq_vals_1d_nlast_avg']
        elif '/pressure' in myfile:
            sqvals = data['p2q_vals_1d_nlast_avg']
        elif '/strain' in myfile:
            sqvals = data['str2q_vals_1d_nlast_avg']
        qvals = data['qvals_avg']
    else:
        if '/sq' in myfile:
            sqvals = data['sq_vals_1d_nlast']
        elif '/pressure' in myfile:
            sqvals = data['p2q_vals_1d_nlast']
        elif '/strain' in myfile:
            sqvals = data['str2q_vals_1d_nlast']
        qvals = data['qvals']
    
    print(sqvals)
    print(qvals)
    for i in range(qvals.shape[0]):
        print(qvals[i,:])
    
    print('Computing inverse transform...')
    cr = compute_inverse_correlation(sqvals, qvals, 2, spacing=spacing, rmax=rmax)
    print('Computed inverse transform.')

    #### Output C(r) to file in same directory as input S(q) file ####        
    mystr = ((myfile.split('/'))[-1]).replace('q', 'r')
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/' + mystr
    print(outfile)
    np.savez(outfile, **cr)


#### Methods ####

def compute_inverse_correlation(sqvals, qvals, dim, spacing=0.5, rmax=30.0):
    
    the_dict = {}
    
    rvals = get_allowed_r(rmax, spacing, dim)
    
    r1d, cravg, crvals = get_inverse_range(sqvals, qvals, dim, rvals)
    
    the_dict['corr'] = crvals
    the_dict['rvals'] = rvals
    the_dict['corr_1d'] = cravg
    the_dict['rvals_1d'] = r1d
    
    return the_dict

@numba.jit(nopython=True)
def get_inverse_range(sqvals, qvals, dim, rvals):
    
    crvals = np.zeros(rvals.shape[0], dtype=numba.float64)
    for i in range(rvals.shape[0]):
        crvals[i] = get_single_point_cr(sqvals, qvals, dim, rvals[i,:])
        
    #Get "isotropic" C(r) by histogramming
    #make big enough to only group values with same |r|
    
    nbins = 1000#int(2.0/(2*np.pi/edges[0]))
    rnorm = np.zeros(rvals.shape[0])
    for i in range(rvals.shape[0]):
        rnorm[i] = np.linalg.norm(rvals[i])
    r1d = np.linspace(0,np.max(rnorm)*(1+1.0/nbins),num=nbins)
    counts = np.zeros(nbins,dtype=numba.float64)
    cravg = np.zeros(nbins, dtype=numba.float64)
    for i in range(rnorm.shape[0]): 
        index = int(np.floor(rnorm[i]/np.max(r1d)*nbins))
        counts[index] += 1.0
        cravg[index] += crvals[i]
    cravg = np.divide(cravg,counts)
    r1d = r1d[1:]
    cravg = cravg[1:]
    r1d = (r1d)[~np.isnan(cravg)]
    cravg = cravg[~np.isnan(cravg)]

    return r1d, cravg, crvals
    
    
@numba.jit(nopython=True)
def get_single_point_cr(sqvals, qvals, dim, r):
    
    cr = 0
    N = sqvals.shape[0]
    rho_real = 0
    rho_imag = 0
    for i in range(N):
        rho_real += np.cos(-qvals[i,0]*r[0]-qvals[i,1]*r[1])*sqvals[i]
        rho_imag += np.sin(-qvals[i,0]*r[0]-qvals[i,1]*r[1])*sqvals[i]
    cr = rho_real#**2 + rho_imag**2
    cr = cr/N
    
    return cr
    


@numba.jit(nopython=True)
def get_allowed_r(rmax, dr, dim):

    """
    Generate a grid of points in space (w/ lattice constant dr),
    then select those within a sphere of radius rmax.
    Exclude one half line/disk/sphere because of the symmetry
    C(r) = C(-r)
    """

    rvals = np.arange(-rmax, rmax+dr, dr)

    if dim==3:
        rlist = np.zeros((rvals.shape[0]**3,3))
        cnt = 0
        for kx in range(rvals.shape[0]):
            for ky in range(rvals.shape[0]):
                for kz in range(rvals.shape[0]):
                    rvec = np.array([rvals[kx], rvals[ky], rvals[kz]])
                    if not(np.abs(rvals[kx])<1e-8 and np.abs(rvals[ky])<1e-8 and np.abs(rvals[kz])<1e-8) and np.any(structure_factor.np_all_axis1(np.abs(rlist+rvec)<1e-8)):
                        #print(qvec, 'equivalent to', -qvec, 'by symmetry')
                        continue
                    if np.linalg.norm(rvec)<=rmax:
                        rlist[cnt,:] = rvec
                        cnt += 1
    elif dim==2:
        rlist = np.zeros((rvals.shape[0]**2,2))
        cnt = 0
        for kx in range(rvals.shape[0]):
            for ky in range(rvals.shape[0]):
                if np.abs(rvals[kx])<1e-8 and np.abs(rvals[ky])<1e-8:
                    print(np.array([rvals[kx], rvals[ky]]))
                rvec = np.array([rvals[kx], rvals[ky]])
                if not(np.abs(rvals[kx])<1e-8 and np.abs(rvals[ky])<1e-8) and np.any(structure_factor.np_all_axis1(np.abs(rlist+rvec)<1e-8)):
                    #print(qvec, 'equivalent to', -qvec, 'by symmetry')
                    continue
                if np.linalg.norm(rvec)<=rmax:
                    rlist[cnt,:] = rvec
                    cnt += 1
    elif dim==1:
        rlist = np.zeros((rvals.shape[0],1))
        cnt = 0
        for kx in range(rvals.shape[0]):
            rvec = np.array([rvals[kx]])
            if not(np.abs(rvals[kx])<1e-8) and np.any(structure_factor.np_all_axis1(np.abs(rlist+rvec)<1e-8)):
                #print(qvec, 'equivalent to', -qvec, 'by symmetry')
                continue
            if np.linalg.norm(rvec)<=rmax:
                rlist[cnt,:] = rvec
                cnt += 1

    else:
        print('Error: dim must be 1, 2, or 3.')
        raise ValueError

    return rlist[:cnt,:]

if __name__ == '__main__':
    main()
