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
mpl.use('Agg')
import matplotlib.pyplot as plt

import particle_io
import measurement_tools

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_traj(myfile) #Extract data
    eq_frac = float(sys.argv[2]) #cut off first eq_frac*100% of data (equilibration)

    #### Compute allowed wavevectors ###
    dim=3
    if traj['edges'][2]==0.0 and traj['edges'][1]!=0.0:
        dim=2
    elif traj['edges'][2]==0.0 and traj['edges'][1]==0.0:
        dim=1
    dq = np.pi/(5*np.max(traj['edges'])) #spacing
    qmax = 2*np.pi #Assumes smallest relevant distance =1 for now
    qvals = get_allowed_q(qmax, dq, dim)

    #### Compute S(q) for each wavevector ####
    sqvals = get_sq_range(traj['pos'], traj['edges'], qvals, eq_frac)

    #### Output S(q) to file in same directory as input h5 file ####        
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/sq.npz'
    np.savez(outfile, q=qvals, qmag=np.linalg.norm(qvals, axis=1), sq=sqvals)

    fig = plt.figure()
    if dim==3:
        plt.scatter(qvals[(qvals[:,1]==0) & (qvals[:,2]==0)][:,0],sqvals[(qvals[:,1]==0) & (qvals[:,2]==0)],color='blue',s=1)
    elif dim==2:
        plt.scatter(qvals[(qvals[:,1]==0)][:,0],sqvals[(qvals[:,1]==0)],color='blue',s=1)
    else:
        plt.plot(qvals[:,0],sqvals,color='blue')
    plt.yscale('log')
    plt.savefig('test.png')
    plt.show()

#### Methods ####

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

@numba.jit(nopython=True)
def get_sq_range(pos, edges, qvals, eq_frac):

    sqvals = np.zeros(qvals.shape[0],dtype=numba.complex128)
    for i in range(qvals.shape[0]):
        print(qvals[i,:])
        sqvals[i] = get_sq(pos, edges, qvals[i,:], eq_frac)

    return sqvals

@numba.jit(nopython=True)
def get_sq(pos, edges, q, eq_frac):

    sq = 0. + 0.j
    N = pos.shape[1]
    traj_len = pos.shape[0]
    eq_len = int(eq_frac*traj_len)
    for t in range(eq_len, traj_len):
        rho = 0. + 0.j
        for i in range(N):
            rho += np.exp(-1j*np.dot(q, pos[t,i,:]))
        sq += rho * np.conjugate(rho)
    sq *= (1.0/(N*(traj_len-eq_len)))

    return sq

if __name__ == '__main__':
    main()