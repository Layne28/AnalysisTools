#Compute potential and kinetic energies, 
#plus average forces

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
    pe = traj['potential_energy']
    ke = get_kinetic_energy(traj['vel'])
    net_active_force = get_net_force(traj['active_force'])
    net_conservative_force = get_net_force(traj['conservative_force'])

    #### Output average energies and forces to file in same directory as input h5 file ####        
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/energies_and_forces.npz'
    np.savez(outfile, potential_energy=pe, kinetic_energy=ke, net_active_force=net_active_force, net_conservative_force=net_conservative_force)

@numba.jit(nopython=True)
def get_kinetic_energy(vel):

    """
    Compute kinetic energy from velocity at each timestep.
    
    INPUT: Velocities (nframes x N x d numpy array)
    OUTPUT: Kinetic energy (nframes numpy array)
    """
    
    ke = np.zeros(vel.shape[0])
    for t in range(vel.shape[0]):
        for i in range(vel.shape[1]):
            for mu in range(vel.shape[2]):
                ke[t] += 0.5*vel[t,i,mu]*vel[t,i,mu]

    return ke

@numba.jit(nopython=True)
def get_net_force(forces):

    """
    Compute net force at each timestep.
    
    INPUT: Forces (nframes x N x d numpy array)
    OUTPUT: Net force (nframes x d numpy array)
    """

    net_force = np.zeros((forces.shape[0],forces.shape[-1]))
    for t in range(forces.shape[0]):
        for i in range(forces.shape[1]):
            for mu in range(forces.shape[2]):
                net_force[t,mu] += forces[t,i,mu]

if __name__ == '__main__':
    main()