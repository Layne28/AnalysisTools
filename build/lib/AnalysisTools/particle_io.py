#This script contains functions for dealing with h5md trajectory I/O

import h5py
import numpy as np
import sys

def load_traj(myfile):
    
    #Check whether this is an h5 file
    if not(myfile.endswith('.h5')):
        print('Error: input to "load_traj" must be h5md file.')
        return {}

    traj = h5py.File(myfile)
    traj_dict = {}
    has_topology = 0

    pos = np.array(traj['/particles/all/position/value'])
    vel = np.array(traj['/particles/all/velocity/value'])
    potential_energy = np.array(traj['/observables/potential_energy/value'])
    active_force = np.array(traj['/particles/all/active_force/value'])
    conservative_force = np.array(traj['/particles/all/conservative_force/value'])
    image = np.array(traj['/particles/all/image/value'])
    times = np.array(traj['/particles/all/position/time'])
    edges = np.array(traj['/particles/all/box/edges'])
    dim = traj['/particles/all/box'].attrs['dimension']
    if (('parameters/vmd_structure/bond_from' in traj) and
        ('parameters/vmd_structure/bond_to' in traj)):
        has_topology = 1
        bonds_from = np.array(traj['parameters/vmd_structure/bond_from'])-1
        bonds_to = np.array(traj['parameters/vmd_structure/bond_to'])-1
        bonds = np.vstack((bonds_from, bonds_to)).T
        bonds = np.unique(bonds, axis=0)
    elif ('/particles/all/connectivity' in traj):
        has_topology = 1
        bonds = np.array(traj['/particles/all/connectivity/value'])
    else:
        has_topology = 0


    N = pos.shape[1]
    
    traj.close()

    traj_dict['pos'] = pos
    traj_dict['vel'] = vel
    traj_dict['active_force'] = active_force
    traj_dict['conservative_force'] = conservative_force
    traj_dict['potential_energy'] = potential_energy
    traj_dict['image'] = image
    traj_dict['times'] = times
    traj_dict['edges'] = edges
    traj_dict['N'] = N
    traj_dict['dim'] = dim
    if has_topology:
        traj_dict['bonds'] = bonds

    return traj_dict

def load_noise_traj(myfile):
    
    #Check whether this is an h5 file
    if not(myfile.endswith('.h5')):
        print('Error: input to "load_noise_traj" must be h5md file.')
        return {}

    traj = h5py.File(myfile)
    traj_dict = {}

    #noise parameters
    Lambda = np.array(traj['/parameters/lambda'])
    tau = np.array(traj['/parameters/tau'])
    D = np.array(traj['/parameters/D'])

    #grid dimensions
    ncells = np.array(traj['/grid/dimensions'])
    spacing = np.array(traj['/grid/spacing'])
    dim = ncells.shape[0]

    #noise data
    times = np.array(traj['/noise/time'])
    if dim==1:
        noise = np.zeros((times.shape[0],ncells[0],1))
        noise = np.array(traj['/noise/value/x'])
    elif dim==2:
        noise = np.zeros((times.shape[0],ncells[0],ncells[1],2))
        noise[:,:,:,0] = np.array(traj['noise/value/x'])
        noise[:,:,:,1] = np.array(traj['noise/value/y'])
    elif dim==3:
        noise = np.zeros((times.shape[0],ncells[0],ncells[1],ncells[2],3))
        noise[:,:,:,:,0] = np.array(traj['noise/value/x'])
        noise[:,:,:,:,1] = np.array(traj['noise/value/y'])
        noise[:,:,:,:,2] = np.array(traj['noise/value/z'])
    else:
        print('Error: dim is ', dim, ', not 1, 2, or 3.')
        traj.close()
        exit()

    traj.close()

    #Assign data to dictionary
    traj_dict['tau'] = tau
    traj_dict['lambda'] = Lambda
    traj_dict['D'] = D

    traj_dict['dim'] = dim
    traj_dict['ncells'] = ncells
    traj_dict['spacing'] = spacing

    traj_dict['times'] = times
    traj_dict['noise'] = noise

    return traj_dict

if __name__ == '__main__':
    myfile = sys.argv[1]
    traj = load_traj(myfile)
    print(traj)