#This script contains functions for dealing with h5md trajectory I/O

import h5py
import numpy as np
import sys

def load_traj(myfile):
    
    traj = h5py.File(myfile)
    traj_dict = {}
    has_topology = 0

    pos = np.array(traj['/particles/all/position/value'])
    vel = np.array(traj['/particles/all/velocity/value'])
    times = np.array(traj['/particles/all/position/time'])
    edges = np.array(traj['/particles/all/box/edges'])
    if (('parameters/vmd_structure/bond_from' in traj) and
        ('parameters/vmd_structure/bond_to' in traj)):
        has_topology = 1
        bonds_from = np.array(traj['parameters/vmd_structure/bond_from'])-1
        bonds_to = np.array(traj['parameters/vmd_structure/bond_to'])-1
        bonds = np.vstack((bonds_from, bonds_to)).T
        bonds = np.unique(bonds, axis=0)
    else:
        has_topology = 0


    N = pos.shape[1]
    
    traj.close()

    traj_dict['pos'] = pos
    traj_dict['vel'] = vel
    traj_dict['times'] = times
    traj_dict['edges'] = edges
    traj_dict['N'] = N
    if has_topology:
        traj_dict['bonds'] = bonds

    return traj_dict

if __name__ == '__main__':
    myfile = sys.argv[1]
    traj = load_traj(myfile)
    print(traj)