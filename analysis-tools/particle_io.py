#This script contains functions for dealing with h5md trajectory I/O

import h5py
import numpy as np
import sys

def load_traj(myfile):
    
    traj = h5py.File(myfile)
    traj_dict = {}

    pos = np.array(traj['/particles/all/position/value'])
    vel = np.array(traj['/particles/all/velocity/value'])
    times = np.array(traj['/particles/all/position/time'])
    edges = np.array(traj['/particles/all/box/edges'])
    N = pos.shape[1]
    
    traj.close()

    traj_dict['pos'] = pos
    traj_dict['vel'] = vel
    traj_dict['times'] = times
    traj_dict['edges'] = edges
    traj_dict['N'] = N

    return traj_dict

if __name__ == '__main__':
    myfile = sys.argv[1]
    traj = load_traj(myfile)
    print(traj)