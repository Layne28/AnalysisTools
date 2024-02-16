#Compute histogram of specified quantity

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as tools

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_traj(myfile) #Extract data
    quantity = sys.argv[2]
    if quantity=='vel' or quantity=='velocity':
        vel = traj['vel']
        vx_hist = tools.get_histogram(vel[:,:,0], nskip=1)
        if traj['dim']==2 or traj['dim']==3:
            vy_hist = tools.get_histogram(vel[:,:,1], nskip=1)
        if traj['dim']==3:
            vz_hist = tools.get_histogram(vel[:,:,2], nskip=1)
        speed_hist = tools.get_histogram(np.linalg.norm(vel, axis=2), nskip=1)

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

if __name__ == '__main__':
    main()