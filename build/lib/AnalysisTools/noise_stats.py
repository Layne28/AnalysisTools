#Compute histogram of specified quantity

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as tools
import AnalysisTools.correlation as corr

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = particle_io.load_noise_traj(myfile) #Extract data
    
    rms_noise = tools.get_rms_noise(traj['noise'])
    print('rms noise:', rms_noise)
    
    print('Getting time correlation...')
    print(traj['tau'])
    tcorr = corr.get_time_corr_noise(traj['noise'], traj['times'])
    print(tcorr)
    print('done')

    outfile = '/'.join((myfile.split('/'))[:-1]) + '/noise_stats.npz'
    np.savez(outfile, rms=rms_noise, tcorr=tcorr)

if __name__ == '__main__':
    main()