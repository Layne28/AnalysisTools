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
    
    #Do tests
    test_get_count_matrix()
    
    ### Load data ####
    parser = argparse.ArgumentParser(description='Tools for building and using MSMs.')
    parser.add_argument('--mode', default='build', help='"build" or "analyze"')
    parser.add_argument('--traj_type', default='single', help='"single" trajectory or "multiple" trajectories')
    parser.add_argument('--input_file', default='traj.h5', help='Input trajectory file (for "build" option") or .npz file (for "analyze" option).')
    parser.add_argument('--input_folder', default='.', help='Input folder (for "build" option and traj_type="multiple" only).')
    parser.add_argument('--output_folder', default='.', help='Where to put results.')
    parser.add_argument('--lag_time', default = 1, help='MSM lag time (float)')
    parser.add_argument('--state_decomp_type', default='double_well', help='Method to define microstates.')
    
    args = parser.parse_args()
    mode = args.mode
    traj_type = args.traj_type
    input_folder = args.input_folder
    output_folder = args.output_folder
    lag_time = float(args.lag_time)
    input_file = args.input_file
    state_decomp_type = args.state_decomp_type
    
    if mode=='build':
        build_msm(traj_type, input_folder, input_file, output_folder, lag_time, state_decomp_type)
        
    elif mode=='analyze':
        my_msm = np.load(input_file)
        analyze_msm(my_msm)
        
    else:
        print('Error: mode "%s" not implemented. Mode should be either "build" or "analyze."' % mode)
        exit()
    
    
#### Methods #### 

def build_msm(traj_type, input_folder, input_file, output_folder, lag_time, state_decomp_type):
    
    """
    Build an MSM from trajectory data.
    
    INPUT: traj_type (string, "single" or "multiple"),
           input_folder (string, used for for "multiple")
           input_file (string, used for for "single")
    """
    
    if traj_type=='single':
        traj = particle_io.load_traj(input_file)
        
        #Decompose trajectory into microstates
        microstate_traj, nstates = get_state_decomp_traj(traj, state_decomp_type)
        
        #Get (unnormalized) count matrix
        dt = traj['times'][1]-traj['times'][0] #get time difference between subsequent frames
        lag_num = int(round(lag_time/dt))
        C = get_count_matrix(microstate_traj, lag_num, nstates)
        
        #Apply detailed balance and normalize count matrix
        C = C/np.linalg.norm(C, axis=0)
        C = (C+C.T)/2 #symmetrize count matrix to enforce detailed balance
        print(C)
        
        #Output transition matrix 
        np.savez(output_folder + '/tmat_tlag=%f.npz' % lag_time)

    elif traj_type=='multiple':
        print('Error: "multiple" option not yet implemented!')
        exit()
    else:            
        print('Error: "traj_type" not recognized.')
        exit()
        
    return 0

def analyze_msm(my_msm):
    return 0   


def get_state_decomp_traj(traj, state_decomp_type):
    
    if state_decomp_type=='double_well':
        nstates = 2
        microstate_traj = np.where(traj['pos'][:,0,0]<0, 0, 1)
    else:
        print('Error: state_decomp_type "%s" not yet implemented.' % state_decomp_type)
        exit()
    
    return microstate_traj, nstates


def get_count_matrix(traj, lag_time, nstates):
    
    """
    Get count matrix from a state-decomposed trajectory for a given lag time
    
    INPUT: State-decomposed trajectory (1d numpy array of microstate labels)
           Lag time (units of num frames)
           Number of microstates (int)
    OUTPUT: 2d numpy array (matrix) of transition counts
    """
    
    #For now, assume traj is an array of ints which label the states
    C = np.zeros((nstates, nstates))
    
    #C[traj[:-1],traj[1:]]
    #subsample trajectory every lag_time
    traj = traj[::lag_time]
    for i in range(traj.shape[0]-1):
        ind1 = traj[i]
        ind2 = traj[i+1]
        C[ind1,ind2] += 1
        
    return C
    
    
def test_get_count_matrix():
    
    #Test get_count_matrix
    #Should be:
    #C = [[1, 1],
    #     [1, 2]]
    
    C_actual = np.array([[1,1],[1,2]])
    
    traj = np.array([0,1,1,1,0,0])
    
    C = get_count_matrix(traj, 1, 2)
    
    diff = np.sum(np.abs(C-C_actual))
    
    if diff>1e-10:
        print('get_count_matrix test failed!')
    else:
        print('get_count_matrix test passed!')
        

if __name__ == '__main__':
    main()