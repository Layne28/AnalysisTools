#Compute statistics over trajectories

import numpy as np
import h5py
import sys
import os

import AnalysisTools.particle_io as particle_io
import AnalysisTools.measurement_tools as tools

def main():

    ### Load data ####
    basefolder = sys.argv[1]
    filename = sys.argv[2] #Should either have no extension or be .npz
    if not('.npz' in filename):
        filename += '.npz'
    subfolder = ''
    if len(sys.argv)>3:
        subfolder = sys.argv[3]
    data = get_trajectory_data(basefolder, filename, subfolder=subfolder)
    mystats = get_trajectory_stats(data)
    np.savez(basefolder + '/' + filename.split('.npz')[0] + '_avg.npz', **mystats)

def get_trajectory_data(basefolder, filename, subfolder='prod'):

    """
    Retrieve data for different random seeds.

    INPUT: Name of folder containing subfolders named "seed=*"
           Name of data file (npz format)
           Name of subfolder within each "seed=*" folder to look in
    OUTPUT: List of dictionaries containing data for each trajectory.
    """

    #Check that basefolder contains "seed=*"
    dirs = [d for d in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, d)) and 'seed=' in d]
    if len(dirs)==0:
        raise Exception('Error: base folder does not contain seed subdirectories! Exiting.')

    data_list = []
    for d in dirs:
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                data = np.load(dir + '/' + filename)
            else:
                data = np.load(dir + '/' + subfolder + '/' + filename)
            data_list.append(data)
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping...')
    if len(data_list)==0:
        raise Exception('Error: no trajectory data found! Exiting.')
    
    return data_list

def get_trajectory_avg(data_list):

    """
    Compute average of data over trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing trajectory-averaged data
    """
    
    keys = data_list[0].keys()
    avg = dict.fromkeys(keys)
    for data in data_list:
        for key in data.keys():
            if avg[key] is None:
                avg[key] = data[key]
            else:
                avg[key] += data[key]
    for key in data.keys():
        avg[key] /= len(data_list)

    return avg

def get_trajectory_stderr(data_list):

    """
    Compute standard error of data over trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing standard error of data
    """

    avg = get_trajectory_avg(data_list)
    keys = avg.keys()
    stderr = dict.fromkeys(keys)

    for data in data_list:
        for key in data.keys():
            if stderr[key] is None:
                stderr[key] = (data[key]-avg[key])**2
            else:
                stderr[key] += (data[key]-avg[key])**2

    if len(data_list)>1:
        for key in data.keys():
            stderr[key] /= len(data_list)-1
            stderr[key] = np.sqrt(stderr[key])
            stderr[key] /= np.sqrt(len(data_list))

    return stderr

def get_trajectory_stats(data_list):

    """
    Compute average and standard error of data over trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing average and standard error of data
    """

    avg = get_trajectory_avg(data_list)
    stderr = get_trajectory_stderr(data_list)
    keys = avg.keys()
    stats = {}
    for key in keys:
        stats[key + '_avg'] = avg[key]
        stats[key + '_stderr'] = stderr[key]
    stats['nsample'] = len(data_list)

    return stats

if __name__ == '__main__':
    main()