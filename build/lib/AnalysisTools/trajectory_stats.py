#Compute statistics over trajectories

import numpy as np
import h5py
import sys
import os
import argparse

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools
import AnalysisTools.histogram as hist_tools

def main():

    parser = argparse.ArgumentParser(description='Compute statistics over trajectories (either avg+stderr OR histogram).')
    parser.add_argument('base_folder', help='Directory within which to look for trajectories w/ different seeds.')
    parser.add_argument('quantity', help='e.g. velocity, msd, ...')
    parser.add_argument('stats_type', help='"average" or "histogram"')
    parser.add_argument('traj_type', help='"particle" or "noise" or postprocessed. If "postprocessed," will look for "quantity.npz"; else will look for "quantity" dataset in trajectory file')
    parser.add_argument('--subfolder', default='prod', help='Within each seed folder, look in this subfolder.')
    parser.add_argument('--max_num_traj', default=1000, help='Max number of trajectories to load and analyze.')

    args = parser.parse_args()

    ### Load data ####
    basefolder = args.base_folder
    subfolder = args.subfolder
    dataset=None
    if args.traj_type=='postprocessed':
        filename = args.quantity + '.npz'
    elif args.traj_type=='noise':
        filename = 'noise_traj.h5'
        dataset = args.quantity
    else:
        filename = 'traj.h5'
        dataset = args.quantity

    data = get_trajectory_data(basefolder, filename, dataset=dataset, subfolder=subfolder, max_num_traj=int(args.max_num_traj))
    if args.stats_type=='average':
        mystats = get_trajectory_stats(data)
        np.savez(basefolder + '/' + args.quantity + '_avg.npz', **mystats)
    elif args.stats_type=='histogram':
        myhisto = get_trajectory_histogram(data, args.quantity)
        np.savez(basefolder + '/' + args.quantity + '_histo.npz', **myhisto)

def get_trajectory_data(basefolder, filename, dataset=None, subfolder='prod', max_num_traj=1000):

    """
    Retrieve data for different random seeds.

    INPUT: Name of folder containing subfolders named "seed=*"
           Name of data file (npz or h5md format)
           Name of dataset (only required for h5md)
           Name of subfolder within each "seed=*" folder to look in
    OUTPUT: List of dictionaries containing data for each trajectory.
    """

    #Check that basefolder contains "seed=*"
    dirs = [d for d in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, d)) and 'seed=' in d]
    dirs = dirs[:max_num_traj]
    if len(dirs)==0:
        raise Exception('Error: base folder does not contain seed subdirectories! Exiting.')
    
    if (filename=='traj.h5' or filename=='noise_traj.h5') and dataset==None:
        raise Exception('Error: need to specify dataset to retrieve from h5 file. Exiting.')

    if not (filename.endswith('.h5') or filename.endswith('.npz')):
        raise Exception('Error: file must either be in h5 or npz format. Exiting.')

    data_list = []
    for d in dirs:
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                thefile = dir + '/' + filename
            else:
                thefile = dir + '/' + subfolder + '/' + filename
            if filename.endswith('.npz'):
                data = np.load(thefile)
            elif filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
                data = {dataset: traj[dataset]}
            else:
                traj = io.load_traj(thefile)
                data = {dataset: traj[dataset][:,:,:traj['dim']]}
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

def get_trajectory_histogram(data_list, dataset):

    """
    Compute histogram of data over trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing histogram w/ data from all trajectories
    """
    
    all_data = np.array([])
    for data in data_list:
        all_data = np.append(all_data, data[dataset])
    myhisto = hist_tools.get_histogram(all_data,nskip=0,nchunks=1)

    return myhisto

if __name__ == '__main__':
    main()