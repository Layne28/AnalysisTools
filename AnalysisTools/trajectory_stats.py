#Compute statistics over trajectories

import numpy as np
import h5py
import sys
import os
import argparse
import copy

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools
import AnalysisTools.histogram as hist_tools
import AnalysisTools.cluster as cluster

def main():

    parser = argparse.ArgumentParser(description='Compute statistics over trajectories (either avg+stderr OR histogram OR CSD).')
    parser.add_argument('base_folder', help='Directory within which to look for trajectories w/ different seeds.')
    parser.add_argument('quantity', help='e.g. velocity, msd, ...')
    parser.add_argument('stats_type', help='"average" or "histogram" or "csd"')
    parser.add_argument('traj_type', help='"particle" or "noise" or postprocessed. If "postprocessed," will look for "quantity.npz"; else will look for "quantity" dataset in trajectory file')
    parser.add_argument('--subfolder', default='prod', help='Within each seed folder, look in this subfolder.')
    parser.add_argument('--max_num_traj', default=1000, help='Max number of trajectories to load and analyze.')
    parser.add_argument('--nchunks', default=5, help='Number of chunks to divide trajectories into (if applicable).')
    parser.add_argument('--rc', default=1.5, help='Cluster size cutoff, only needed if stats_type=csd.')

    args = parser.parse_args()

    ### Load data ####
    basefolder = args.base_folder
    subfolder = args.subfolder
    dataset=None
    if args.traj_type=='postprocessed':
        if args.quantity=='csd':
            filename = args.quantity + '_rc=%f.npz' % float(args.rc)
        else:
            filename = args.quantity + '.npz'
    elif args.traj_type=='noise':
        filename = 'noise_traj.h5'
        dataset = args.quantity
    else:
        filename = 'traj.h5'
        dataset = args.quantity

    #Full trajectories are too big to load all at once -- deal with them separately
    if args.traj_type=='postprocessed':
        print('reading postprocessed data...')
        data = get_trajectory_data(basefolder, filename, dataset=dataset, subfolder=subfolder, max_num_traj=int(args.max_num_traj))
        if args.stats_type=='average':
            mystats = get_postprocessed_stats(data)
            np.savez(basefolder + '/' + args.quantity + '_avg.npz', **mystats)
        elif args.stats_type=='histogram':
            myhisto = get_postprocessed_histogram(data, args.quantity)
            np.savez(basefolder + '/' + args.quantity + '_histo.npz', **myhisto)
        elif args.stats_type=='csd':
            print('getting csd')
            mycsd = get_postprocessed_csd(data)
            np.savez(basefolder + 'csd_rc=%f.npz' % float(args.rc), **mycsd)
    
    else:
        if args.stats_type=='average':
            print('getting statistics...')
            mystats = get_trajectory_stats(basefolder, filename, dataset=dataset, subfolder=subfolder)
            np.savez(basefolder + '/' + args.quantity + '_avg.npz', **mystats)
        elif args.stats_type=='histogram':
            myhisto = get_trajectory_histogram(basefolder, filename, dataset=dataset, subfolder=subfolder)
            np.savez(basefolder + '/' + args.quantity + '_histo.npz', **myhisto)

def get_trajectory_data(basefolder, filename, dataset=None, subfolder='prod', max_num_traj=1000):

    """
    Retrieve data for different random seeds.
    WARNING: if trajectories are large then this may require
    a very large amount of memory.

    INPUT: Name of folder containing subfolders named "seed=*"
           Name of data file (npz or h5md format)
           Name of dataset (only required for h5md)
           Name of subfolder within each "seed=*" folder to look in
    OUTPUT: List of dictionaries containing data for each trajectory.
    """

    #Check that basefolder contains "seed=*"
    dirs = [d for d in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, d)) and 'seed=' in d]
    print(dirs)
    def seedIndex(val):
        return int((val.split('='))[-1])
    dirs.sort(key=seedIndex)
    #print(dirs)
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
                print(dir)
                data = dict(np.load(thefile, allow_pickle=True))
            elif filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
                data = {dataset: traj[dataset]}
            else:
                traj = io.load_traj(thefile)
                data = {dataset: traj[dataset][:,:,:traj['dim']]}
            data_list.append(data)
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping directory', d)
    if len(data_list)==0:
        raise Exception('Error: no trajectory data found! Exiting.')
    
    return data_list

def get_postprocessed_avg(data_list):

    """
    Compute average of data over postprocessed trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing trajectory-averaged data
    """
    
    avg = {}
    thekeys = [str(k) for k in data_list[0].keys()]
    for k in thekeys:
        avg[k] = None
    #avg = copy.deepcopy(data_list[0])
    #for key in avg.keys():
    #    avg[key] = None
    #avg = {k: None for k in data_list[0].keys()}
    for data in data_list:
        for key in avg.keys():
            try:
                #print(key)
                #print(avg[key])
                if avg[key] is None:
                    avg[key] = copy.deepcopy(data[str(key)])
                else:
                    avg[key] += data[str(key)]
            except:
                print('Warning: key %s not found!' % str(key))
                continue
    for key in avg.keys():
        avg[key] = avg[key]/len(data_list)

    return avg

def get_postprocessed_stderr(data_list):

    """
    Compute standard error of data over postprocessed trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing standard error of data
    """

    avg = get_postprocessed_avg(data_list)
    #print(avg)
    stderr = {}
    thekeys = [str(k) for k in data_list[0].keys()]
    for k in thekeys:
        stderr[k] = None

    for data in data_list:
        for key in stderr.keys():
            try:
                if stderr[key] is None:
                    stderr[key] = (copy.deepcopy(data[key])-copy.deepcopy(avg[key]))**2
                else:
                    stderr[key] += (data[key]-avg[key])**2
            except:
                print('Warning: key %s not found!' % key)
                continue

    if len(data_list)>1:
        for key in data.keys():
            stderr[key] /= len(data_list)-1
            stderr[key] = np.sqrt(stderr[key])
            stderr[key] /= np.sqrt(len(data_list))

    return stderr

def get_postprocessed_stats(data_list):

    """
    Compute average and standard error of data over postprocessed trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing average and standard error of data
    """

    avg = get_postprocessed_avg(data_list)
    stderr = get_postprocessed_stderr(data_list)
    keys = avg.keys()
    stats = {}
    for key in keys:
        stats[key + '_avg'] = avg[key]
        stats[key + '_stderr'] = stderr[key]
    stats['nsample'] = len(data_list)

    return stats

def get_postprocessed_histogram(data_list, dataset):

    """
    Compute histogram of data over postprocessed trajectories

    INPUT: List of dictionaries containing data for each trajectory
    OUTPUT: Dictionary containing histogram w/ data from all trajectories
    """
    
    all_data = np.array([])
    for data in data_list:
        all_data = np.append(all_data, data[dataset])
    myhisto = hist_tools.get_histogram(all_data,nskip=0,nchunks=1)

    return myhisto

##################
#Functions dealing with statistics of full trajectories
##################

def get_trajectory_stats(basefolder, filename, dataset=None, subfolder='prod'):

    """
    Compute average and standard error of data over trajectories

    INPUT: Name of folder containing subfolders named "seed=*"
           Name of data file (hdf5 format)
           Name of dataset
           Name of subfolder within each "seed=*" folder to look in
    OUTPUT: Dictionary containing average and standard error of data
    """

    #Check that basefolder contains "seed=*"
    dirs = [d for d in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, d)) and 'seed=' in d]
    if len(dirs)==0:
        raise Exception('Error: base folder does not contain seed subdirectories! Exiting.')
    
    if dataset==None:
        raise Exception('Error: need to specify dataset to retrieve from h5 file. Exiting.')

    if not filename.endswith('.h5'):
        raise Exception('Error: file must be in hdf5 format. Exiting.')
    
    #First compute mean
    ntraj = 0
    avg = 0.0
    for d in dirs:
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                thefile = dir + '/' + filename
            else:
                thefile = dir + '/' + subfolder + '/' + filename
            if filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
                avg += np.mean(traj[dataset])
                ntraj += 1
            else:
                traj = io.load_traj(thefile)
                data = traj[dataset][:,:,:traj['dim']] #this will only work right now for per-particle quantities
                avg += np.mean(data)
                ntraj += 1

            traj.clear() #clear memory
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping...')
    avg = avg/ntraj
    print('Computed mean.')

    #Then compute stderr, kurtosis
    q2 = 0.0 #variance
    q4 = 0.0 #4th central moment
    for d in dirs:
        print(d)
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                thefile = dir + '/' + filename
            else:
                thefile = dir + '/' + subfolder + '/' + filename
            if filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
                q2 += np.mean((traj[dataset]-avg)**2)
                q4 += np.mean(((traj[dataset]-avg)**2)**2) #have to compute fourth power this funny way to avoid oom error
                print(q2, q4)
            else:
                traj = io.load_traj(thefile)
                data = traj[dataset][:,:,:traj['dim']] #this will only work right now for per-particle quantities
                q2 += np.mean((data-avg)**2)
                q4 += np.mean(((data-avg)**2)**2)

            traj.clear() #clear memory
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping...')
    q2 = q2/ntraj
    q4 = q4/ntraj
    stderr = np.sqrt(q2/ntraj)
    kurtosis = q4/q2**2

    stats = {}
    stats['avg'] = avg
    stats['stderr'] = stderr
    stats['variance'] = q2
    stats['kurtosis'] = kurtosis
    stats['nsample'] = ntraj

    for key in stats.keys():
        print(key, stats[key])

    return stats

def get_trajectory_histogram(basefolder, filename, dataset=None, subfolder='prod', nbins=50, nchunks=5):

    """
    Compute histogram of data over trajectories

    INPUT: Name of folder containing subfolders named "seed=*"
           Name of data file (hdf5 format)
           Optional: Name of dataset
           Optional: Name of subfolder within each "seed=*" folder to look in
           Optional: Number of bins
    OUTPUT: Dictionary containing histogram of data
    """

    #Check that basefolder contains "seed=*"
    dirs = [d for d in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, d)) and 'seed=' in d]
    if len(dirs)==0:
        raise Exception('Error: base folder does not contain seed subdirectories! Exiting.')
    
    if dataset==None:
        raise Exception('Error: need to specify dataset to retrieve from h5 file. Exiting.')

    if not filename.endswith('.h5'):
        raise Exception('Error: file must be in hdf5 format. Exiting.')
    
    #First compute upper and lower histogram limits
    llim = 0.0
    ulim = 0.0
    for d in dirs:
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                thefile = dir + '/' + filename
            else:
                thefile = dir + '/' + subfolder + '/' + filename
            if filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
            else:
                traj = io.load_traj(thefile)
                data = traj[dataset][:,:,:traj['dim']] #this will only work right now for per-particle quantities
            if filename=='noise_traj.h5':
                min_curr = np.min(traj[dataset])
                max_curr = np.max(traj[dataset])
            else:
                min_curr = np.min(data)
                max_curr = np.max(data)
            if min_curr < llim:
                llim = min_curr
            if max_curr > ulim:
                ulim = max_curr

            traj.clear() #clear memory
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping...')
    print('Computed histogram bounds.')

    my_bin_edges = np.linspace(llim,ulim,num=nbins+1)
    all_counts = np.zeros(nbins)
    print(all_counts.shape)
    total_num = 0

    #Then compute histogram
    for d in dirs:
        print(d)
        dir = os.path.join(basefolder, d)
        try:
            if subfolder=='':
                thefile = dir + '/' + filename
            else:
                thefile = dir + '/' + subfolder + '/' + filename
            if filename=='noise_traj.h5':
                traj = io.load_noise_traj(thefile)
            else:
                traj = io.load_traj(thefile)
                data = traj[dataset][:,:,:traj['dim']] #this will only work right now for per-particle quantities
            
            #Compute histogram
            if filename=='noise_traj.h5':
                counts, bin_edges = np.histogram(traj[dataset], bins=my_bin_edges)
                total_num += traj[dataset].size
            else:
                counts, bin_edges = np.histogram(data, bins=my_bin_edges, density=False)
                total_num += data.size
            print(counts.shape)
            all_counts += counts
            
            traj.clear() #clear memory
        except FileNotFoundError:
            print('Warning: trajectory data file not found. Skipping...')

        bins = (bin_edges[:-1]+bin_edges[1:])/2
        hist = all_counts/(1.0*total_num)
            
    the_dict = {}
    the_dict['bins'] = bins
    the_dict['hist'] = hist

    print(bins)
    print(hist)

    return the_dict

def get_postprocessed_csd(data_list, nlast=3):

    """
    Compute CSD over trajectories

    INPUT: List of dictionaries containing csd data for each trajectory
    OUTPUT: Dictionary containing CSD for different chunks over all trajectories
    """

    nchunks = data_list[0]['nchunks']
    the_dict = {}
    the_dict['bins'] = data_list[0]['bins_0']
    the_dict['nchunks'] = nchunks
    the_dict['hist_nlast'] = None
    for n in range(nchunks):
        the_dict['hist_%d' % n] = None
        for t in range(len(data_list)):
            hist = data_list[t]['hist_%d' % n]
            #print(hist)
            
            if the_dict['hist_%d' % n] is None:
                the_dict['hist_%d' % n] = hist
            else:
                the_dict['hist_%d' % n] += hist

            if n>=(nchunks-nlast):
                if the_dict['hist_nlast'] is None:
                    the_dict['hist_nlast'] = hist
                else:
                    the_dict['hist_nlast'] += hist
        #normalize counts to probability
        the_dict['hist_%d' % n] = the_dict['hist_%d' % n]/np.sum(the_dict['hist_%d' % n])
        #print(the_dict['hist_%d' % n])
    the_dict['hist_nlast'] = the_dict['hist_nlast']/np.sum(the_dict['hist_nlast'])
    return the_dict

if __name__ == '__main__':
    main()
