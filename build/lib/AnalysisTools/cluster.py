#Perform clustering to identify groups of particles
#in close physical contact.

import numpy as np
import numpy.linalg as la
import pylab as plt
import sys
import os
import glob
import h5py
import pandas as pd
import numba
#from numba import gdb_init
import math
import faulthandler
import argparse
import AnalysisTools.measurement_tools as tools
import AnalysisTools.particle_io as io
import AnalysisTools.cell_list as cl
import AnalysisTools.histogram as histo

def main():

    sys.setrecursionlimit(25000) #allow more recursion

    parser = argparse.ArgumentParser(description='Cluster particles in trajectory and compute CSD.')
    parser.add_argument('file', help='Trajectory file (h5md format).')
    parser.add_argument('--rc', default=1.5, help='Cutoff distance for defining cluster.')
    parser.add_argument('--nchunks', default=5, help='Divide trajectory into this many chunks.')

    args = parser.parse_args()
    myfile = args.file
    rc = float(args.rc)
    nchunks = args.nchunks

    ### Load data ####
    traj = io.load_traj(myfile) #Extract data

    out_folder = '/'.join((myfile.split('/'))[:-1])

    #Do clustering if files don't already exist
    #if not(os.path.isfile(out_folder + '/clusters_rc=%f.h5' % rc)):
    cluster_traj(traj,out_folder,rc)
    #cluster_traj_final(traj,out_folder,rc)

    #Get CSD
    csd = get_csd(out_folder + '/clusters_rc=%f.h5' % rc, nchunks=nchunks, nskip=0)

    np.savez(out_folder + '/csd_rc=%f.npz' % rc, **csd)

##################
#Get Cluster Size Distribution (CSD) from cluster-labeled trajectory
##################

def get_csd(cluster_traj_name, nchunks=5, nskip=0):

    #Read in data
    clusters = h5py.File(cluster_traj_name, 'r')
    times = np.array(clusters['data/time'])
    cluster_ids = np.array(clusters['/data/cluster_ids'])
    clusters.close()

    traj_length = times.shape[0]
    N = cluster_ids.shape[1]
    num_clusters = np.zeros(traj_length)
    cluster_sizes = []

    #Count no. of clusters of different sizes
    for t in range(traj_length):

        cluster_id = cluster_ids[t,:]
        num_clusters[t] = np.max(cluster_id)
        unique, counts = np.unique(cluster_id, return_counts=True)
        cluster_sizes.append(counts)

    #Divide cluster sizes into chunks
    the_dict = {}
    the_dict['nchunks'] = nchunks
    seglen = traj_length//nchunks

    if seglen>0:
        for n in range(nchunks):
            
            cluster_chunk = np.concatenate(cluster_sizes[(n*seglen):((n+1)*seglen)]).ravel()
            hist, bin_edges = np.histogram(cluster_chunk, np.arange(0,N+2,1)-0.5, density=True)
            bins = (bin_edges[:-1]+bin_edges[1:])/2

            the_dict['bins_%d' % n] = bins
            the_dict['hist_%d' % n] = hist
    else:
        the_dict['nchunks'] = 1
        cluster_chunk = np.concatenate(cluster_sizes).ravel()
        hist, bin_edges = np.histogram(cluster_chunk, np.arange(0,N+2,1)-0.5, density=False)
        bins = (bin_edges[:-1]+bin_edges[1:])/2

        the_dict['bins_0'] = bins
        the_dict['hist_0'] = hist

    the_dict['avg_hist'] = histo.get_hist_avg(the_dict, nskip=nskip)
    the_dict['bins'] = bins
    the_dict['stddev_hist'] = histo.get_hist_stddev(the_dict, nskip=nskip)
    the_dict['nskipped'] = nskip

    return the_dict

        
##################
#Cluster an input trajectory
##################

#Just cluster final frame
def cluster_traj_final(traj,out_folder,rc):

    faulthandler.enable()

    #Initialize data containers
    cluster_sizes = [] #number of nodes in clusters
    cluster_areas = [] #number of surface-exposed nodes in clusters

    #Get information from trajectory
    traj_length = traj['times'].shape[0]
    num_clusters = np.zeros(traj_length)

    #Initialize cell list
    print("Initializing cell list parameters...")
    #ncell_x, ncell_y, cellsize_x, cellsize_y, cell_neigh = init_cell_list(traj['edges'], 2.5)
    ncell_arr, cellsize_arr, cell_neigh = cl.init_cell_list(traj['edges'], 2.5, traj['dim'])
    print("Done.")

    #Create file for dumping clusters
    cluster_file = h5py.File(out_folder + '/clusters_rc=%f.h5' % rc, 'w')

    #if t%1000==0:
    #    print('frame ', t)

    print(traj['times'][-1], np.max(traj['pos'][-1,:,:]))
    #Create cell list for locating pairs of particles
    pos = tools.apply_pbc(traj['pos'][-1,:,:], traj['edges'])
    #print(np.max(pos), np.min(pos))
    head, cell_list, cell_index = cl.create_cell_list(pos, traj['edges'], ncell_arr, cellsize_arr, traj['dim'])

    cluster_id = get_clusters(pos, traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, rc, traj['dim'], traj['N'])
    cluster_id = sort_clusters(cluster_id)
    num_clusters[-1] = np.max(cluster_id)
    unique, counts = np.unique(cluster_id, return_counts=True)          

    #Save cluster-labeled data to file
    #TODO: change this to modify original traj.h5 file
    cluster_file.create_dataset('/data/time', data=np.array([traj['times'][-1]]), chunks=True, maxshape=(None,))
    cluster_file.create_dataset('/data/cluster_ids', data=np.array([cluster_id]), chunks=True, maxshape=(None,traj['N']))

    cluster_file.close()

def cluster_traj(traj,out_folder,rc):

    faulthandler.enable()

    #Initialize data containers
    cluster_sizes = [] #number of nodes in clusters
    cluster_areas = [] #number of surface-exposed nodes in clusters

    #Get information from trajectory
    traj_length = traj['times'].shape[0]
    num_clusters = np.zeros(traj_length)

    #Initialize cell list
    print("Initializing cell list parameters...")
    #ncell_x, ncell_y, cellsize_x, cellsize_y, cell_neigh = init_cell_list(traj['edges'], 2.5)
    ncell_arr, cellsize_arr, cell_neigh = cl.init_cell_list(traj['edges'], 2.5, traj['dim'])
    print("Done.")

    #Create file for dumping clusters
    cluster_file = h5py.File(out_folder + '/clusters_rc=%f.h5' % rc, 'w')

    for t in range(traj_length):
        #if t%1000==0:
        #    print('frame ', t)

        print(t, np.max(traj['pos'][t,:,:]))
        #Create cell list for locating pairs of particles
        pos = tools.apply_pbc(traj['pos'][t,:,:], traj['edges'])
        #print(np.max(pos), np.min(pos))
        head, cell_list, cell_index = cl.create_cell_list(pos, traj['edges'], ncell_arr, cellsize_arr, traj['dim'])

        cluster_id = get_clusters(pos, traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, rc, traj['dim'], traj['N'])
        cluster_id = sort_clusters(cluster_id)
        num_clusters[t] = np.max(cluster_id)
        unique, counts = np.unique(cluster_id, return_counts=True)          

        #Save cluster-labeled data to file
        #TODO: change this to modify original traj.h5 file
        if t==0:
            cluster_file.create_dataset('/data/time', data=np.array([traj['times'][t]]), chunks=True, maxshape=(None,))
            cluster_file.create_dataset('/data/cluster_ids', data=np.array([cluster_id]), chunks=True, maxshape=(None,traj['N']))
        else:
            cluster_file['/data/time'].resize(cluster_file['/data/time'].shape[0] + 1, axis=0)
            cluster_file['/data/time'][-1] = traj['times'][t]
            cluster_file['/data/cluster_ids'].resize(cluster_file['/data/cluster_ids'].shape[0] + 1, axis=0)
            cluster_file['/data/cluster_ids'][-1] = cluster_id

    cluster_file.close()

    #Get histograms
    #cluster_size_hist, size_bin_edges = np.histogram(cluster_sizes, np.arange(0,traj['N']+2,1)-0.5, density=True)
    #size_bins = (size_bin_edges[:-1]+size_bin_edges[1:])/2

    #num_hist, num_bin_edges = np.histogram(num_clusters, np.arange(0,traj['N']+2,1)-0.5, density=True)

    #Write data
    #np.savetxt(out_folder + '/cluster_hist_rc=%f.txt' % rc, np.c_[size_bins,cluster_size_hist,num_hist], header='bin size num')

def get_avg_root_size(n, p):
    return np.sum(np.sqrt(n)*p)

def get_avg_size(n, p):
    return np.sum(n*p)

def get_avg_mass_weighted_size(n, p):
    return np.sum(n**2*p)/np.sum(n*p)

def get_avg_mass_weighted_root_size(n, p):
    return np.sum(n**(3.0/2.0)*p)/np.sum(n*p)

##################
#Cluster functions
##################

#@numba.jit(nopython=True)
def get_clusters(pos, edges, head, cell_list, cell_index, cell_neigh, rc, dim, N):

    #Returns a numpy array specifying the index of the cluster
    #to which each node belongs

    #cluster_id = np.zeros((N,),dtype=numba.int32)
    cluster_id = np.zeros((N,),dtype=int)

    clusternumber = 0
    for i in range(N):
        if cluster_id[i]==0:
            clusternumber += 1
            cluster_id[i] = clusternumber
            harvest_cluster(clusternumber, i, cluster_id, pos, edges, head, cell_list, cell_index, cell_neigh, rc, dim)

    return cluster_id

#@numba.jit(nopython=True)
def harvest_cluster(clusternumber, ipart, cluster_id, pos, edges, head, cell_list, cell_index, cell_neigh, rc, dim):

    #Note that due to limitations of numba this is restricted to 2d for now
    #pos1 = np.array([pos[ipart,0],pos[ipart,1]]) #pos[ipart,:]
    if dim==1:
        pos1 = np.array([pos[ipart,0]])
    elif dim==2:
        pos1 = np.array([pos[ipart,0], pos[ipart,1]])
    else:
        pos1 = np.array([pos[ipart,0], pos[ipart,1], pos[ipart,2]])

    icell = int(cell_index[ipart])
    #print('nneigh: ', cell_neigh[icell][0])
    for nc in range(cell_neigh[icell][0]):
        #print('icell: ', icell)
        #print('nc: ', nc)
        jcell = int(cell_neigh[icell][nc+1])
        #print('jcell: ', jcell)
        jpart = int(head[jcell])
        while jpart != -1:
            if dim==1:
                pos2 = np.array([pos[jpart,0]])
            elif dim==2:
                pos2 = np.array([pos[jpart,0],pos[jpart,1]])
            else:
                pos2 = np.array([pos[jpart,0],pos[jpart,1],pos[jpart,2]])
            rij = tools.get_min_dist(pos1,pos2,edges)
            if jpart!=ipart and rij<=rc and cluster_id[jpart]==0:
                cluster_id[jpart] = clusternumber
                harvest_cluster(clusternumber, jpart, cluster_id, pos, edges, head, cell_list, cell_index, cell_neigh, rc, dim)
            jpart = int(cell_list[jpart])

def sort_clusters(cluster_id_arr):

    #Re-order cluster ids from largest to smallest
    sizes = np.zeros((np.max(cluster_id_arr)+1,2),dtype=int)
    for i in range(sizes.shape[0]):
        sizes[i,0] = i
        sizes[i,1] = cluster_id_arr[cluster_id_arr==i].shape[0]
    sorted_sizes = sizes[1:]
    sorted_sizes = sorted_sizes[(-sorted_sizes[:,1]).argsort()] #sort in descending order, excluding zero cluster
    size_map = {} #make map from cluster id to size rank
    #print('sizes:', sorted_sizes)
    for i in range(sorted_sizes.shape[0]):
        size_map[sorted_sizes[i,0]] = i+1
    size_map[0] = 0

    #Now rename cluster ids
    cluster_id_sorted = np.array([size_map[a] for a in cluster_id_arr])
    if sizes[1:,1].shape[0]>0 and sizes[1:,1].max() != cluster_id_sorted[cluster_id_sorted==1].shape[0]:
        print('Error in re-labeling clusters! Biggest cluster does not have cluster id = 1.')
        exit()

    return cluster_id_sorted

if __name__ == '__main__':
    main()
