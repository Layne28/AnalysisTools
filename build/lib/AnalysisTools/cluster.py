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
import AnalysisTools.measurement_tools as tools
import AnalysisTools.particle_io as io

def cluster_traj(in_file,rc):

    faulthandler.enable()

    if not(in_file.endswith('.h5')):
        print('Error: requires h5md file!')
        exit()

    in_folder = os.path.dirname(in_file)
    #Initialize data containers
    cluster_sizes = [] #number of nodes in clusters
    cluster_areas = [] #number of surface-exposed nodes in clusters

    #Load trajectory
    traj = io.load_traj(in_file)
    traj_length = traj['times'].shape[0]
    num_clusters = np.zeros(traj_length)

    #Initialize cell list
    print("Initializing cell list parameters...")
    #ncell_x, ncell_y, cellsize_x, cellsize_y, cell_neigh = init_cell_list(traj['edges'], 2.5)
    ncell_arr, cellsize_arr, cell_neigh = init_cell_list(traj['edges'], 2.5, traj['dim'])
    print("Done.")

    #Create file for dumping clusters
    cluster_file = h5py.File(in_folder + '/clusters_rc=%f.h5' % rc, 'w')

    for t in range(traj_length):
        if t%100==0:
            print('frame ', t)

        #Create cell list for locating pairs of particles
        head, cell_list, cell_index = create_cell_list(traj['pos'][t,:,:], traj['edges'], ncell_arr, cellsize_arr, traj['dim'])

        cluster_id = get_clusters(traj['pos'][t,:,:], traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, rc, traj['dim'])
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
    cluster_size_hist, size_bin_edges = np.histogram(cluster_sizes, np.arange(0,traj['N']+2,1)-0.5, density=True)
    size_bins = (size_bin_edges[:-1]+size_bin_edges[1:])/2

    num_hist, num_bin_edges = np.histogram(num_clusters, np.arange(0,traj['N']+2,1)-0.5, density=True)

    #Write data
    np.savetxt(in_folder + '/cluster_hist_rc=%f.txt' % rc, np.c_[size_bins,cluster_size_hist,num_hist], header='bin size num')

    return cluster_id

##################
#Cell list functions
##################

def init_cell_list(edges, rcut, dim):

    #Err on the side of making cells bigger than necessary
    ncell_arr = np.zeros(dim)
    cellsize_arr = np.zeros(dim)
    for i in range(dim):
        ncell_arr[i] = int(math.floor(edges[i]/rcut))
        cellsize_arr[i] = edges[i]/ncell_arr[i]
    #if ncell_arr[0]!=ncell_arr[1]:
    #    print('Warning: unequal x and y dimensions in cell grid.')

    cellneigh = fill_cellneigh(ncell_arr, dim)

    return ncell_arr, cellsize_arr, cellneigh

#@numba.jit()
def create_cell_list(pos, edges, narr, sarr, dim):

    if dim==1:
        ncells = int(narr[0])
        #print(ncells)
    elif dim==2:
        ncells = int(narr[0]*narr[1])
    elif dim==3:
        ncells = int(narr[0]*narr[1]*narr[2])
    else:
        print('Error: dim is not 1,2, or 3')
        return -1

    N = pos.shape[0]
    head = (-1)*np.ones(ncells)
    cell_list = np.zeros(N)
    cell_index = np.zeros(N)
    
    N = pos.shape[0]
    for i in range(N):
        #print(np.min(pos))
        if dim==1:
            shiftx = pos[i,0]
            if np.min(pos)<0:
                shiftx += edges[0]/2.0
            icell = int(shiftx/sarr[0])
        elif dim==2:
            shiftx = pos[i,0]
            shifty = pos[i,1]
            if np.min(pos)<0:
                shiftx += edges[0]/2.0
                shifty += edges[1]/2.0
            icell = int(shiftx/sarr[0]) + int(shifty/sarr[1])*int(narr[0])
        elif dim==3:
            shiftx = pos[i,0]
            shifty = pos[i,1]
            shiftz = pos[i,2]
            if np.min(pos)<0:
                shiftx += edges[0]/2.0
                shifty += edges[1]/2.0
                shiftz += edges[2]/2.0
            icell = int(shiftx/sarr[0]) + int(shifty/sarr[1])*int(narr[0]) + int(shiftz/sarr[2])*int(narr[0]*narr[1])
        else:
            icell = -1
            
        if icell>=ncells:
            print('WARNING: icell greater than or equal to ncells')#: icell=%d' % icell)
        cell_index[i] = icell
        cell_list[i] = head[icell]
        if cell_list[i]>=N:
            print('ERROR: list[i]>=N')
        head[icell] = i

    return head, cell_list, cell_index

@numba.jit(nopython=True)
def fill_cellneigh(narr, dim):

    if dim==1:
        nx = int(narr[0])
        cellneigh = np.zeros((nx, 4),dtype=numba.int32)
        for ix in range(nx):
            icell = ix
            nneigh = 0
            for i in range(-1,2):
                jx = ix + i
                #Enforce pbc
                if jx<0:
                    jx += nx
                if jx>=nx:
                    jx -= nx
                jcell = jx
                cellneigh[icell][nneigh+1] = jcell
                #print('nn+1: ', nneigh+1)
                nneigh += 1
            cellneigh[icell][0] = nneigh
            if nneigh!=3:
                print('Error: number of neighbors should be 3 including cell itself.')
                
    elif dim==2:
        nx = int(narr[0])
        ny = int(narr[1])
        cellneigh = np.zeros((nx*ny, 10),dtype=numba.int32) #at most 8 neighbor cells in 2D
        for ix in range(nx):
            for iy in range(ny):
                icell = ix + iy*nx
                nneigh = 0
                for i in range(-1,2):
                    jx = ix + i
                    #Enforce pbc
                    if jx<0:
                        jx += nx
                    if jx>=nx:
                        jx -= nx
                    for j in range(-1,2):
                        jy = iy + j
                        #Enforce pbc
                        if jy<0:
                            jy += ny
                        if jy>=ny:
                            jy -= ny
                        jcell = jx + jy*nx
                        cellneigh[icell][nneigh+1] = jcell
                        nneigh += 1
                cellneigh[icell][0] = nneigh
                if nneigh!=9:
                    print('Error: number of neighbors should be 9 including cell itself.')
    elif dim==3:
        nx = int(narr[0])
        ny = int(narr[1])
        nz = int(narr[2])
        cellneigh = np.zeros((nx*ny*nz, 28),dtype=numba.int32) #at most 26 neighbor cells in 3D
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    icell = ix + iy*nx + iz*nx*ny
                    nneigh = 0
                    for i in range(-1,2):
                        jx = ix + i
                        #Enforce pbc
                        if jx<0:
                            jx += nx
                        if jx>=nx:
                            jx -= nx
                        for j in range(-1,2):
                            jy = iy + j
                            #Enforce pbc
                            if jy<0:
                                jy += ny
                            if jy>=ny:
                                jy -= ny
                            for k in range(-1,2):
                                jz = iz + k
                                #Enforce pbc
                                if jz<0:
                                    jz += nz
                                if jz>=nz:
                                    jz -= nz
                                jcell = jx + jy*nx + jz*nx*ny
                                cellneigh[icell][nneigh+1] = jcell
                                nneigh += 1
                    cellneigh[icell][0] = nneigh
                    if nneigh!=27:
                        print('Error: number of neighbors should be 27 including cell itself.')
    else:
        print('Error: dim is not 1,2, or 3')
        cellneigh = np.zeros((1, 1),dtype=numba.int32)

    return cellneigh


##################
#Cluster functions
##################

#@numba.jit(nopython=True)
def get_clusters(pos, edges, head, cell_list, cell_index, cell_neigh, rc, dim):

    #Returns a numpy array specifying the index of the cluster
    #to which each node belongs

    N = pos.shape[0]

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
