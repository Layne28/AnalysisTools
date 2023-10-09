#Contains functions for implementing cell lists.

import numpy as np
import numpy.linalg as la
import pylab as plt
import sys
import os
import glob
import h5py
import numba
#from numba import gdb_init
import math
import faulthandler
import AnalysisTools.measurement_tools as tools
import AnalysisTools.particle_io as io

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

@numba.jit(nopython=True)
def create_cell_list(pos, edges, narr, sarr, dim):

    if dim==1:
        ncells = int(narr[0])
        #print(ncells)
    elif dim==2:
        ncells = int(narr[0]*narr[1])
    else:
        ncells = int(narr[0]*narr[1]*narr[2])

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