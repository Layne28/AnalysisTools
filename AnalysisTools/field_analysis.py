#Compute properties of (active noise) field

import numpy as np
import h5py
import sys
import numba

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools

def main():

    ### Load data ####
    myfile = sys.argv[1] #Expects .h5 input file
    traj = io.load_noise_traj(myfile) #Extract data
    print(traj['noise'].shape[:-1])
    div = np.zeros(traj['noise'].shape[:-1])
    vor = np.zeros(traj['noise'].shape[:-1])
    for t in range(traj['times'].shape[0]):
        div[t,:,:] = get_divergence(traj['dim'], traj['ncells'], traj['spacing'], traj['noise'][t,:,:,:])
        #print(div)
        vor[t,:,:] = get_curl(traj['dim'], traj['ncells'], traj['spacing'], traj['noise'][t,:,:,:])
    print(np.mean(np.abs(traj['noise'][0,:,:,:])))
    print(np.mean(np.abs(div)))
    print(np.median(np.abs(div)))
    print(np.max(div))
    print(np.min(div))
    #### Output field properties to file in same directory as input h5 file ####        
    outfile = '/'.join((myfile.split('/'))[:-1]) + '/field_properties.npz'
    np.savez(outfile, times=traj['times'], divergence=div, vorticity=vor)

def get_divergence(dim, ncells, spacing, field):
    """
    Compute divergence of a vector field
    
    INPUT: Vector field (d+1-dimensional numpy array, with
           first d dimensions corresponding to grid cells,
           last dimension having length d (components of field)
    OUTPUT: Divergence field (d-dimensional numpy array)
    """
    #dim = len(field.shape)-1

    if dim==2:
        div = get_2d_divergence(ncells, spacing, field)
    elif dim==3:
        div = get_3d_divergence(ncells, spacing, field)
    else:
        print('Error: can only compute divergence in 2 or 3 dimensions.')
        exit()
    return div

@numba.jit(nopython=True)
def get_2d_divergence(ncells, spacing, field):
    div = np.zeros((field.shape[0], ncells[0],ncells[1]), dtype=np.float64)
    for t in range(field.shape[0]):
        for i in range(ncells[0]):
            for j in range(ncells[1]):
                #Use central difference formula to compute derivatives
                fxp = field[t][(i+1)%ncells[0]][j][0]
                fxm = field[t][(i-1+ncells[0])%ncells[0]][j][0]
                term1 = (fxp-fxm)/(2.0*spacing[0])

                fyp = field[t][i][(j+1)%ncells[1]][1]
                fym = field[t][i][(j-1+ncells[1])%ncells[1]][1]
                term2 = (fyp-fym)/(2.0*spacing[1])

                div[t][i][j] = term1 + term2
    return div

@numba.jit(nopython=True)
def get_3d_divergence(ncells, spacing, field):
    div = np.zeros((ncells[0],ncells[1],ncells[2]))
    for i in range(ncells[0]):
        for j in range(ncells[1]):
            for k in range(ncells[2]):
                #Use central difference formula to compute derivatives
                fxp = field[(i+1)%ncells[0]][j][k][0]
                fxm = field[(i-1+ncells[0])%ncells[0]][j][k][0]
                term1 = (fxp-fxm)/(2*spacing[0])

                fyp = field[i][(j+1)%ncells[1]][k][1]
                fym = field[i][(j-1+ncells[1])%ncells[1]][k][1]
                term2 = (fyp-fym)/(2*spacing[1])

                fzp = field[i][j][(k+1)%ncells[2]][2]
                fzm = field[i][j][(k-1+ncells[2])%ncells[2]][2]
                term3 = (fzp-fzm)/(2*spacing[2])

                div[i][j] = term1 + term2 + term3
    return div

def get_curl(dim, ncells, spacing, field):
    """
    Compute curl of a vector field
    
    INPUT: Vector field (d+1-dimensional numpy array, with
           first d dimensions corresponding to grid cells,
           last dimension having length d (components of field)
    OUTPUT: Curl field (d-dimensional numpy array)
    """
    dim = len(field.shape)-1

    if dim==2:
        curl = get_2d_curl(ncells, spacing, field)
    elif dim==3:
        curl = get_3d_curl(ncells, spacing, field)
    else:
        print('Error: can only compute curl in 2 or 3 dimensions.')
        exit()
    return curl

@numba.jit(nopython=True)
def get_2d_curl(ncells, spacing, field):
    nx = field.shape[0]
    ny = field.shape[1]
    curl = np.zeros((nx,ny))

    for i in range(ncells[0]):
        for j in range(ncells[1]):
            #Use central difference formula to compute derivatives
            fyxp = field[(i+1)%ncells[0]][j][1]
            fyxm = field[(i-1+ncells[0])%ncells[0]][j][1]
            term1 = (fyxp-fyxm)/(2*spacing[0])

            fxyp = field[i][(j+1)%ncells[1]][0]
            fxym = field[i][(j-1+ncells[1])%ncells[1]][0]
            term2 = (fxyp-fxym)/(2*spacing[1])

            curl[i][j] = term1 - term2

    return curl

@numba.jit(nopython=True)
def get_3d_curl(ncells, spacing, field):
    nx = field.shape[0]
    ny = field.shape[1]
    nz = field.shape[2]
    curl = np.zeros((nx,ny,nz))
    return curl

if __name__ == '__main__':
    main()
