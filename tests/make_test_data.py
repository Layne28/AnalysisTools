#Create example data for testing.

import h5py
import numpy as np

def make_cubic_lattice():

    #Create simple cubic lattice array
    nx = 10
    lat = np.zeros((1,nx**3,3))
    cnt = 0
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                lat[0,cnt,0] = 1.0*i
                lat[0,cnt,1] = 1.0*j
                lat[0,cnt,2] = 1.0*k
                cnt += 1

    #Create output file
    myfile = h5py.File('./test_data/sc_lattice.h5')
    myfile.create_dataset('/particles/all/position/value', data=lat)
    myfile.create_dataset('/particles/all/position/time', data=np.array([0.0]))
    myfile.create_dataset('/particles/all/position/step', data=np.array([0]))
    myfile.create_dataset('/particles/all/velocity/value', data=np.zeros(lat.shape))
    myfile.create_dataset('/particles/all/box/edges', data=np.array([nx,nx,nx]))

def make_fcc_lattice():
    return 0

if __name__=="__main__":
    make_cubic_lattice()
    make_fcc_lattice()