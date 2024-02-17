#Create example data for testing.

import h5py
import numpy as np
import os

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
    myfile = h5py.File('tests/test_data/sc_lattice.h5', 'w')
    if 'particles' in myfile:
        del myfile['/particles']
    myfile.create_dataset('/particles/all/position/value', data=lat)
    myfile.create_dataset('/particles/all/position/time', data=np.array([0.0]))
    myfile.create_dataset('/particles/all/position/step', data=np.array([0]))
    myfile.create_dataset('/particles/all/velocity/value', data=np.zeros(lat.shape))
    myfile.create_dataset('/particles/all/box/edges', data=np.array([nx,nx,nx]))

def make_fcc_lattice():
    return 0

def make_1d_lattice():

    #Create 1d lattice array
    nx = 100
    lat = np.zeros((1,nx,1))
    for i in range(nx):
        lat[0,i,0] = 1.0*i

    #Create output file
    myfile = h5py.File('tests/test_data/1d_lattice.h5', 'w')
    if '/particles' in myfile:
        del myfile['/particles']
    myfile.create_dataset('/particles/all/position/value', data=lat)
    myfile.create_dataset('/particles/all/position/time', data=np.array([0.0]))
    myfile.create_dataset('/particles/all/position/step', data=np.array([0]))
    myfile.create_dataset('/particles/all/velocity/value', data=np.zeros(lat.shape))
    myfile.create_dataset('/particles/all/box/edges', data=np.array([nx,0,0]))

def make_traj_data():

    mypath = 'tests/test_data/seeded_folder'
    if not os.path.exists(mypath):
        os.makedirs(mypath)
        os.makedirs(mypath + '/seed=1')
        os.makedirs(mypath + '/seed=2')
        os.makedirs(mypath + '/seed=3')
    
    for i in range(3):
        data1 = np.array([20.0*(i+1),20.0*(i+1)])
        data2 = np.array([10.0*(i+1),10.0*(i+1)])
        np.savez(mypath+'/seed=%d/dummy.npz' % (i+1), data1=data1, data2=data2)

if __name__=="__main__":
    make_cubic_lattice()
    make_fcc_lattice()
    make_1d_lattice()
    make_traj_data()