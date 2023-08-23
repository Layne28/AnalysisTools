#This script contains functions for making common measurements
#on trajectory data.

import h5py
import numpy as np
import numba

@numba.jit(nopython=True)
def get_min_disp(r1, r2, edges):
    arr1 = edges/2.0
    arr2 = -edges/2.0
    rdiff = r1-r2
    rdiff = np.where(rdiff>arr1, rdiff-edges, rdiff)
    rdiff = np.where(rdiff<arr2, rdiff+edges, rdiff)
    return rdiff