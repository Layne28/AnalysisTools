import unittest
import numpy as np
import os

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools
import AnalysisTools.cluster as cluster

class TestClusterMethods(unittest.TestCase):

    def test_cluster_1d(self):

        rc=1.6

        name_1d = './tests/test_data/1d_lattice.h5'

        traj = io.load_traj(name_1d)

        ncell_arr, cellsize_arr, cell_neigh = cluster.init_cell_list(traj['edges'], 2.5, traj['dim'])
        head, cell_list, cell_index = cluster.create_cell_list(traj['pos'][0,:,:], traj['edges'], ncell_arr, cellsize_arr, traj['dim'])

        cluster_id_zeros = cluster.get_clusters(traj['pos'][0,:,:], traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, 0.5, traj['dim'], traj['N'])
        cluster_id_ones = cluster.get_clusters(traj['pos'][0,:,:], traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, rc, traj['dim'], traj['N'])

        #Check that CSD has support only for n=1
        hist = cluster.get_csd('./tests/test_data/clusters_rc=0.500000.h5')
        self.assertEqual(hist['avg_hist'][1], 1.0)

        #Check that configuration with no nodes above threshold
        #except self has N clusters
        self.assertEqual(np.max(cluster_id_zeros),traj['N'])
        
        #Check that configuration with all nodes above threshold has only one cluster
        self.assertEqual(np.max(cluster_id_ones),1)
        
        #Check that configuration with all particles in one cluster
        #has size N
        unique, counts = np.unique(cluster_id_ones, return_counts=True)
        self.assertEqual(counts[0],traj['N'])

    def test_cluster_3d(self):

        rc=1.6

        name_3d = './tests/test_data/sc_lattice.h5'

        traj = io.load_traj(name_3d)

        ncell_arr, cellsize_arr, cell_neigh = cluster.init_cell_list(traj['edges'], 2.5, traj['dim'])
        head, cell_list, cell_index = cluster.create_cell_list(traj['pos'][0,:,:], traj['edges'], ncell_arr, cellsize_arr, traj['dim'])

        cluster_id_zeros = cluster.get_clusters(traj['pos'][0,:,:], traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, 0.5, traj['dim'], traj['N'])
        cluster_id_ones = cluster.get_clusters(traj['pos'][0,:,:], traj['edges'][:(traj['dim'])], head, cell_list, cell_index, cell_neigh, rc, traj['dim'], traj['N'])

        #Check that configuration with no nodes above threshold
        #except self has N clusters
        self.assertEqual(np.max(cluster_id_zeros),traj['N'])
        
        #Check that configuration with all nodes above threshold has only one cluster
        self.assertEqual(np.max(cluster_id_ones),1)
        
        #Check that configuration with all particles in one cluster
        #has size N
        unique, counts = np.unique(cluster_id_ones, return_counts=True)
        self.assertEqual(counts[0],traj['N'])

if __name__ == '__main__':
    unittest.main()
