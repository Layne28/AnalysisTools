import unittest
import numpy as np

import AnalysisTools.particle_io as io
import AnalysisTools.measurement_tools as tools
import AnalysisTools.cluster as cluster

class TestClusterMethods(unittest.TestCase):

    def test_cluster(self):

        rc=1.6

        traj = io.load_traj('./tests/test_data/1d_lattice.h5')

        cluster_id_zeros = cluster.cluster_traj('./tests/test_data/1d_lattice.h5', 0.5)
        cluster_id_ones = cluster.cluster_traj('./tests/test_data/1d_lattice.h5', rc)

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
