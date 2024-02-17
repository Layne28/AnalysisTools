import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.trajectory_stats as stats

class TestTrajectoryStatsMethods(unittest.TestCase):

    def test_get_trajectory_data(self):

        #Check for seed subdirectories
        with self.assertRaises(Exception):
            stats.get_trajectory_data('tests/test_data/dummy_folder', '')

        #Check for trajectory data files
        with self.assertRaises(Exception):
            stats.get_trajectory_data('tests/test_data/seeded_dummy_folder', '')
        
        #Check that data is loaded
        data = stats.get_trajectory_data('tests/test_data/seeded_folder', 'dummy.npz', subfolder='')
        self.assertEqual(len(data),3)

    def test_get_trajectory_avg(self):
        data = stats.get_trajectory_data('tests/test_data/seeded_folder', 'dummy.npz', subfolder='')
        avg = stats.get_trajectory_avg(data)
        self.assertAlmostEqual(avg['data1'][0],40)
        self.assertAlmostEqual(avg['data1'][1],40)
        self.assertAlmostEqual(avg['data2'][0],20)
        self.assertAlmostEqual(avg['data2'][1],20)

    def test_get_trajectory_stderr(self):
        data = stats.get_trajectory_data('tests/test_data/seeded_folder', 'dummy.npz', subfolder='')
        stderr = stats.get_trajectory_stderr(data)
        self.assertAlmostEqual(stderr['data1'][0],20.0/np.sqrt(3.0))
        self.assertAlmostEqual(stderr['data1'][1],20.0/np.sqrt(3.0))
        self.assertAlmostEqual(stderr['data2'][0],10.0/np.sqrt(3.0))
        self.assertAlmostEqual(stderr['data2'][1],10.0/np.sqrt(3.0))
    
    def test_Get_trajectory_stats(self):
        data = stats.get_trajectory_data('tests/test_data/seeded_folder', 'dummy.npz', subfolder='')
        mystats = stats.get_trajectory_stats(data)
        self.assertAlmostEqual(mystats['data1_avg'][0],40)
        self.assertAlmostEqual(mystats['data1_avg'][1],40)
        self.assertAlmostEqual(mystats['data2_avg'][0],20)
        self.assertAlmostEqual(mystats['data2_avg'][1],20)
        self.assertAlmostEqual(mystats['data1_stderr'][0],20.0/np.sqrt(3.0))
        self.assertAlmostEqual(mystats['data1_stderr'][1],20.0/np.sqrt(3.0))
        self.assertAlmostEqual(mystats['data2_stderr'][0],10.0/np.sqrt(3.0))
        self.assertAlmostEqual(mystats['data2_stderr'][1],10.0/np.sqrt(3.0))
        self.assertEqual(mystats['nsample'],3)


if __name__ == '__main__':
    unittest.main()
