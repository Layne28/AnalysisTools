import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.correlation as correlation

class TestCorrelationMethods(unittest.TestCase):

    def test_get_single_particle_time_corr(self):

        #Load simple harmonic oscillator
        traj = particle_io.load_traj('./tests/test_data/sho.h5')

        

if __name__ == '__main__':
    unittest.main()
