import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.structure_factor as structure_factor

class TestStructureFactorMethods(unittest.TestCase):

    def test_get_sq(self):

        traj = particle_io.load_traj('./tests/test_data/sc_lattice.h5')
        #Test on q=0 first
        q0 = np.array([0.0,0.0,0.0])
        structure_factor.get_sq(traj['pos'], traj['edges'], q0, 0.0)
        #self.assertEqual(s.split(), ['hello', 'world'])

if __name__ == '__main__':
    unittest.main()
