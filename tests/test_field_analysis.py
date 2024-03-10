import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.field_analysis as analysis

class TestMSDMethods(unittest.TestCase):

    def test_get_div(self):

        #Test on constant-velocity field
        nx = 50
        ny = 50
        field = np.zeros((nx,ny,2))
        for i in range(nx):
            for j in range(ny):
                field[i][j][0] = 3.0
                field[i][j][1] = 2.0
        div = analysis.get_divergence(2, np.array([nx,ny]), np.array([1.0,1.0]), field)
        for i in range(nx):
            for j in range(ny):
                self.assertAlmostEqual(div[i][j],0,places=8)

        #Test on solenoidal field
        nx = 50
        ny = 50
        field = np.zeros((nx,ny,2))
        for i in range(nx):
            for j in range(ny):
                field[i][j][0] = j-nx*1.0/2 
                field[i][j][1] = -i*ny*1.0/2
        div = analysis.get_divergence(2, np.array([nx,ny]), np.array([1.0,1.0]), field)
        print(div)
        for i in range(nx):
            for j in range(ny):
                self.assertAlmostEqual(div[i][j],0,places=8)

if __name__ == '__main__':
    unittest.main()
