import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.msd as msd

class TestMSDMethods(unittest.TestCase):

    def test_get_msd(self):

        #Test on constant-velocity single-particle trajectory
        mylen = 10000
        dt = 0.01
        v0 = 30.0
        times = np.arange(mylen)*dt
        pos = np.zeros((mylen,1,2))
        pos[:,0,1] = times*v0
        mymsd = msd.get_msd(pos, times, 5.0)
        for i in range(mymsd.shape[0]):
            self.assertAlmostEqual(mymsd[i,1],v0**2*(i*dt)**2,places=8) #S(q=0) = N

if __name__ == '__main__':
    unittest.main()
