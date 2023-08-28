import unittest
import numpy as np

import AnalysisTools.particle_io as particle_io
import AnalysisTools.structure_factor as structure_factor

class TestStructureFactorMethods(unittest.TestCase):

    def test_get_sq(self):

        #Load simple cubic lattice 
        traj = particle_io.load_traj('./tests/test_data/sc_lattice.h5')

        #Test on q=0
        q0 = np.array([0.0,0.0,0.0])
        sq0 = structure_factor.get_sq(traj['pos'], traj['edges'], q0, 0.0)
        self.assertEqual(sq0,traj['N']) #S(q=0) = N

        #Test on range of q along x
        dq = np.pi/np.max(traj['edges'])
        qmax = np.pi
        qvals = structure_factor.get_allowed_q(qmax, dq)
        qvals = qvals[qvals[:,1]==0.0]
        qvals = qvals[qvals[:,2]==0.0]
        print(qvals)
        sqvals = structure_factor.get_sq_range(traj['pos'], traj['edges'], qvals, 0.0)

        exact_vals = np.zeros(sqvals.shape[0],dtype=complex)

        for i in range(qvals.shape[0]):
            sum = np.sum(np.exp(1j*qvals[i,0]*np.arange(0,int((traj['edges'])[0]))))
            exact_vals[i] = (traj['edges'][1]*traj['edges'][2]/traj['edges'][0])*sum*np.conjugate(sum)
            self.assertLess(np.abs(sqvals[i]-exact_vals[i]), 1e-10)

if __name__ == '__main__':
    unittest.main()
