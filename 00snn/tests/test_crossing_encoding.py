import unittest
import numpy as np
from src.crossing_encoding import encode_session,fit_thresholds

class CrossingTests(unittest.TestCase):
    def test_crossing_hysteresis_and_phase(self):
        x=np.array([.2,1.1,1.2,.9,1.1,.4,1.2])[:,None]
        t={'high':np.array([1.,1.]),'low':np.array([.5,.5])}
        e=encode_session(x,t)
        self.assertEqual(np.flatnonzero(e[:,0,0]).tolist(),[1,6])
        self.assertFalse(e[:,1:].any())
        self.assertTrue(np.array_equal(e[:4],encode_session(x[:4],t)))
    def test_session_boundary_and_negative(self):
        t={'high':np.array([1.,1.]),'low':np.array([.5,.5])}
        e=encode_session(np.array([1.2,1.3,.2,-1.2])[:,None],t)
        self.assertEqual(np.argwhere(e).tolist(),[[3,0,1]])
    def test_disabled_channel(self):
        x=np.zeros((30,1));t=fit_thresholds(x)
        self.assertTrue(np.isinf(t['high']).all())
        self.assertFalse(encode_session(x,t).any())

if __name__=='__main__':unittest.main()
