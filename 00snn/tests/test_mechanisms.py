import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.encoding import encode
from src.network import ContinuousSNN
from src.stdp import delta
from src.market_data import load,ROOT

CFG=json.loads((ROOT/'config.json').read_text())


class MechanismTests(unittest.TestCase):
    def test_encoder_chunk_invariant(self):
        x=np.random.default_rng(7).uniform(-1,1,(12,12));ids=np.arange(12)
        full=encode(x,ids,104)
        np.testing.assert_array_equal(full,np.concatenate([encode(x[:5],ids[:5],104),encode(x[5:],ids[5:],104)]))

    def test_reset_preserves_parameters(self):
        m=ContinuousSNN(CFG,104);e=np.ones((8,24),bool)
        m.minute(e,learn=True,adapt=True)
        w,t=m.weight.copy(),m.theta.copy()
        m.reset_state()
        np.testing.assert_array_equal(w,m.weight);np.testing.assert_array_equal(t,m.theta)
        for a in [m.voltage,m.pre_trace,m.post_trace]:self.assertEqual(a.sum(),0)

    def test_next_minute_retains_memory(self):
        m=ContinuousSNN(CFG,104);m.weight[:]=0
        m.weight[0,0]=.9
        e=np.zeros((8,24));e[-1,0]=1
        m.minute(e)
        self.assertGreater(m.voltage[0],0);self.assertGreater(m.pre_trace[0],0)
        z=np.zeros((8,24));z[0,0]=1
        continuation,_=m.minute(z)
        fresh=ContinuousSNN(CFG,104);fresh.weight[:]=m.weight
        isolated,_=fresh.minute(z)
        self.assertEqual(continuation[0,0],1);self.assertEqual(isolated[0,0],0)

    def test_prefix_invariance_and_frozen(self):
        e=encode(np.ones((10,12))*.4,np.arange(10),104)
        a,b=ContinuousSNN(CFG,104),ContinuousSNN(CFG,104)
        before=a.weight.copy();left=np.array([a.minute(v)[0] for v in e[:4]])
        right=np.array([b.minute(v)[0] for v in e])
        np.testing.assert_array_equal(left,right[:4]);np.testing.assert_array_equal(a.weight,before)
        np.testing.assert_array_equal(a.theta,np.zeros(64))

    def test_stdp_local_timing(self):
        w=np.full((1,1),.5)
        self.assertGreater(delta(w,np.ones(1),np.zeros(1),np.zeros(1),np.ones(1),CFG)[0,0],0)
        self.assertLess(delta(w,np.zeros(1),np.ones(1),np.ones(1),np.zeros(1),CFG)[0,0],0)
        self.assertEqual(delta(w,np.zeros(1),np.zeros(1),np.ones(1),np.ones(1),CFG)[0,0],0)

    def test_data_boundaries_and_labels(self):
        b,x,s,y,f=load()
        self.assertEqual(len(b),43440);self.assertEqual(len(s),27150);self.assertEqual(len(f),23)
        self.assertTrue(np.isfinite(y).all())
        for fold in f:
            self.assertLess(max(fold['train_dates']),min(fold['validation_dates']))
            self.assertLess(max(fold['validation_dates']),min(fold['forecast_dates']))


if __name__=='__main__':unittest.main()
