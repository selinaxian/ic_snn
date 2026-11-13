import unittest,json
import numpy as np
import pandas as pd
from src.data import ROOT,factors
from src.encoding import encode,fit_scaler,rates
from src.network import GaoNetwork
from src.strategy import directions,paper_candidates,executable_trades

CFG=json.loads((ROOT/'config.json').read_text())

class PipelineTests(unittest.TestCase):
    def test_encoding_causal_chunked_and_signed_shift(self):
        x=np.arange(120).reshape(10,12)/100;mu,sd=fit_scaler(x[:5]);ids=np.arange(10)
        a=encode(x,ids,mu,sd,104,20,CFG['encoding'])
        self.assertTrue(np.array_equal(a[:7],encode(x[:7],ids[:7],mu,sd,104,20,CFG['encoding'])))
        r=rates(mu[None],mu,sd,CFG['encoding']);self.assertTrue(np.allclose(r,.13))
    def test_batch_single_exact_and_frozen(self):
        rng=np.random.default_rng(3);ev=rng.random((3,8,20,2,12))<.2
        batch=GaoNetwork(CFG['network'],104,3);batch.simulate(ev,True)
        for i in range(3):
            one=GaoNetwork(CFG['network'],104);one.simulate(ev[i:i+1],True)
            self.assertTrue(np.array_equal(one.w_in[0],batch.w_in[i]));self.assertTrue(np.array_equal(one.w_out[0],batch.w_out[i]))
        wi=batch.w_in.copy();wo=batch.w_out.copy();a,b=batch.simulate(ev)
        self.assertTrue(np.array_equal(wi,batch.w_in));self.assertTrue(np.array_equal(wo,batch.w_out))
        c,d=batch.simulate(ev[:,:4]);self.assertTrue(np.array_equal(b[:,:4],d))
    def test_branch_separation_and_all_synapses_plastic(self):
        net=GaoNetwork(CFG['network'],104);pre=np.zeros((1,2,12));pre[:,0]=1
        for _ in range(20):h,o=net.step(pre,True);self.assertFalse(h[:,1].any())
        self.assertTrue(np.array_equal(net.w_in[:,1],net.initial_in[:,1]))
        self.assertGreater(np.linalg.norm(net.w_in-net.initial_in),0)
        self.assertGreater(np.linalg.norm(net.w_out-net.initial_out),0)
    def test_stdp_timing_and_reset(self):
        net=GaoNetwork(CFG['network'],104);net.w_in.fill(.001);net.v[0,0,0]=2
        pre=np.zeros((1,2,12));pre[0,0,0]=1;before=net.w_in.copy();net.step(pre,True)
        self.assertGreater(net.w_in[0,0,0,0],before[0,0,0,0])
        before=net.w_in.copy();net.step(pre,True)
        self.assertLess(net.w_in[0,0,0,0],before[0,0,0,0]);w=net.w_in.copy();net.reset();self.assertTrue(np.array_equal(w,net.w_in))
    def test_strategy_alignment_and_cost_arithmetic(self):
        b=pd.DataFrame(dict(segment_id=[1]*10,close=np.arange(100,110.),open=np.arange(100,110.),trading_date=['2026-01-01']*10,contract_id=['IC']*10))
        rr=paper_candidates(b,[3]);self.assertTrue(all(r['entry_bar']==4 and r['exit_bar']==7 for r in rr))
        r,d=executable_trades(b,[3,4,5,6],CFG)
        self.assertTrue(all(a['entry_bar']>=a['signal_bar']+2 for a in r))
        for a in r:self.assertAlmostEqual(a['net_yuan'],a['gross_yuan']-a['slippage_yuan']-a['fee_yuan'])
        self.assertEqual(directions(np.arange(10.),4),[1,1,1])

if __name__=='__main__':unittest.main()
