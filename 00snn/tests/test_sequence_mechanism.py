import json
import unittest
import numpy as np
from mechanism_study import make_data,alternatives,shuffled,train,ROOT,branch,copy_state
from src.network import ContinuousSNN


class SequenceChecks(unittest.TestCase):
    def setUp(self):self.cfg=json.loads((ROOT/'config.json').read_text())

    def test_counts_and_same_last(self):
        x,anchors=make_data(2,10101,self.cfg)
        for s in range(2):
            for t in anchors[s]:
                a=alternatives(x[s,t:t+7])
                np.testing.assert_array_equal(a[0,3],a[2,3])
                np.testing.assert_array_equal(a[:5].sum(axis=(1,2)),np.repeat(a[0].sum(axis=(0,1))[None],5,axis=0))
        y=shuffled(x,anchors)
        np.testing.assert_array_equal(x.sum(axis=(1,2)),y.sum(axis=(1,2)))

    def test_audit_does_not_change_learning(self):
        cfg=dict(self.cfg,epochs=1)
        x,_=make_data(1,10101,cfg)
        a,b=ContinuousSNN(cfg,104),ContinuousSNN(cfg,104)
        audit=train(a,x,True)
        for pre in x[0].reshape(-1,24):b.step(pre,learn=True,adapt=True)
        np.testing.assert_array_equal(a.weight,b.weight)
        np.testing.assert_array_equal(a.theta,b.theta)
        self.assertLess(float(audit['reconstruction_error']),1e-9)

    def test_branch_same_history_and_no_parent_mutation(self):
        m=ContinuousSNN(self.cfg,104)
        for _ in range(12):m.step(np.ones(24))
        state=copy_state(m);w=m.weight.copy()
        a,b=branch(m,state,w.copy()),branch(m,state,w.copy())
        for _ in range(12):np.testing.assert_array_equal(a.step(np.ones(24)),b.step(np.ones(24)))
        for before,after in zip(state,copy_state(m)):np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(w,m.weight)


if __name__=='__main__':unittest.main()
