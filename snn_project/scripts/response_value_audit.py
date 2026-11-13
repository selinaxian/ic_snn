"""Reuse three frozen NeuCube-style folds and complete 5 x 216 responses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from walk_forward_spike_readout import load_data,FOLDS
from snn_core.snncube import SNNCube
from snn_core.encoding import encode_factor_thresholds
from audit_utils import fit_predict,perturb_measure,finish

def main():
    out=Path('experiments/response_value_audit');out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(window=5,target='abs(log(close[t+1]/close[t]))*10000 within segment',
        folds=FOLDS,network='reuse frozen traces and weights; no retraining',
        readout='ridge MSE penalty 0.01; each block standardized to mean squared norm 1; calibration20 days only; no tuning',
        raw='five steps of six existing factors standardized by training scale; no time-window or encoding changes',
        perturb='reverse last5 encoded minutes; or circularly shift input channel0 by1 minute; 2 fixed endpoints per test session',
        scope='historical diagnostic, no new untouched dates'),indent=2),encoding='utf-8')
    sessions,dayids,_=load_data();results=[];pert=[]
    for fold,(tr,ca,te) in enumerate(FOLDS,1):
        folder=out/f'fold_{fold}';folder.mkdir()
        zs={pool:np.load(f'experiments/familiarity_conditional_audit/fold{fold}_{pool}_frozen_trace.npz') for pool in ['random','stdp']}
        spikes={pool:np.unpackbits(z['packed_spikes'],axis=1)[:,:216] for pool,z in zs.items()}
        raw=[];yy=[];days=[];times=[];features={pool:[] for pool in zs}
        for sid,(t,b,x) in enumerate(sessions):
            day=dayids[t[0].date()]
            if day>te:break
            if day<=tr:continue
            n=len(x);ends=np.arange(4,n);assert n==len(b)-1
            z=zs['stdp'];e=encode_factor_thresholds(x,z['center'],z['scale'])
            raw.append(((x-z['center'])/z['scale'])[ends[:,None]+np.arange(-4,1)].reshape(len(ends),-1))
            y=np.full(len(ends),np.nan);y[:-1]=abs(np.log(b[ends[:-1]+2,3]/b[ends[:-1]+1,3]))*1e4
            yy.extend(y);days.extend([day]*len(ends));times.extend(str(t[i+1]) for i in ends)
            for pool,z in zs.items():
                lo,hi=z['session_offsets'][sid:sid+2];s=spikes[pool][lo:hi];assert len(s)==n
                features[pool].append(s[ends[:,None]+np.arange(-4,1)].reshape(len(ends),-1))
                if day<=ca:continue
                m=SNNCube(11,43,2.5,side=6,depression=1,tau=2)
                for name in ['ports','pre','post','sign']:assert np.array_equal(getattr(m,name),z[name])
                m.weights=z['weights'].copy();before=m.weights.copy()
                assert np.array_equal(m.simulate(e),s)
                for end in [n//2,n-2]:
                    seq=e[:end+1];orig=s[end-4:end+1]
                    for kind in ['reverse_minutes','shift_channel0']:
                        change=seq.copy()
                        if kind=='reverse_minutes':change[-5:]=change[-5:][::-1]
                        else:change[-5:,0]=np.roll(change[-5:,0],1)
                        assert np.array_equal(seq.sum(0),change.sum(0))
                        altered=m.simulate(change)[-5:]
                        pert.append(dict(fold=fold,pool=pool,seed=43,perturbation=kind,timestamp=str(t[end+1]),
                            input_hamming=float((seq[-5:]!=change[-5:]).mean()),**perturb_measure(orig,altered)))
                assert np.array_equal(m.weights,before)
        raw=np.vstack(raw);y=np.array(yy);days=np.array(days);cal=(days<=ca)&np.isfinite(y);test=days>ca
        p=pd.DataFrame(dict(fold=fold,timestamp=np.array(times)[test],actual_abs_1m_bp=y[test]))
        p['date']=p.timestamp.str[:10];p['pred_constant']=y[cal].mean()
        p['pred_raw']=fit_predict([raw],cal,test,y,folder/'raw_readout.npz')
        for pool in features:p['pred_'+pool]=fit_predict([raw,np.vstack(features[pool])],cal,test,y,folder/f'{pool}_readout.npz')
        results.append(p);print('fold',fold,'complete',len(p),flush=True)
    finish(out,pd.concat(results,ignore_index=True),pert)

if __name__=='__main__':main()
