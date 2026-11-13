"""Recompute test predictions from saved readouts and original frozen traces."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from walk_forward_spike_readout import load_data,FOLDS

def main():
    out=Path('experiments/response_value_audit');p=pd.read_csv(out/'minute_signals.csv');sessions,dayids,_=load_data();errors=[]
    for fold,(_,ca,te) in enumerate(FOLDS,1):
        q=p[p.fold==fold];raw=[];features={pool:[] for pool in ['random','stdp']};times=[];ys=[]
        for pool in features:
            z=np.load(f'experiments/familiarity_conditional_audit/fold{fold}_{pool}_frozen_trace.npz')
            s=np.unpackbits(z['packed_spikes'],axis=1)[:,:216]
            for sid,(t,b,x) in enumerate(sessions):
                day=dayids[t[0].date()]
                if not ca<day<=te:continue
                ends=np.arange(4,len(x));lo,hi=z['session_offsets'][sid:sid+2]
                features[pool].append(s[lo:hi][ends[:,None]+np.arange(-4,1)].reshape(len(ends),-1))
                if pool=='random':
                    raw.append(((x-z['center'])/z['scale'])[ends[:,None]+np.arange(-4,1)].reshape(len(ends),-1))
                    times.extend(str(t[i+1]) for i in ends)
                    y=np.full(len(ends),np.nan);y[:-1]=abs(np.log(b[ends[:-1]+2,3]/b[ends[:-1]+1,3]))*1e4;ys.extend(y)
        assert np.array_equal(times,q.timestamp) and np.allclose(ys,q.actual_abs_1m_bp,equal_nan=True)
        raw=np.vstack(raw)
        for pool in ['raw','random','stdp']:
            blocks=[raw] if pool=='raw' else [raw,np.vstack(features[pool])]
            z=np.load(out/f'fold_{fold}/{pool}_readout.npz');parts=[]
            for j,b in enumerate(blocks):
                a=(b-z[f'mean_{j}'])/z[f'scale_{j}'];a[:,~z[f'keep_{j}']]=0;parts.append(a)
            values=np.maximum(np.concatenate(parts,axis=1)@z['coef']+z['target_mean'],0)
            error=float(np.max(abs(values-q['pred_'+pool])));assert error<1e-10;errors.append(error)
    assert not p.duplicated(['fold','timestamp']).any()
    (out/'verification.json').write_text(json.dumps(dict(complete=True,max_prediction_reconstruction_error=max(errors),
        rows=len(p),labels=int(p.actual_abs_1m_bp.notna().sum()),labels_rebuilt=True,
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('scripts/response_value_audit.py'),Path('scripts/audit_utils.py'),Path('scripts/snn_core/snncube.py'),Path('scripts/snn_core/encoding.py')]}),indent=2),encoding='utf-8')

if __name__=='__main__':main()
