"""Reconstruct unchanged deterministic folds; audit refractory-period recurrence.

Not a new model experiment. Verify saved firing fractions and chosen-neighbor
similarities before interpreting reconstructed traces. Save packed frozen traces.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from walk_forward_spike_readout import load_data,FOLDS
from snn_core.encoding import encode_factor_thresholds
from snn_core.snncube import SNNCube


def main():
    out=Path('experiments/familiarity_conditional_audit')
    original=pd.read_csv('experiments/whole_window_familiarity/whole_window_signals.csv')
    sessions,ids,_=load_data()
    rows,checks=[] ,[]
    for fold,(tr,ca,te) in enumerate(FOLDS,1):
        active=[(ids[t[0].date()],t,b,x) for t,b,x in sessions if ids[t[0].date()]<=te]
        train=np.concatenate([x for day,_,_,x in active if day<=tr])
        center,scale=np.median(train,axis=0),np.maximum(train.std(axis=0),1e-10)
        enc=[(day,t,encode_factor_thresholds(x,center,scale)) for day,t,_,x in active]
        models={name:SNNCube(11,43,2.5,side=6,depression=1,tau=2) for name in ['random','stdp']}
        for day,_,e in enc:
            if day<=tr: models['stdp'].simulate(e,learn=True)
        for name,model in models.items():
            window_times=[]; windows=[]; packed=[]; offsets=[0];spike_times=[]
            for day,t,e in enc:
                s=model.simulate(e)
                packed.append(np.packbits(s,axis=1));offsets.append(offsets[-1]+len(s))
                spike_times.extend(st.isoformat(sep=' ') for st in t[1:])
                w=np.lib.stride_tricks.sliding_window_view(s,5,axis=0).transpose(0,2,1).reshape(-1,1080)
                windows.append(w);window_times.extend(st.isoformat(sep=' ') for st in t[5:])
                if day<=ca: continue
                # Ignore initial 15 simulation steps for stationary-like summary.
                a,b=s[15:],s[12:-3]
                ea,eb=e[15:],e[12:-3]
                # Mechanism control: same first 15 steps, then remove external
                # events. It is not a tradeable input or a return experiment.
                muted=e.copy();muted[15:]=False
                silent=model.simulate(muted)
                sa,sb=silent[60:],silent[57:-3]
                rows.append(dict(fold=fold,pool=name,session_start=t[0].isoformat(sep=' '),
                    firing_fraction_after15=a.mean(),
                    fraction_neurons_rate_ge_point30=(a.mean(axis=0)>=.30).mean(),
                    lag3_spike_repeat=np.logical_and(a,b).sum()/max(b.sum(),1),
                    lag3_spike_jaccard=np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1),
                    input_lag3_jaccard=np.logical_and(ea,eb).sum()/max(np.logical_or(ea,eb).sum(),1),
                    actual_firing_after60=s[60:].mean(),
                    muted_firing_after60=sa.mean(),
                    muted_lag3_repeat_after60=np.logical_and(sa,sb).sum()/max(sb.sum(),1)))
            ws=np.vstack(windows);lookup={t:i for i,t in enumerate(window_times)}
            frame=original[original.fold==fold]
            q=ws[[lookup[t] for t in frame.timestamp]].astype(float)
            r=ws[[lookup[t] for t in frame[name+'_nearest_training_time']]].astype(float)
            density_error=np.max(np.abs(q.mean(axis=1)-frame[name+'_firing_fraction']))
            q-=q.mean(axis=1,keepdims=True);r-=r.mean(axis=1,keepdims=True)
            sim=(q*r).sum(axis=1)/(np.linalg.norm(q,axis=1)*np.linalg.norm(r,axis=1))
            sim_error=np.max(np.abs(sim-frame[name+'_similarity']))
            # Original nearest-neighbor GEMM uses float32 over 1080 entries.
            assert density_error<1e-12 and sim_error<1e-5,(density_error,sim_error)
            checks.append(dict(fold=fold,pool=name,density_max_error=float(density_error),chosen_neighbor_similarity_max_error=float(sim_error)))
            np.savez_compressed(out/f'fold{fold}_{name}_frozen_trace.npz',
                packed_spikes=np.vstack(packed),timestamps=np.array(spike_times),session_offsets=np.array(offsets),
                weights=model.weights,pre=model.pre,post=model.post,sign=model.sign,ports=model.ports,
                center=center,scale=scale,size=np.array(216))
            print('reconstructed',fold,name,'max similarity error',sim_error,flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'session_recurrence.csv',index=False)
    summary=d.groupby(['fold','pool']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'recurrence_summary.csv',index=False)
    (out/'trace_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':main()
