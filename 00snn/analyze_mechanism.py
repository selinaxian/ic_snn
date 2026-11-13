"""Read-only analysis of completed fixed model outputs; no fitting or selection."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from mechanism_study import ROOT,KINDS,alternatives


def main():
    out=ROOT/'results/sequence_mechanism'
    r=pd.read_csv(out/'test_responses.csv');c=pd.read_csv(out/'component_responses.csv')
    base=r[r.intervention=='none'].copy()
    combined=pd.concat([base.drop(columns='intervention'),c],ignore_index=True)
    combined['responded']=combined.response>0
    profile=combined.groupby(['arm','kind']).agg(mean_spikes=('response','mean'),response_rate=('responded','mean'),observations=('response','size')).reset_index()
    profile.to_csv(out/'response_profiles.csv',index=False)
    a=np.load(out/'test_inputs.npz')
    x=np.unpackbits(a['packed_spikes'],axis=-1).astype(bool);anchors=a['anchors']
    coverage=[];baselines=[]
    for sid,starts in enumerate(anchors):
        for start in starts:
            counts=np.array([x[sid,start+t,:,ch].sum() for ch,t in enumerate([0,1,3])])
            coverage.append(dict(session=sid,anchor=int(start),complete=bool((counts>0).all())))
            variants=alternatives(x[sid,start:start+7])
            # Known generating template is a positive diagnostic control, not STDP learning.
            template=np.array([sum(v[t,:,ch].sum() for ch,t in enumerate([0,1,3])) for v in variants])
            total=variants.sum(axis=(1,2,3))
            for j,kind in enumerate(KINDS[1:5],1):
                for name,score in [('known_input_template',template),('total_input_count',total)]:
                    baselines.append(dict(session=sid,anchor=int(start),control=kind,baseline=name,
                        paired_order_accuracy=float(score[0]>score[j])+.5*float(score[0]==score[j])))
    pd.DataFrame(baselines).groupby(['baseline','control']).paired_order_accuracy.mean().reset_index().to_csv(out/'input_controls.csv',index=False)
    complete=base.merge(pd.DataFrame(coverage),on=['session','anchor'])
    complete=complete[complete.complete]
    wide=complete.pivot(index=['seed','arm','session','anchor'],columns='kind',values='response').reset_index()
    rows=[]
    for arm,g in wide.groupby('arm'):
        for k in KINDS[1:5]:
            rows.append(dict(arm=arm,control=k,background_cases=g[['session','anchor']].drop_duplicates().shape[0],
                mean_difference=(g.target-g[k]).mean(),paired_order_accuracy=((g.target>g[k])+.5*(g.target==g[k])).mean()))
    pd.DataFrame(rows).to_csv(out/'all_events_encoded_subset.csv',index=False)
    choice=pd.read_csv(out/'selected_units.csv');w=pd.read_csv(out/'connection_updates.csv')
    picked=w.merge(choice[['seed','arm','neuron','key_channel','control_channel']],on=['seed','arm','neuron'])
    picked['net_change']=picked.final-picked.initial
    picked[picked.channel<3].to_csv(out/'selected_signal_connections.csv',index=False)
    err=np.max(abs(w.final-w.initial-(w.ltp-w.ltd+w.normalization_and_clipping)))
    assert err<1e-9
    audit=json.loads((out/'verification.json').read_text())
    audit.update(connection_decomposition_max_error=float(err),selected_before_test=True,
                 supplemental_component_check_complete=True,complete_encoding_test_cases=len(complete[['session','anchor']].drop_duplicates()),
                 supplemental_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'mechanism_component_check.py',Path(__file__)]})
    (out/'verification.json').write_text(json.dumps(audit,indent=2))
    print(profile.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
