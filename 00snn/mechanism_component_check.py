"""Additional fixed-weight missing-component check. Never reselect a unit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mechanism_study import ROOT,ARMS,contexts,branch
from src.network import ContinuousSNN


def main():
    out=ROOT/'results/sequence_mechanism'
    cfg=json.loads((out/'config.json').read_text())
    a=np.load(out/'test_inputs.npz')
    spikes=np.unpackbits(a['packed_spikes'],axis=-1).astype(bool);anchors=a['anchors']
    keep={'only_a':[0],'only_b':[1],'only_c':[2],'missing_a':[1,2],'missing_b':[0,2],'no_events':[]}
    rows=[]
    for seed in cfg['seeds']:
        for arm in ARMS:
            choice=json.loads((out/f'{arm}_{seed}_selection_lock.json').read_text())
            unit=choice['neuron']
            model=ContinuousSNN(cfg,seed).load(out/f'{arm}_{seed}_network.npz')
            cases,ids=contexts(model,spikes,anchors)
            for i,(state,inputs) in enumerate(cases):
                for name,channels in keep.items():
                    events=inputs[0].copy()
                    for ch in [0,1,2]:
                        if ch not in channels:events[:,:,ch]=False
                    m=branch(model,state,model.weight)
                    recorded=np.array([m.minute(v)[0] for v in events])
                    rows.append(dict(seed=seed,arm=arm,session=int(ids[i,0]),anchor=int(ids[i,1]),
                        kind=name,neuron=unit,response=int(recorded[3:,:,unit].sum())))
    pd.DataFrame(rows).to_csv(out/'component_responses.csv',index=False)


if __name__=='__main__':main()
