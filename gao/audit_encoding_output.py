"""Read-only frozen-model diagnostics; no training, tuning or strategy selection."""
import json
import numpy as np
import pandas as pd
from src.data import ROOT,load
from src.network import GaoNetwork
from src.encoding import encode

def main():
    source=ROOT/'results/baseline';out=ROOT/'results/encoding_output_review';out.mkdir(exist_ok=False)
    cfg=json.loads((source/'config.json').read_text());b,x,_,groups=load();pairs=pd.read_csv(source/'date_splits.csv')
    sig=pd.read_csv(source/'minute_signals.csv');allrows=[];weightrows=[]
    (out/'protocol.json').write_text(json.dumps(dict(
        scope='diagnostic only, no fitting, no future labels in interventions',
        constant='each raw factor fixed at its saved training-day mean; both normalized branches have rate0.13; same bar-key RNG',
        zero='all external spikes zero from segment start; confirms no spontaneous response from rest',
        actual='reproduce original test spike counts exactly',
        limitation='constant-rate control changes rates and counts; tests response specificity, not an isolated timing shuffle'),indent=2),encoding='utf-8')
    for seed in cfg['seeds']:
        for pool in ['random','stdp']:
            for start in range(0,len(pairs),24):
                sub=pairs.iloc[start:start+24];n=GaoNetwork(cfg['network'],seed,len(sub));means=[];scales=[];dates=list(sub.test_date)
                for j,date in enumerate(dates):
                    z=np.load(source/f'models/{date}_{seed}_{pool}.npz');n.w_in[j]=z['w_in'];n.w_out[j]=z['w_out'];means.append(z['mean']);scales.append(z['scale'])
                    weightrows.append(dict(seed=seed,pool=pool,date=date,output_sum=float(z['w_out'].sum()),
                        output_initial_sum=float(z['initial_out'].sum()),fraction_output_ge_point99=float((z['w_out']>=.99).mean())))
                wi=n.w_in.copy();wo=n.w_out.copy()
                for control in ['actual','constant_rate','zero']:
                    for half in [0,1]:
                        ev=[]
                        for j,date in enumerate(dates):
                            ids=groups[date][half];v=x[ids] if control=='actual' else np.broadcast_to(means[j],x[ids].shape)
                            a=encode(v,ids,means[j],scales[j],seed,cfg['network']['steps_per_minute'],cfg['encoding'])
                            if control=='zero':a[:]=False
                            ev.append(a)
                        _,s=n.simulate(np.array(ev))
                        for j,date in enumerate(dates):
                            ids=groups[date][half];counts=s[j].sum(1)
                            if control=='actual':
                                ref=sig[(sig.seed==seed)&(sig.pool==pool)].set_index('bar_id').loc[ids,'output_count'].to_numpy()
                                assert np.array_equal(counts,ref)
                            if control=='zero':assert counts.sum()==0
                            for k,bar in enumerate(ids):allrows.append(dict(seed=seed,pool=pool,date=date,bar_id=int(bar),control=control,
                                output_count=int(counts[k]),event=bool(counts[k]>0)))
                    assert np.array_equal(wi,n.w_in) and np.array_equal(wo,n.w_out)
            print(seed,pool,'frozen controls complete',flush=True)
    d=pd.DataFrame(allrows);d.to_csv(out/'frozen_controls.csv',index=False)
    d.groupby(['seed','pool','control']).agg(event_fraction=('event','mean'),mean_output_count=('output_count','mean')).to_csv(out/'control_summary.csv')
    pd.DataFrame(weightrows).to_csv(out/'weight_scale.csv',index=False)
    stats=[]
    for (seed,pool),g in sig.groupby(['seed','pool']):
        per=g.groupby('date').agg(binary_unique=('event','nunique'),count_unique=('output_count','nunique'),count_var=('output_count','var'))
        stats.append(dict(seed=seed,pool=pool,days=len(per),constant_binary_days=int((per.binary_unique==1).sum()),
            varying_count_days=int((per.count_unique>1).sum()),mean_within_day_count_variance=per.count_var.mean()))
    pd.DataFrame(stats).to_csv(out/'output_compression.csv',index=False)
    actual=d[d.control.eq('actual')].set_index(['seed','pool','bar_id'])
    constant=d[d.control.eq('constant_rate')].set_index(['seed','pool','bar_id'])
    agree=pd.DataFrame(dict(event_equal=actual.event.eq(constant.event),count_equal=actual.output_count.eq(constant.output_count)))
    agree.groupby(['seed','pool']).mean().to_csv(out/'constant_agreement.csv')
    (out/'verification.json').write_text(json.dumps(dict(complete=True,all_actual_counts_match=True,all_weights_frozen=True,all_zero_input_silent=True)),encoding='utf-8')

if __name__=='__main__':main()
