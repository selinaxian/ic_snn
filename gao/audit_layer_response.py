"""Frozen layer-by-layer intervention audit. Never trains or selects signals."""
import json
import numpy as np
import pandas as pd
from src.data import ROOT, load
from src.network import GaoNetwork
from src.encoding import encode


def compare(a, b):
    # First two dimensions: batch, minute; remaining dimensions describe activity.
    aa=a.reshape(*a.shape[:2],-1).astype(float)
    bb=b.reshape(*b.shape[:2],-1).astype(float)
    return dict(equal=(aa==bb).all(-1), mae=np.abs(aa-bb).mean(-1),
                total_absdiff=np.abs(aa.sum(-1)-bb.sum(-1)))


def main():
    source=ROOT/'results/baseline'; out=ROOT/'results/layer_response_audit'
    out.mkdir(exist_ok=False)
    cfg=json.loads((source/'config.json').read_text())
    bars,x,_,groups=load(); pairs=pd.read_csv(source/'date_splits.csv')
    reference=pd.read_csv(source/'minute_signals.csv').set_index(['seed','pool','bar_id'])
    rows=[]; values=[]
    for seed in cfg['seeds']:
        for pool in ['random','stdp']:
            for start in range(0,len(pairs),24):
                sub=pairs.iloc[start:start+24]; dates=list(sub.test_date)
                net=GaoNetwork(cfg['network'],seed,len(sub)); means=[]; scales=[]
                for j,date in enumerate(dates):
                    z=np.load(source/f'models/{date}_{seed}_{pool}.npz')
                    net.w_in[j]=z['w_in'];net.w_out[j]=z['w_out']
                    means.append(z['mean']);scales.append(z['scale'])
                wi=net.w_in.copy();wo=net.w_out.copy()
                for half in [0,1]:
                    results={}
                    for control in ['actual','constant','new_encoding_seed']:
                        events=[]
                        for j,date in enumerate(dates):
                            ids=groups[date][half]
                            v=np.broadcast_to(means[j],x[ids].shape) if control=='constant' else x[ids]
                            encseed=seed+1000003 if control=='new_encoding_seed' else seed
                            events.append(encode(v,ids,means[j],scales[j],encseed,
                                                 cfg['network']['steps_per_minute'],cfg['encoding']))
                        events=np.array(events);h,o=net.simulate(events)
                        hc=h.sum(2);oc=o.sum(2)
                        results[control]=dict(input_spikes=events,hidden_spikes=h,
                                              hidden_counts=hc,output_spikes=o,
                                              output_count=oc[:,:,None],binary=(oc>0)[:,:,None])
                        for j,date in enumerate(dates):
                            ids=groups[date][half]
                            if control=='actual':
                                ref=reference.loc[(seed,pool)].loc[ids]
                                assert np.array_equal(oc[j],ref.output_count)
                                assert np.array_equal(hc[j,:,0].sum(-1),ref.hidden_h1_count)
                                assert np.array_equal(hc[j,:,1].sum(-1),ref.hidden_h2_count)
                            for k,bar in enumerate(ids):
                                values.append(dict(seed=seed,pool=pool,date=date,bar_id=int(bar),half=half,
                                    control=control,input_count=int(events[j,k].sum()),
                                    hidden_count=int(hc[j,k].sum()),hidden_active=int((hc[j,k]>0).sum()),
                                    output_count=int(oc[j,k]),event=int(oc[j,k]>0)))
                    for control in ['constant','new_encoding_seed']:
                        stats={layer:compare(results['actual'][layer],results[control][layer])
                               for layer in results['actual']}
                        for j,date in enumerate(dates):
                            for k,bar in enumerate(groups[date][half]):
                                row=dict(seed=seed,pool=pool,date=date,bar_id=int(bar),control=control)
                                for layer,metrics in stats.items():
                                    for name,arr in metrics.items():row[f'{layer}_{name}']=float(arr[j,k])
                                rows.append(row)
                    assert np.array_equal(wi,net.w_in) and np.array_equal(wo,net.w_out)
            print(seed,pool,'complete',flush=True)
    d=pd.DataFrame(rows);v=pd.DataFrame(values)
    d.to_csv(out/'paired_layer_differences.csv',index=False)
    v.to_csv(out/'minute_layer_signals.csv',index=False)
    d.groupby(['seed','pool','control']).mean(numeric_only=True).drop(columns='bar_id').to_csv(out/'layer_summary.csv')
    v.groupby(['seed','pool','control']).mean(numeric_only=True).drop(columns=['bar_id','half']).to_csv(out/'activity_summary.csv')
    # Descriptive finance check, all historical dates already used for research.
    # No fitting/selection, no new signal claim. Same-half next-minute close return.
    labels=bars[['bar_id','segment_id','close','datetime']].copy()
    labels['return_1_bp']=labels.groupby('segment_id')['close'].shift(-1)/labels.close*10000-10000
    labels['abs_return_1_bp']=labels.return_1_bp.abs()
    labels['clock']=labels.datetime.dt.strftime('%H:%M')
    actual=v[v.control.eq('actual')].merge(labels[['bar_id','return_1_bp','abs_return_1_bp','clock']],on='bar_id').dropna()
    finance=[]
    for (seed,pool),g in actual.groupby(['seed','pool']):
        for signal in ['hidden_count','hidden_active','output_count','event']:
            for label in ['return_1_bp','abs_return_1_bp']:
                for adjust in ['none','date_and_clock']:
                    a=g[signal].astype(float).copy();b=g[label].copy()
                    if adjust!='none':
                        # Balanced minute grid except last minute removed, two-way demeaning.
                        a=a-g.groupby('date')[signal].transform('mean')-g.groupby('clock')[signal].transform('mean')+a.mean()
                        b=b-g.groupby('date')[label].transform('mean')-g.groupby('clock')[label].transform('mean')+b.mean()
                    corr=a.rank().corr(b.rank()) if a.std()>1e-12 and b.std()>1e-12 else np.nan
                    finance.append(dict(seed=seed,pool=pool,signal=signal,label=label,adjust=adjust,
                                        n=len(g),descriptive_rank_correlation=corr))
    pd.DataFrame(finance).to_csv(out/'exploratory_label_associations.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(all_original_output_and_branch_counts_match=True,
        weights_unchanged=True,training=False,model_days=1080,
        notes=['All existing 180 test dates, 3 seeds, random and STDP.',
               'Constant control shares random numbers, changes rates/counts, including session_progress.',
               'New encoding seed changes only test encoding; models fixed, one perturbation per seed.',
               'Finance associations are historical descriptive checks, not out-of-sample evidence.',
               'Count equality is not information-theoretic information loss or financial usefulness.']),indent=2),encoding='utf-8')
    print('Saved',out,flush=True)


if __name__=='__main__':main()
