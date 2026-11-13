"""Post-result explanatory control: common input AND baseline learned theta."""
import json
import numpy as np
import pandas as pd
from src.market_data import ROOT,load
from src.network import ContinuousSNN
from run import stream
from encoding_stability_audit import apply_saved,corr
from audit_utils import perturb_measure

def main():
    out=ROOT/'results/encoding_stability_audit'
    (out/'theta_control_protocol.json').write_text(json.dumps(dict(
        status='supplement added after main results; not preregistered primary outcome',
        reason='random weights stayed identical but common-input responses changed: learned theta differs',
        method='use baseline encoding and respective pool baseline theta; keep each trained weight; fixed baseline readout, no refit')))
    cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load();fold=folds[0]
    groups=[]
    for _,g in bars.groupby('segment_id',sort=False):
        ids=g.index.to_numpy();ids=ids[np.isfinite(x[ids]).all(1)]
        if len(ids) and bars.trading_date.iloc[ids[0]] in fold['forecast_dates']:groups.append(ids)
    ids=np.concatenate(groups);ends=np.concatenate([g[11:] for g in groups]);ix=ends[:,None]+np.arange(-11,1)
    scale=np.load(ROOT/'results/first_fold/fold_00/factor_scale.npy');raw=np.clip(x[ix]/scale,-1,1).reshape(len(ends),-1)
    inp=np.load(out/'base_input.npz');ev=np.zeros((len(bars),8,24),bool);ev[inp['bar_ids']]=np.unpackbits(inp['packed'],axis=2)[:,:,:24]
    summary=[];preds=[];channels=[]
    for pool in ['random','stdp']:
        z=np.load(out/f'base_{pool}_network.npz');ref=np.load(out/f'base_{pool}_common_test_trace.npz')['spikes']
        reference=pd.read_csv(out/'minute_predictions.csv');reference=reference[(reference.variant=='base')&(reference.pool==pool)&(reference.probe=='common')].prediction.to_numpy()
        for variant in ['rate_211','rate_307','timing_211','timing_307']:
            m=ContinuousSNN(cfg,104).load(out/f'{variant}_{pool}_network.npz');m.theta=z['theta'].copy()
            _,_,s=stream(m,ev,groups,capture=True,total_rows=len(bars))
            if pool=='random':assert np.array_equal(s[ids],ref)
            f=s[ix].reshape(len(ends),-1);p=apply_saved(out/f'base_{pool}_common_readout.npz',[raw,f])
            summary.append(dict(variant=variant,pool=pool,**perturb_measure(ref,s[ids]),fixed_output_correlation=corr(p,reference),
                fixed_output_mae=float(abs(p-reference).mean()),minute_neuron_count_correlation=corr(ref.sum(1),s[ids].sum(1))))
            preds.append(pd.DataFrame(dict(variant=variant,pool=pool,timestamp=bars.datetime.iloc[ends].astype(str).to_numpy(),prediction=p)))
    pd.DataFrame(summary).to_csv(out/'theta_control.csv',index=False)
    pd.concat(preds).to_csv(out/'theta_control_predictions.csv',index=False)
    # Check all stored per-minute channel counts, initial weights and output keys.
    base=np.unpackbits(inp['packed'],axis=2)[:,:,:24]
    for v in ['timing_211','timing_307']:
        a=np.load(out/f'{v}_input.npz');assert np.array_equal(inp['bar_ids'],a['bar_ids'])
        assert np.array_equal(base.sum(1),np.unpackbits(a['packed'],axis=2)[:,:,:24].sum(1))
    for path in out.glob('*_network.npz'):
        a=np.load(path);assert np.array_equal(a['initial_weight'],z['initial_weight'])
        assert np.allclose(a['weight'].sum(1),cfg['weight_sum'])
    d=pd.read_csv(out/'minute_predictions.csv');assert len(d)==17600 and not d.duplicated(['variant','pool','probe','timestamp']).any()
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
