"""Reuse first-fold frozen models; preserve all 12 x 8 x 64 spike positions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.market_data import ROOT,load
from src.encoding import encode
from src.network import ContinuousSNN
from audit_utils import fit_predict,perturb_measure,finish

def replay(m,events):
    m.reset_state()
    return np.array([m.minute(e)[0] for e in events])

def main():
    out=ROOT/'results/response_value_audit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load();fold=folds[0]
    source=ROOT/'results/first_fold/fold_00';scale=np.load(source/'factor_scale.npy')
    (out/'protocol.json').write_text(json.dumps(dict(window=12,steps=8,target='abs(log(close[t+1]/close[t]))*10000 within segment',
        network='existing first fold; no retraining',calibration=fold['validation_dates'],test=fold['forecast_dates'],
        readout='block-standardized ridge MSE penalty 0.01; calibration only; nonnegative output; mean of 3 seed predictions',
        perturb='last12 encoded minute blocks reversed; or positive channel0 circularly shifted by1 minute; 2 fixed endpoints per test session',
        theta_control='restore initial weights while retaining learned STDP theta; refit same readout',
        scope='historical, one existing fold only; financial label differs from original delayed 5m labels'),indent=2),encoding='utf-8')
    dates=fold['validation_dates']+fold['forecast_dates'];groups=[];ends=[];labels=[]
    for _,g in bars.groupby('segment_id',sort=False):
        ids=g.index.to_numpy();ids=ids[np.isfinite(x[ids]).all(1)]
        if not len(ids) or bars.trading_date.iloc[ids[0]] not in dates:continue
        groups.append(ids)
        for i in ids[11:]:
            ends.append(i)
            labels.append(abs(np.log(bars.close.iloc[i+1]/bars.close.iloc[i]))*1e4 if i<ids[-1] else np.nan)
    ends=np.array(ends);y=np.array(labels);meta=bars.iloc[ends]
    cal=meta.trading_date.isin(fold['validation_dates']).to_numpy()&np.isfinite(y)
    test=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
    raw=np.clip(x[ends[:,None]+np.arange(-11,1)]/scale,-1,1).reshape(len(ends),-1)
    p=pd.DataFrame(dict(fold=0,timestamp=meta.datetime[test].astype(str),date=meta.trading_date[test],actual_abs_1m_bp=y[test])).reset_index(drop=True)
    p['pred_constant']=y[cal].mean();p['pred_raw']=fit_predict([raw],cal,test,y,out/'raw_readout.npz')
    pert=[];seed_rows=[];preds={k:[] for k in ['random','stdp','restored']}
    for seed in cfg['seeds']:
        ev=np.zeros((len(bars),8,24),bool);ids=np.concatenate(groups)
        ev[ids]=encode(np.clip(x[ids]/scale,-1,1),bars.bar_id.to_numpy()[ids],seed,8,cfg['p_max'])
        for pool in preds:
            m=ContinuousSNN(cfg,seed).load(source/f'{"stdp" if pool=="restored" else pool}_{seed}_network.npz')
            if pool=='restored':m.weight=m.initial_weight.copy()
            before=m.weight.copy();theta=m.theta.copy();records=np.zeros((len(bars),8,64),np.uint8)
            for g in groups:
                records[g]=replay(m,ev[g])
                if bars.trading_date.iloc[g[0]] not in fold['forecast_dates']:continue
                for end in [len(g)//2,len(g)-2]:
                    seq=ev[g[:end+1]].copy();orig=records[g[end-11:end+1]]
                    for kind in ['reverse_minutes','shift_channel0']:
                        change=seq.copy()
                        if kind=='reverse_minutes':change[-12:]=change[-12:][::-1]
                        else:change[-12:,:,0]=np.roll(change[-12:,:,0],1,axis=0)
                        assert np.array_equal(seq.sum((0,1)),change.sum((0,1)))
                        altered=replay(m,change)[-12:]
                        pert.append(dict(fold=0,pool=pool,seed=seed,perturbation=kind,timestamp=str(bars.datetime.iloc[g[end]]),
                            input_hamming=float((seq[-12:]!=change[-12:]).mean()),**perturb_measure(orig,altered)))
            assert np.array_equal(m.weight,before) and np.array_equal(m.theta,theta)
            features=records[ends[:,None]+np.arange(-11,1)].reshape(len(ends),-1)
            preds[pool].append(fit_predict([raw,features],cal,test,y,out/f'{pool}_{seed}_readout.npz'))
            seed_rows.append(pd.DataFrame(dict(timestamp=p.timestamp,seed=seed,pool=pool,prediction=preds[pool][-1])))
            np.savez_compressed(out/f'{pool}_{seed}_test_trace.npz',bar_ids=ends[test],spikes=records[ends[test]])
            print(pool,seed,'done',flush=True)
        # Frozen replay is checked against previously saved original forecast traces.
        for pool in ['random','stdp']:
            old=np.load(source/f'{pool}_{seed}_forecast_trace.npz');m=ContinuousSNN(cfg,seed).load(source/f'{pool}_{seed}_network.npz')
            for g in groups:
                if bars.trading_date.iloc[g[0]] in fold['forecast_dates']:
                    ix=np.searchsorted(old['bar_ids'],g);assert np.array_equal(replay(m,ev[g]),old['spikes'][ix])
    for pool,ps in preds.items():p['pred_'+pool]=np.mean(ps,axis=0)
    pd.concat(seed_rows,ignore_index=True).to_csv(out/'seed_predictions.csv',index=False)
    finish(out,p,pert)
    print('complete',len(p),flush=True)

if __name__=='__main__':main()
