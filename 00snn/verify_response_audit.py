"""Independently reconstruct saved test predictions from frozen models/readouts."""
import json
import hashlib
import numpy as np
import pandas as pd
from src.market_data import ROOT,load
from src.encoding import encode
from src.network import ContinuousSNN
from response_value_audit import replay

def main():
    out=ROOT/'results/response_value_audit';source=ROOT/'results/first_fold/fold_00'
    p=pd.read_csv(out/'minute_signals.csv');cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load()
    fold=folds[0];scale=np.load(source/'factor_scale.npy');dates=fold['forecast_dates']
    groups=[g.index.to_numpy()[np.isfinite(x[g.index]).all(1)] for _,g in bars[bars.trading_date.isin(dates)].groupby('segment_id',sort=False)]
    ends=np.concatenate([g[11:] for g in groups]);assert np.array_equal(bars.datetime.iloc[ends].astype(str),p.timestamp)
    raw=np.clip(x[ends[:,None]+np.arange(-11,1)]/scale,-1,1).reshape(len(ends),-1);rows=[];errors=[]
    for seed in cfg['seeds']:
        ev=np.zeros((len(bars),8,24),bool);ids=np.concatenate(groups)
        ev[ids]=encode(np.clip(x[ids]/scale,-1,1),bars.bar_id.to_numpy()[ids],seed,8,cfg['p_max'])
        for pool in ['random','stdp','restored']:
            m=ContinuousSNN(cfg,seed).load(source/f'{"stdp" if pool=="restored" else pool}_{seed}_network.npz')
            if pool=='restored':m.weight=m.initial_weight.copy()
            s=np.zeros((len(bars),8,64),np.uint8)
            for g in groups:s[g]=replay(m,ev[g])
            blocks=[raw,s[ends[:,None]+np.arange(-11,1)].reshape(len(ends),-1)]
            z=np.load(out/f'{pool}_{seed}_readout.npz');parts=[]
            for j,b in enumerate(blocks):
                a=(b-z[f'mean_{j}'])/z[f'scale_{j}'];a[:,~z[f'keep_{j}']]=0;parts.append(a)
            values=np.maximum(np.concatenate(parts,axis=1)@z['coef']+z['target_mean'],0)
            rows.append(pd.DataFrame(dict(timestamp=p.timestamp,seed=seed,pool=pool,prediction=values)))
    allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(out/'seed_predictions.csv',index=False)
    for pool in ['random','stdp','restored']:
        reconstructed=allrows[allrows.pool==pool].groupby('timestamp').prediction.mean().reindex(p.timestamp).to_numpy()
        error=float(np.max(abs(reconstructed-p['pred_'+pool])));errors.append(error);assert error<1e-10
    joined=allrows.merge(p[['timestamp','actual_abs_1m_bp']],on='timestamp').dropna()
    joined['squared_error']=(joined.prediction-joined.actual_abs_1m_bp)**2
    joined.groupby(['pool','seed']).squared_error.mean().pow(.5).rename('rmse_bp').to_csv(out/'seed_metrics.csv')
    (out/'verification.json').write_text(json.dumps(dict(max_prediction_reconstruction_error=max(errors),
        unique_minutes=not p.timestamp.duplicated().any(),test_points=len(p),labeled_points=int(p.actual_abs_1m_bp.notna().sum()),
        complete=True,source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'response_value_audit.py',ROOT/'audit_utils.py',ROOT/'src/network.py',ROOT/'src/encoding.py']}),indent=2),encoding='utf-8')

if __name__=='__main__':main()
