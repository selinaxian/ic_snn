"""Fixed chronological diagnostic readout and paired test-day uncertainty."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def fit_predict(blocks,cal,test,y,path):
    # Each block has mean squared norm ~1, avoiding a reward for feature count.
    zs=[];params={};slices=[];offset=0
    for j,x in enumerate(blocks):
        x=np.asarray(x,dtype=np.float64)
        mean=x[cal].mean(0);sd=x[cal].std(0);keep=sd>1e-8
        scale=np.where(keep,sd,1)*np.sqrt(max(int(keep.sum()),1))
        z=(x-mean)/scale;z[:,~keep]=0
        zs.append(z);params[f'mean_{j}']=mean;params[f'scale_{j}']=scale;params[f'keep_{j}']=keep
        slices.append((offset,offset+x.shape[1]));offset+=x.shape[1]
    z=np.concatenate(zs,axis=1);a=z[cal];ym=float(y[cal].mean());target=y[cal]-ym
    penalty=len(a)*.01
    if a.shape[1]<=len(a):
        gram=a.T@a;gram.flat[::len(gram)+1]+=penalty
        coef=np.linalg.solve(gram,a.T@target)
    else:
        gram=a@a.T;gram.flat[::len(gram)+1]+=penalty
        coef=a.T@np.linalg.solve(gram,target)
    params.update(coef=coef,target_mean=ym,lambda_mse=.01,slices=np.array(slices))
    np.savez_compressed(path,**params)
    return np.maximum(z[test]@coef+ym,0)

def perturb_measure(original,changed):
    a=original.astype(bool);b=changed.astype(bool);union=(a|b).sum()
    return dict(hamming=float((a!=b).mean()),jaccard=float((a&b).sum()/union) if union else 1.,
        both_silent=bool(union==0),original_rate=float(a.mean()),changed_rate=float(b.mean()),
        final_minute_hamming=float((a[-1]!=b[-1]).mean()))

def finish(out,pred,perturb):
    pred.to_csv(out/'minute_signals.csv',index=False)
    pd.DataFrame(perturb).to_csv(out/'input_perturbations.csv',index=False)
    metrics=[];intervals=[]
    for fold,g in pred.groupby('fold'):
        valid=np.isfinite(g.actual_abs_1m_bp);q=g[valid]
        for model in [c[5:] for c in g if c.startswith('pred_')]:
            err=q['pred_'+model]-q.actual_abs_1m_bp
            metrics.append(dict(fold=fold,model=model,points=len(q),days=q.date.nunique(),
                rmse_bp=float(np.sqrt(np.mean(err**2))),mae_bp=float(np.mean(abs(err)))))
        for ref in ['raw','random','restored']:
            if 'pred_'+ref not in q:continue
            d=(q.pred_stdp-q.actual_abs_1m_bp)**2-(q['pred_'+ref]-q.actual_abs_1m_bp)**2
            days=pd.DataFrame(dict(date=q.date,delta=d)).groupby('date').delta.agg(['sum','count'])
            ids=np.random.default_rng(501).integers(0,len(days),(2000,len(days)))
            bs=days['sum'].to_numpy()[ids].sum(1)/days['count'].to_numpy()[ids].sum(1)
            lo,hi=np.quantile(bs,[.025,.975])
            intervals.append(dict(fold=fold,comparison='stdp minus '+ref,mse_delta_bp2=d.mean(),low=lo,high=hi,test_days=len(days)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_mse_intervals.csv',index=False)
    pd.DataFrame(perturb).groupby(['fold','pool','perturbation']).mean(numeric_only=True).to_csv(out/'perturbation_summary.csv')
    (out/'complete.json').write_text(json.dumps(dict(complete=True,minute_rows=len(pred),labeled_rows=int(pred.actual_abs_1m_bp.notna().sum()),
        note='Historical diagnostic only; no new untouched data; fixed readout; no strategy or costs'),indent=2),encoding='utf-8')
