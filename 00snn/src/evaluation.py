"""Forecast metrics and paired five-day-block diagnostic intervals."""
import numpy as np
import pandas as pd


def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else np.nan


def metrics(frame,targets):
    rows=[]
    for model,g in frame.groupby('model'):
        for target in targets:
            y=g['actual_'+target].to_numpy();p=g[target].to_numpy()
            rows.append(dict(model=model,target=target,n=len(g),ic=correlation(y,p),
                rank_ic=correlation(pd.Series(y).rank(),pd.Series(p).rank()),
                rmse=float(np.sqrt(np.mean((y-p)**2)))))
    return pd.DataFrame(rows)


def paired(frame,targets):
    keyed={m:g.sort_values('sample_id').reset_index(drop=True) for m,g in frame.groupby('model')}
    left=keyed['stdp'];days=left.trading_date.to_numpy();unique=np.unique(days)
    # Nonoverlapping contiguous five-day blocks, resampled as whole blocks.
    blocks=[np.flatnonzero(np.isin(days,unique[i:i+5])) for i in range(0,len(unique),5)]
    if len(blocks)<2:return pd.DataFrame()
    rng=np.random.default_rng(20261111);draws=rng.integers(0,len(blocks),(400,len(blocks)))
    rows=[]
    for rightname in ['random','raw_history']:
        right=keyed[rightname];assert np.array_equal(left.sample_id,right.sample_id)
        for target in targets:
            a=left[target].to_numpy();b=right[target].to_numpy();y=left['actual_'+target].to_numpy()
            values=[]
            for draw in draws:
                idx=np.concatenate([blocks[k] for k in draw])
                values.append(correlation(y[idx],a[idx])-correlation(y[idx],b[idx]))
            lo,hi=np.nanquantile(values,[.025,.975])
            rows.append(dict(left='stdp',right=rightname,target=target,
                ic_difference=correlation(y,a)-correlation(y,b),low95=lo,high95=hi,blocks=len(blocks)))
    return pd.DataFrame(rows)
