"""Read standalone CSV snapshot, rebuild causal factors and executable labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['return_1_atr','return_3_atr','efficiency_5','resistance_distance','support_distance',
          'close_location','volatility_ratio','range_ratio','volume_surprise','oi_change',
          'volatility_level','session_progress']


def factors(g):
    c,h,l,v,oi=[g[n].astype(float) for n in ['close','high','low','volume','open_interest']]
    prev=c.shift(1)
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    tr.iloc[0]=np.nan
    atr=tr.shift(1).rolling(20).mean().clip(lower=1e-6)
    r=np.log(c).diff();long=r.shift(1).rolling(20).std(ddof=0)
    f=pd.DataFrame(index=g.index)
    f[FEATURES[0]]=c.diff()/atr
    f[FEATURES[1]]=c.diff(3)/atr
    f[FEATURES[2]]=c.diff(5)/c.diff().abs().rolling(5).sum().clip(lower=1e-8)
    f[FEATURES[3]]=(h.shift(1).rolling(20).max()-c)/atr
    f[FEATURES[4]]=(c-l.shift(1).rolling(20).min())/atr
    f[FEATURES[5]]=(2*c-h-l)/(h-l).clip(lower=1e-6)
    f[FEATURES[6]]=np.log((r.rolling(5).std(ddof=0)+1e-8)/(long+1e-8))
    f[FEATURES[7]]=np.log((h-l).clip(lower=1e-6)/atr)
    f[FEATURES[8]]=np.log1p(v)-np.log1p(v.shift(1).rolling(20).mean())
    f[FEATURES[9]]=np.log1p(oi).diff()
    f[FEATURES[10]]=np.log1p(10000*long)
    f[FEATURES[11]]=2*np.arange(len(g))/119-1
    return f


def load():
    bars=pd.read_csv(ROOT/'data/bars.csv',low_memory=False)
    bars.datetime=pd.to_datetime(bars.datetime)
    assert np.array_equal(bars.bar_id,np.arange(len(bars)))
    groups=[]
    for _,g in bars.groupby('segment_id',sort=False):
        assert g.contract_id.nunique()==g.trading_date.nunique()==1
        assert g.datetime.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
        groups.append(factors(g))
    x=pd.concat(groups).sort_index().to_numpy()
    saved=bars[FEATURES].to_numpy()
    if not np.allclose(x,saved,equal_nan=True,rtol=1e-7,atol=1e-9):
        raise ValueError('Rebuilt factors differ from source snapshot')
    samples=pd.read_csv(ROOT/'data/samples.csv')
    end=samples.bar_id.to_numpy(int);entry=samples.entry_bar_id.to_numpy(int)
    paths=entry[:,None]+np.arange(6)
    seg=bars.segment_id.to_numpy()
    assert (seg[paths]==seg[end,None]).all()
    assert np.array_equal(entry,end+2)
    assert (seg[end[:,None]+np.arange(-11,1)]==seg[end,None]).all()
    op=bars.open.to_numpy()[paths]
    ret=(op[:,-1]/op[:,0]-1)*10000
    rv=np.sqrt((np.diff(np.log(op),axis=1)**2).sum(axis=1))*10000
    y=np.column_stack([ret,abs(ret),rv])
    reference=np.load(ROOT/'data/reference_labels.npz')['y'][:,:3]
    assert np.allclose(y,reference,atol=1e-8)
    folds=json.loads((ROOT/'data/folds.json').read_text())
    return bars,x,samples,y,folds


def scale_from_training(bars,x,dates):
    mask=bars.trading_date.isin(dates).to_numpy()&np.isfinite(x).all(axis=1)
    return np.maximum(np.quantile(abs(x[mask]),.99,axis=0),1e-8)
