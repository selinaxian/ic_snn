from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['return_1_atr','return_3_atr','efficiency_5','resistance_distance','support_distance',
 'close_location','volatility_ratio','range_ratio','volume_surprise','oi_change','volatility_level','session_progress']

def factors(g):
    c,h,l,v,oi=[g[n].astype(float) for n in ['close','high','low','volume','open_interest']]
    prev=c.shift(1);tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1);tr.iloc[0]=np.nan
    atr=tr.shift(1).rolling(20).mean().clip(lower=1e-6);r=np.log(c).diff();long=r.shift(1).rolling(20).std(ddof=0)
    f=pd.DataFrame(index=g.index)
    f[FEATURES[0]]=c.diff()/atr;f[FEATURES[1]]=c.diff(3)/atr
    f[FEATURES[2]]=c.diff(5)/c.diff().abs().rolling(5).sum().clip(lower=1e-8)
    f[FEATURES[3]]=(h.shift(1).rolling(20).max()-c)/atr;f[FEATURES[4]]=(c-l.shift(1).rolling(20).min())/atr
    f[FEATURES[5]]=(2*c-h-l)/(h-l).clip(lower=1e-6)
    f[FEATURES[6]]=np.log((r.rolling(5).std(ddof=0)+1e-8)/(long+1e-8))
    f[FEATURES[7]]=np.log((h-l).clip(lower=1e-6)/atr)
    f[FEATURES[8]]=np.log1p(v)-np.log1p(v.shift(1).rolling(20).mean());f[FEATURES[9]]=np.log1p(oi).diff()
    f[FEATURES[10]]=np.log1p(10000*long);f[FEATURES[11]]=2*np.arange(len(g))/119-1
    return f

def load():
    b=pd.read_csv(ROOT/'data/bars.csv',low_memory=False);b.datetime=pd.to_datetime(b.datetime)
    assert np.array_equal(b.bar_id,np.arange(len(b)))
    parts=[]
    for _,g in b.groupby('segment_id',sort=False):
        assert g.contract_id.nunique()==g.trading_date.nunique()==1
        assert g.datetime.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
        assert g.datetime.dt.hour.le(11).all() or g.datetime.dt.hour.ge(13).all()
        parts.append(factors(g))
    x=pd.concat(parts).sort_index().to_numpy()
    assert np.allclose(x,b[FEATURES].to_numpy(),equal_nan=True,atol=1e-9,rtol=1e-7)
    dates=sorted(b.trading_date.unique());daygroups={}
    for d in dates:
        groups=[g.index.to_numpy()[np.isfinite(x[g.index]).all(1)] for _,g in b[b.trading_date==d].groupby('segment_id',sort=False)]
        assert len(groups)==2 and all(len(g)==99 for g in groups),('unexpected day shape',d)
        daygroups[d]=groups
    return b,x,dates,daygroups
