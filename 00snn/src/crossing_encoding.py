"""Causal, deterministic signed threshold-crossing events with hysteresis."""
import numpy as np

def channels(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all():raise ValueError('finite inputs required')
    return np.concatenate([np.maximum(values,0),np.maximum(-values,0)],axis=1)

def fit_thresholds(training_values):
    a=channels(training_values);high=np.full(a.shape[1],np.inf);low=high.copy();n=[]
    for j in range(a.shape[1]):
        positive=a[:,j][a[:,j]>0];n.append(len(positive))
        if len(positive)>=20:
            high[j]=np.quantile(positive,.8)
            low[j]=min(np.quantile(positive,.6),.8*high[j])
    return dict(high=high,low=low,positive_count=np.array(n))

def encode_session(values,thresholds,steps=8):
    a=channels(values);out=np.zeros((len(a),steps,a.shape[1]),bool)
    if not len(a):return out
    hi=thresholds['high'];lo=thresholds['low'];armed=a[0]<hi
    # First valid observation establishes state; it is not an observed crossing.
    for t in range(1,len(a)):
        armed |= a[t]<=lo
        fire=armed & (a[t-1]<hi) & (a[t]>=hi)
        out[t,0]=fire;armed[fire]=False
    return out
