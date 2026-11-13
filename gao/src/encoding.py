import numpy as np

def fit_scaler(x):
    return x.mean(0),np.maximum(x.std(0),1e-8)

def rates(x,mean,scale,cfg):
    # Both x and -x receive positive-mean normalization, not sign clipping.
    z=(x-mean)/scale
    return np.maximum(np.stack([z,-z],axis=-2)*cfg['target_std']+cfg['target_mean'],0)

def encode(x,bar_ids,mean,scale,seed,steps,cfg):
    rate=rates(x,mean,scale,cfg)
    # Discretized Poisson: probability of >=1 event in unit simulation interval.
    p=-np.expm1(-rate)
    with np.errstate(over='ignore'):
        u=np.asarray(bar_ids,dtype=np.uint32)[:,None,None,None]*np.uint32(2654435761)
        u=u+np.arange(steps*24,dtype=np.uint32).reshape(1,steps,2,12)*np.uint32(2246822519)+np.uint32(seed)
        u=(u^(u>>16))*np.uint32(0x7feb352d);u=(u^(u>>15))*np.uint32(0x846ca68b);u=u^(u>>16)
    return (u.astype(float)+.5)/4294967296 < p[:,None,:,:]
