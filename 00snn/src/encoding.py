"""Rate encoding keyed by market bar identity, never by overlapping window."""
import numpy as np


def encode(values, bar_ids, seed, steps=8, p_max=.3):
    values=np.asarray(values)
    if not np.isfinite(values).all() or np.max(np.abs(values))>1.000001:
        raise ValueError('Encoding requires finite scaled factors in [-1,1]')
    channels=np.concatenate([np.maximum(values,0),np.maximum(-values,0)],axis=1)
    with np.errstate(over='ignore'):
        x=np.asarray(bar_ids,dtype=np.uint32)[:,None,None]*np.uint32(2654435761)
        x=x+np.arange(steps*channels.shape[1],dtype=np.uint32).reshape(1,steps,-1)*np.uint32(2246822519)+np.uint32(seed)
        x=(x^(x>>16))*np.uint32(0x7feb352d)
        x=(x^(x>>15))*np.uint32(0x846ca68b)
        x=x^(x>>16)
    uniforms=(x.astype(np.float64)+.5)/4294967296.
    return uniforms < p_max*channels[:,None,:]
