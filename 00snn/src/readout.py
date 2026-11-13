"""Retain old 192-dimensional diagnostics; not a claim that Ridge is optimal."""
import numpy as np


def summarize(counts,voltage,bar_ids,cfg):
    width=cfg['readout_minutes'];recent=cfg['recent_minutes'];steps=cfg['steps_per_minute']
    ix=bar_ids[:,None]+np.arange(-width+1,1)
    ir=bar_ids[:,None]+np.arange(-recent+1,1)
    z=np.concatenate([counts[ix].sum(axis=1)/(width*steps),counts[ir].sum(axis=1)/(recent*steps),voltage[bar_ids]],axis=1)
    assert np.isfinite(z).all()
    return z


def fit(x,y,validation,yvalidation,alphas):
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-6)
    a=(x-mean)/scale;b=(validation-mean)/scale;ym=y.mean(axis=0)
    eig,vec=np.linalg.eigh(a.T@a);rhs=vec.T@(a.T@(y-ym))
    coefs=np.array([vec@(rhs/(np.maximum(eig,0)[:,None]+alpha)) for alpha in alphas])
    preds=np.einsum('nd,adj->anj',b,coefs)+ym
    preds[:,:,1:]=np.maximum(preds[:,:,1:],0)
    mse=np.mean((preds-yvalidation)**2,axis=1);best=np.argmin(mse,axis=0)
    return dict(mean=mean,scale=scale,target_mean=ym,
                coef=np.column_stack([coefs[best[j],:,j] for j in range(y.shape[1])]),
                alpha=np.asarray(alphas)[best],validation_mse=mse)


def predict(model,x):
    y=(x-model['mean'])/model['scale']@model['coef']+model['target_mean']
    y[:,1:]=np.maximum(y[:,1:],0)
    return y
