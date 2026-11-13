"""Local bounded pair STDP; financial labels never enter this module."""
import numpy as np


def delta(weight,pre_trace,post_trace,pre,post,cfg):
    return (cfg['eta_plus']*(1-weight)*np.outer(post,pre_trace)
            -cfg['eta_minus']*weight*np.outer(post_trace,pre))


def update(weight,pre_trace,post_trace,pre,post,cfg):
    weight += delta(weight,pre_trace,post_trace,pre,post,cfg)
    np.clip(weight,1e-6,1,out=weight)
    weight *= cfg['weight_sum']/weight.sum(axis=1,keepdims=True)
    np.minimum(weight,1,out=weight)
