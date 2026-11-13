"""One real minute at a time. reset_state() never modifies learned parameters."""
import numpy as np
from .stdp import update


class ContinuousSNN:
    def __init__(self,cfg,seed,inputs=24):
        self.cfg=dict(cfg)
        w=np.random.default_rng(seed).uniform(.02,.98,(cfg['hidden'],inputs))
        self.weight=w*(cfg['weight_sum']/w.sum(axis=1,keepdims=True))
        self.initial_weight=self.weight.copy()
        self.theta=np.zeros(cfg['hidden'])
        self.reset_state()

    def reset_state(self):
        self.voltage=np.zeros(self.weight.shape[0])
        self.pre_trace=np.zeros(self.weight.shape[1])
        self.post_trace=np.zeros(self.weight.shape[0])

    def step(self,pre,learn=False,adapt=False):
        c=self.cfg
        self.voltage*=np.exp(-1/c['tau_mem'])
        self.voltage+=self.weight@pre
        excess=self.voltage-c['threshold']-self.theta
        winner=int(np.argmax(excess))
        post=np.zeros(len(self.voltage))
        if excess[winner]>=0:
            post[winner]=1
            self.voltage[winner]=0
            self.voltage-=c['inhibition']*(1-post)
            np.maximum(self.voltage,0,out=self.voltage)
        self.pre_trace*=np.exp(-1/c['tau_trace'])
        self.post_trace*=np.exp(-1/c['tau_trace'])
        if learn:
            update(self.weight,self.pre_trace,self.post_trace,pre,post,c)
        self.pre_trace+=pre
        self.post_trace+=post
        if adapt:
            self.theta*=c['theta_decay']
            self.theta+=c['theta_increment']*post
        return post

    def minute(self,spikes,learn=False,adapt=False):
        record=np.array([self.step(pre,learn,adapt) for pre in spikes],dtype=np.uint8)
        return record,self.voltage/(self.cfg['threshold']+self.theta)

    def save(self,path):
        np.savez_compressed(path,weight=self.weight,initial_weight=self.initial_weight,theta=self.theta)

    def load(self,path):
        with np.load(path) as a:
            self.weight=a['weight'].copy();self.initial_weight=a['initial_weight'].copy();self.theta=a['theta'].copy()
        self.reset_state()
        return self
