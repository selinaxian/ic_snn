"""Batched independent days, two feed-forward branches, all synapses plastic.

No recurrent connection, winner-take-all, adaptive threshold or supervised output.
Hidden spikes reach output one simulation step later. Nearest-spike additive STDP
is applied to arriving pre-spikes, with zero-lag pairs potentiated once.
"""
import numpy as np

class GaoNetwork:
    def __init__(self,cfg,seed,batch=1):
        self.cfg=dict(cfg);self.batch=batch;h=cfg['hidden_per_branch'];rng=np.random.default_rng(seed)
        a=rng.uniform(.5,1.5,(2,h,12));a*=cfg['initial_hidden_input_sum']/a.sum(-1,keepdims=True)
        w=rng.uniform(.5,1.5,2*h);w*=cfg['initial_output_input_sum']/w.sum()
        self.w_in=np.broadcast_to(a,(batch,*a.shape)).copy();self.w_out=np.broadcast_to(w,(batch,len(w))).copy()
        self.initial_in=self.w_in.copy();self.initial_out=self.w_out.copy();self.reset()

    def reset(self):
        b=self.batch;h=self.cfg['hidden_per_branch'];self.t=0
        self.v=np.zeros((b,2,h));self.vo=np.zeros(b);self.last_h=np.full((b,2,h),-100000.)
        self.last_x=np.full((b,2,12),-100000.);self.last_o=np.full(b,-100000.)
        self.last_arrival=np.full((b,2*h),-100000.);self.previous=np.zeros((b,2*h),bool)

    def step(self,pre,learn=False):
        c=self.cfg;t=self.t;self.t+=1
        available=t-self.last_h>c['refractory_steps'];ao=t-self.last_o>c['refractory_steps']
        current=np.einsum('bhni,bhi->bhn',self.w_in,pre,optimize=False)
        self.v=np.where(available,np.exp(-1/c['tau_mem'])*self.v+current,0)
        self.vo=np.where(ao,np.exp(-1/c['tau_mem'])*self.vo+(self.w_out*self.previous).sum(1),0)
        fired=(self.v>=c['threshold'])&available;out=(self.vo>=c['threshold'])&ao
        self.last_x=np.where(pre,t,self.last_x);self.last_arrival=np.where(self.previous,t,self.last_arrival)
        if learn:
            pretrace=np.exp(-(t-self.last_x)/c['tau_stdp']);posttrace=np.exp(-(t-self.last_h)/c['tau_stdp'])
            self.w_in+=c['a_plus']*fired[:,:,:,None]*pretrace[:,:,None,:]
            self.w_in-=c['a_minus']*(~fired)[:,:,:,None]*posttrace[:,:,:,None]*pre[:,:,None,:]
            self.w_out+=c['a_plus']*out[:,None]*np.exp(-(t-self.last_arrival)/c['tau_stdp'])
            self.w_out-=c['a_minus']*(~out)[:,None]*np.exp(-(t-self.last_o)/c['tau_stdp'])[:,None]*self.previous
            np.clip(self.w_in,c['weight_min'],c['weight_max'],out=self.w_in)
            np.clip(self.w_out,c['weight_min'],c['weight_max'],out=self.w_out)
        self.v[fired]=0;self.vo[out]=0
        self.last_h=np.where(fired,t,self.last_h);self.last_o=np.where(out,t,self.last_o)
        self.previous=fired.reshape(self.batch,-1)
        return fired,out

    def simulate(self,events,learn=False):
        # events: independent days x real minutes x simulation steps x branch x input
        self.reset();b,n,steps=events.shape[:3];h=self.cfg['hidden_per_branch']
        hidden=np.zeros((b,n,steps,2,h),np.uint8);output=np.zeros((b,n,steps),np.uint8)
        for i in range(n):
            for k in range(steps):hidden[:,i,k],output[:,i,k]=self.step(events[:,i,k],learn)
        return hidden,output

    def save(self,path,index=0,**metadata):
        np.savez_compressed(path,w_in=self.w_in[index],w_out=self.w_out[index],
            initial_in=self.initial_in[index],initial_out=self.initial_out[index],**metadata)
