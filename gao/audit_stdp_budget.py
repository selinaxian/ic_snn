"""STDP accounting and fixed incoming-weight-budget ablation; no return labels."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import ROOT, load
from src.encoding import fit_scaler, encode
from src.network import GaoNetwork


class AccountedNetwork(GaoNetwork):
    def __init__(self,cfg,seed,batch,budget=False):
        super().__init__(cfg,seed,batch)
        self.budget=budget
        self.ledger={layer:{key:np.zeros(batch) for key in ['ltp','ltd','zero_ltp','clip','normalization']}
                     for layer in ['hidden','output']}

    def step(self,pre,learn=False):
        if not learn:return super().step(pre,False)
        c=self.cfg;t=self.t;self.t+=1
        available=t-self.last_h>c['refractory_steps'];ao=t-self.last_o>c['refractory_steps']
        current=np.einsum('bhni,bhi->bhn',self.w_in,pre,optimize=False)
        self.v=np.where(available,np.exp(-1/c['tau_mem'])*self.v+current,0)
        self.vo=np.where(ao,np.exp(-1/c['tau_mem'])*self.vo+(self.w_out*self.previous).sum(1),0)
        fired=(self.v>=c['threshold'])&available;out=(self.vo>=c['threshold'])&ao
        self.last_x=np.where(pre,t,self.last_x);self.last_arrival=np.where(self.previous,t,self.last_arrival)
        pretrace=np.exp(-(t-self.last_x)/c['tau_stdp']);posttrace=np.exp(-(t-self.last_h)/c['tau_stdp'])
        hp=c['a_plus']*fired[:,:,:,None]*pretrace[:,:,None,:]
        hm=c['a_minus']*(~fired)[:,:,:,None]*posttrace[:,:,:,None]*pre[:,:,None,:]
        op=c['a_plus']*out[:,None]*np.exp(-(t-self.last_arrival)/c['tau_stdp'])
        om=c['a_minus']*(~out)[:,None]*np.exp(-(t-self.last_o)/c['tau_stdp'])[:,None]*self.previous
        for layer,w,plus,minus,zero,target,initial in [
            ('hidden',self.w_in,hp,hm,hp*pre[:,:,None,:],c['initial_hidden_input_sum'],self.initial_in),
            ('output',self.w_out,op,om,op*self.previous,c['initial_output_input_sum'],self.initial_out)]:
            axes=tuple(range(1,w.ndim));acc=self.ledger[layer]
            acc['ltp']+=plus.sum(axis=axes);acc['ltd']+=minus.sum(axis=axes);acc['zero_ltp']+=zero.sum(axis=axes)
            w+=plus;w-=minus
            before=w.sum(axis=axes);np.clip(w,c['weight_min'],c['weight_max'],out=w)
            acc['clip']+=w.sum(axis=axes)-before
            if self.budget:
                before=w.sum(axis=axes);s=w.sum(-1,keepdims=True)
                # Explicit zero-row fallback, never used in the run unless recorded by failure below.
                assert (s>0).all(), 'zero incoming row requires separately specified intervention'
                w*=target/s
                assert w.min()>=0 and w.max()<=c['weight_max']+1e-12
                acc['normalization']+=w.sum(axis=axes)-before
        self.v[fired]=0;self.vo[out]=0
        self.last_h=np.where(fired,t,self.last_h);self.last_o=np.where(out,t,self.last_o)
        self.previous=fired.reshape(self.batch,-1)
        return fired,out


def main():
    start=time.monotonic();source=ROOT/'results/baseline';dest=ROOT/'results/stdp_budget_audit'
    dest.mkdir(exist_ok=False);(dest/'models').mkdir();(dest/'traces').mkdir()
    cfg=json.loads((source/'config.json').read_text());bars,x,_,groups=load()
    pairs=pd.read_csv(source/'date_splits.csv');steps=cfg['network']['steps_per_minute']
    (dest/'protocol.json').write_text(json.dumps(dict(config=cfg,variants=['random','baseline','budget'],
        normalization='after each original STDP update and clipping, rescale each hidden incoming row to 2.4 and output row to 1.2; no other change',
        seeds='training unchanged; inference seed = original + repeat*1000003, repeat=0,1,2',
        scope='All existing date pairs. Mechanism diagnosis, no financial labels or parameter selection.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/network.py',ROOT/'src/encoding.py',source/'config.json']}),indent=2),encoding='utf-8')
    accounts=[];training=[];signals=[];comparisons=[]
    for seed in cfg['seeds']:
        for startidx in range(0,len(pairs),24):
            batch=pairs.iloc[startidx:startidx+24];dates=list(batch.test_date);means=[];scales=[];train=[[],[]]
            for row in batch.itertuples():
                mu,sd=fit_scaler(x[np.concatenate(groups[row.train_date])]);means.append(mu);scales.append(sd)
                for half in [0,1]:
                    ids=groups[row.train_date][half]
                    train[half].append(encode(x[ids],ids,mu,sd,seed,steps,cfg['encoding']))
            for variant in ['random','baseline','budget']:
                net=AccountedNetwork(cfg['network'],seed,len(batch),variant=='budget')
                if variant!='random':
                    for half in [0,1]:
                        ev=np.array(train[half]);net.reset()
                        for minute in range(ev.shape[1]):
                            hc=np.zeros(len(batch));oc=np.zeros(len(batch))
                            for k in range(steps):
                                h,o=net.step(ev[:,minute,k],True);hc+=h.sum((1,2));oc+=o
                            for j,date in enumerate(dates):
                                training.append(dict(seed=seed,variant=variant,date=date,half=half,minute=minute,
                                    hidden_count=hc[j],output_count=oc[j],hidden_mean_input_sum=net.w_in[j].sum(-1).mean(),
                                    output_sum=net.w_out[j].sum(),output_fraction_ge99=(net.w_out[j]>=.99).mean()))
                for j,date in enumerate(dates):
                    if variant in ['baseline','random']:
                        old=np.load(source/f'models/{date}_{seed}_{"stdp" if variant=="baseline" else "random"}.npz')
                        assert np.array_equal(net.w_in[j],old['w_in']) and np.array_equal(net.w_out[j],old['w_out'])
                    net.save(dest/f'models/{date}_{seed}_{variant}.npz',j,mean=means[j],scale=scales[j])
                    for layer,w,initial in [('hidden',net.w_in,net.initial_in),('output',net.w_out,net.initial_out)]:
                        acc={k:float(v[j]) for k,v in net.ledger[layer].items()}
                        delta=float(w[j].sum()-initial[j].sum())
                        assert np.isclose(delta,acc['ltp']-acc['ltd']+acc['clip']+acc['normalization'],atol=1e-7)
                        accounts.append(dict(seed=seed,variant=variant,date=date,layer=layer,**acc,
                            final_sum=w[j].sum(),initial_sum=initial[j].sum(),delta=delta,
                            fraction_ge99=(w[j]>=.99).mean(),relative_weight_change=np.linalg.norm(w[j]-initial[j])/np.linalg.norm(initial[j])))
                wi=net.w_in.copy();wo=net.w_out.copy()
                for half in [0,1]:
                    responses={}
                    for control,repeat in [('actual',0),('actual',1),('actual',2),('constant',0)]:
                        events=[]
                        for j,date in enumerate(dates):
                            ids=groups[date][half];v=x[ids] if control=='actual' else np.broadcast_to(means[j],x[ids].shape)
                            events.append(encode(v,ids,means[j],scales[j],seed+repeat*1000003,steps,cfg['encoding']))
                        hidden,output=net.simulate(np.array(events));hc=hidden.sum(2).reshape(len(batch),-1,128);oc=output.sum(2)
                        responses[(control,repeat)]=(hc,oc)
                        for j,date in enumerate(dates):
                            if control=='actual' and repeat==0:
                                np.savez_compressed(dest/f'traces/{date}_{seed}_{variant}_{half}.npz',
                                    bar_ids=groups[date][half],hidden_counts=hc[j].astype(np.uint8),output_spikes=output[j])
                                if variant in ['baseline','random']:
                                    old=np.load(source/f'traces/{date}_{seed}_{"stdp" if variant=="baseline" else "random"}.npz')
                                    idx=slice(half*99,(half+1)*99)
                                    assert np.array_equal(output[j],old['output_spikes'][idx])
                            for k,bar in enumerate(groups[date][half]):
                                signals.append(dict(seed=seed,variant=variant,date=date,bar_id=int(bar),half=half,control=control,repeat=repeat,
                                    hidden_count=int(hc[j,k].sum()),hidden_active=int((hc[j,k]>0).sum()),output_count=int(oc[j,k]),event=int(oc[j,k]>0)))
                    h0,o0=responses[('actual',0)]
                    for (control,repeat),(hh,oo) in responses.items():
                        if (control,repeat)==('actual',0):continue
                        for j,date in enumerate(dates):
                            # Center within half-day, preventing model-day scale differences from inflating correlation.
                            a=o0[j].astype(float);b=oo[j].astype(float);a-=a.mean();b-=b.mean()
                            comparisons.append(dict(seed=seed,variant=variant,date=date,half=half,control=control,repeat=repeat,
                                hidden_vector_changed=(h0[j]!=hh[j]).any(-1).mean(),hidden_cell_mae=np.abs(h0[j].astype(float)-hh[j]).mean(),
                                output_count_changed=(o0[j]!=oo[j]).mean(),output_count_mae=np.abs(o0[j].astype(float)-oo[j]).mean(),
                                event_changed=((o0[j]>0)!=(oo[j]>0)).mean(),cross=float(a@b),var_a=float(a@a),var_b=float(b@b)))
                    assert np.array_equal(wi,net.w_in) and np.array_equal(wo,net.w_out)
            print(seed,startidx+len(batch),'/',len(pairs),'elapsed',round(time.monotonic()-start),flush=True)
    for name,rows in [('learning_accounts',accounts),('training_trajectory',training),('minute_signals',signals),('response_comparisons',comparisons)]:
        pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)
    a=pd.DataFrame(accounts);a.groupby(['seed','variant','layer']).mean(numeric_only=True).to_csv(dest/'learning_summary.csv')
    s=pd.DataFrame(signals);s['all_hidden_active']=s.hidden_active.eq(128)
    s.groupby(['seed','variant','control','repeat']).agg(hidden_count=('hidden_count','mean'),all_hidden_active=('all_hidden_active','mean'),
        output_count=('output_count','mean'),event_rate=('event','mean')).to_csv(dest/'activity_summary.csv')
    c=pd.DataFrame(comparisons);summ=[]
    for keys,g in c.groupby(['seed','variant','control','repeat']):
        row=dict(zip(['seed','variant','control','repeat'],keys));row.update(g.drop(columns=['seed','half','repeat']).mean(numeric_only=True).to_dict())
        den=np.sqrt(g.var_a.sum()*g.var_b.sum());row['within_half_output_corr']=g.cross.sum()/den if den>0 else np.nan;summ.append(row)
    pd.DataFrame(summ).to_csv(dest/'response_summary.csv',index=False)
    (dest/'complete.json').write_text(json.dumps(dict(complete=True,baseline_weights_exact=True,baseline_predictions_exact=True,
        accounts_reconcile=True,test_weights_frozen=True,elapsed_seconds=time.monotonic()-start,trained_models=1080,
        random_models=540,minute_rows=len(s),unique_minutes=s.bar_id.nunique()),indent=2),encoding='utf-8')


if __name__=='__main__':main()
