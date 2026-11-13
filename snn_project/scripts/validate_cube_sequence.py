"""Fixed-parameter synthetic order audit of the project's NeuCube-style model."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from snn_core.snncube import SNNCube
from snn_core.encoding import encode_factor_thresholds

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/cube_sequence_mechanism'
KINDS=['ABC','BAC','CBA','gap','simultaneous','AB','C_only','none']
SPECS=[[(0,0),(1,1),(3,2)],[(1,0),(0,1),(3,2)],[(3,0),(1,1),(0,2)],
       [(0,0),(2,1),(3,2)],[(3,0),(3,1),(3,2)],[(0,0),(1,1)],[(3,2)],[]]

def background(rng,n):
    x=rng.normal(0,.35,(n,6));x[:,[0,1,3]]=0
    return encode_factor_thresholds(x,np.zeros(6),np.ones(6))

def probes(n,seed):
    rng=np.random.default_rng(seed);out=[]
    for _ in range(n):
        start=int(rng.integers(15,91));x=background(rng,start+7)
        variants=[]
        for spec in SPECS:
            y=x.copy()
            for lag,ch in spec:y[start+lag,ch]=1
            variants.append(y)
        assert all(np.array_equal(v.sum(0),variants[0].sum(0)) for v in variants[:5])
        out.append((start,variants))
    return out

def responses(model,ps):
    before=model.weights.copy()
    result=np.array([[model.simulate(x)[start+3:start+7] for x in xs] for start,xs in ps])
    assert np.array_equal(model.weights,before)
    return result

def paired_win(a,b):return ((a>b)+.5*(a==b)).astype(float)

def main():
    OUT.mkdir(exist_ok=False)
    protocol={'seeds':[43,104,211],'train_sessions':40,'session_minutes':120,
        'patterns_per_session':10,'training_passes':1,'selection_contexts':64,'test_contexts':128,
        'model':{'inputs':11,'gain':2.5,'side':6,'depression':1,'tau':2},
        'encoding':'unchanged factor threshold encoder; center=0 scale=1 synthetic units',
        'background':'N(0,0.35) on other features; A/B/C channels otherwise silent',
        'response':'four steps starting at C time; excludes 11 input port neurons',
        'unit_selection':'max mean ABC minus BAC spike count on selection only',
        'population':'nearest selection ABC/BAC mean 4 x internal-neuron response; equal prior, Euclidean',
        'limitation':'synthetic events; no assertion of real-market learning; no tuning; 128 independent contexts',
        'frozen_financial':'all three existing folds, seed43, same synthetic probes; not trained on ABC'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rng=np.random.default_rng(4101);train=[];shuffled=[]
    for _ in range(40):
        x=background(rng,120);y=x.copy()
        for cell in range(10):
            start=cell*12+int(rng.integers(1,4));times=start+np.array([0,1,3])
            x[times,np.arange(3)]=True
            y[times,rng.permutation(3)]=True
        assert np.array_equal(x.sum(0),y.sum(0))
        train.append(x);shuffled.append(y)
    np.savez_compressed(OUT/'training_inputs.npz',ordered=train,shuffled=shuffled)
    selection=probes(64,4102);test=probes(128,4103)
    models=[]
    for seed in protocol['seeds']:
        for arm in ['random','stdp','shuffled_stdp']:
            m=SNNCube(11,seed,2.5,side=6,depression=1,tau=2)
            initial=m.weights.copy()
            if arm!='random':
                for x in train if arm=='stdp' else shuffled:m.simulate(x,learn=True)
            models.append((f'synthetic_{seed}_{arm}',seed,arm,m,initial))
    for fold in [1,2,3]:
        for arm in ['random','stdp']:
            m=SNNCube(11,43,2.5,side=6,depression=1,tau=2);initial=m.weights.copy()
            p=ROOT/f'experiments/familiarity_conditional_audit/fold{fold}_{arm}_frozen_trace.npz'
            z=np.load(p)
            for key in ['pre','post','sign','ports']:assert np.array_equal(getattr(m,key),z[key])
            m.weights=z['weights'].copy()
            models.append((f'financial_fold{fold}_{arm}',43,arm,m,initial))
    rows=[];details=[];examples={};vectors={}
    for name,seed,arm,m,initial in models:
        a=responses(m,selection);internal=np.flatnonzero(m.internal)
        counts=a[:,:,:,internal].sum(2)
        unit=int(internal[np.argmax((counts[:,0]-counts[:,1]).mean(0))])
        flat=a[:,:,:,internal].reshape(len(a),len(KINDS),-1).astype(float)
        means=flat[:,:2].mean(0)
        (OUT/f'{name}_lock.json').write_text(json.dumps({'unit':unit,'selection_contexts':64}),encoding='utf-8')
        b=responses(m,test);cnt=b[:,:,:,unit].sum(2)
        bf=b[:,:,:,internal].reshape(len(b),len(KINDS),-1).astype(float)
        # Positive score means closer to the selection ABC prototype.
        scores=((bf-means[1])**2).sum(2)-((bf-means[0])**2).sum(2)
        acc=.5*((scores[:,0]>0)+.5*(scores[:,0]==0)+(scores[:,1]<0)+.5*(scores[:,1]==0))
        win=paired_win(cnt[:,0],cnt[:,1])
        vectors[name]=acc
        rows.append(dict(model=name,seed=seed,arm=arm,unit=unit,unit_order_win=win.mean(),
            population_accuracy=acc.mean(),mean_abs_weight_change=np.abs(m.weights-initial).mean(),
            firing_fraction=b.mean(),**{k+'_response_rate':(cnt[:,j]>0).mean() for j,k in enumerate(KINDS)}))
        for i in range(len(b)):
            details.append(dict(model=name,context=i,population_accuracy=acc[i],unit_order_win=win[i],
                **{k+'_count':int(cnt[i,j]) for j,k in enumerate(KINDS)}))
        np.savez_compressed(OUT/f'{name}_responses.npz',selection=a,test=b,initial_weights=initial,
            weights=m.weights,pre=m.pre,post=m.post,ports=m.ports,prototype_means=means)
        print(name,'unit',unit,'unit_order',round(win.mean(),4),'population',round(acc.mean(),4),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'test_contexts.csv',index=False)
    boot=np.random.default_rng(4104).integers(0,128,(2000,128));comparisons=[]
    for control in ['random','shuffled_stdp']:
        d=np.mean([vectors[f'synthetic_{s}_stdp']-vectors[f'synthetic_{s}_{control}'] for s in protocol['seeds']],axis=0)
        lo,hi=np.quantile(d[boot].mean(1),[.025,.975])
        comparisons.append(dict(comparison='stdp minus '+control,difference=d.mean(),low=lo,high=hi))
    pd.DataFrame(comparisons).to_csv(OUT/'population_paired_intervals.csv',index=False)
    files=[Path(__file__),ROOT/'scripts/snn_core/snncube.py',ROOT/'scripts/snn_core/stdp.py',ROOT/'scripts/snn_core/encoding.py']
    (OUT/'verification.json').write_text(json.dumps({'complete':True,'model_count':len(models),
        'matched_counts':True,'evaluation_weights_frozen':True,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2),encoding='utf-8')

if __name__=='__main__':main()
