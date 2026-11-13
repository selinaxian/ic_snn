"""Fixed synthetic sequence audit; no financial labels, no parameter search."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.encoding import encode
from src.network import ContinuousSNN

ROOT=Path(__file__).resolve().parent
KINDS=['target','reverse','same_last','other_gap','simultaneous','missing_c','other_identity']
ARMS=['random','stdp','shuffled_stdp']
SIGNAL_CHANNELS=[0,1,2]


def make_data(sessions,seed,cfg):
    rng=np.random.default_rng(seed)
    values=rng.uniform(-.2,.2,(sessions,120,12));values[:,:,:3]=0
    anchors=np.arange(10)[None,:]*12+rng.integers(1,4,(sessions,10))
    for s in range(sessions):
        for start in anchors[s]:
            for ch,lag in enumerate([0,1,3]):values[s,start+lag,ch]=rng.uniform(.8,1.)
    ids=np.arange(sessions*120)+seed*10000
    spikes=encode(values.reshape(-1,12),ids,771,cfg['steps_per_minute'],cfg['p_max']).reshape(sessions,120,8,24)
    return spikes,anchors


def alternatives(region):
    pulses=[region[t,:,ch].copy() for ch,t in enumerate([0,1,3])]
    variants=[]
    specifications=[([0,1,3],[0,1,2]),([3,1,0],[0,1,2]),([1,0,3],[0,1,2]),
                    ([0,2,3],[0,1,2]),([3,3,3],[0,1,2]),([0,1,-1],[0,1,2]),
                    ([0,1,3],[12,13,2])]
    for times,channels in specifications:
        x=region.copy();x[:,:,[0,1,2,12,13,14]]=False
        for i,(t,ch) in enumerate(zip(times,channels)):
            if t>=0:x[t,:,ch]=pulses[i]
        variants.append(x)
    a=np.array(variants)
    assert np.array_equal(a[0],region)
    assert np.all(a[:5].sum(axis=(1,2))==a[0].sum(axis=(0,1)))
    assert a[6].sum()==a[0].sum()
    return a


def shuffled(spikes,anchors):
    rng=np.random.default_rng(90123);out=spikes.copy()
    for sid,starts in enumerate(anchors):
        for start in starts:
            times=start+np.array([0,1,3]);perm=rng.permutation(3)
            block=spikes[sid,times,:,:3].copy()
            out[sid,times,:,:3]=block[perm]
    assert np.array_equal(out.sum(axis=(1,2)),spikes.sum(axis=(1,2)))
    return out


def train(model,spikes,plastic):
    shape=model.weight.shape
    ltp=np.zeros(shape);ltd=np.zeros(shape);norm=np.zeros(shape)
    snapshots=[]
    for epoch in range(model.cfg['epochs']):
        for session in spikes:
            model.reset_state()
            for pre in session.reshape(-1,24):
                if plastic:
                    old=model.weight.copy()
                    pretrace=model.pre_trace*np.exp(-1/model.cfg['tau_trace'])
                    posttrace=model.post_trace*np.exp(-1/model.cfg['tau_trace'])
                post=model.step(pre,learn=plastic,adapt=True)
                if plastic:
                    p=model.cfg['eta_plus']*(1-old)*np.outer(post,pretrace)
                    n=model.cfg['eta_minus']*old*np.outer(posttrace,pre)
                    ltp+=p;ltd+=n;norm+=model.weight-old-p+n
        snapshots.append(model.weight.copy())
    error=np.max(np.abs(ltp-ltd+norm-(model.weight-model.initial_weight)))
    assert error<1e-9
    return dict(ltp=ltp,ltd=ltd,normalization_and_clipping=norm,epoch_weights=np.array(snapshots),reconstruction_error=error)


def copy_state(model):
    return [model.voltage.copy(),model.pre_trace.copy(),model.post_trace.copy()]


def branch(model,state,weight):
    other=ContinuousSNN.__new__(ContinuousSNN)
    other.cfg=model.cfg;other.weight=weight;other.theta=model.theta
    other.voltage,other.pre_trace,other.post_trace=[v.copy() for v in state]
    return other


def contexts(model,spikes,anchors):
    cases=[];ids=[]
    for sid,session in enumerate(spikes):
        model.reset_state();positions=set(anchors[sid])
        for minute,events in enumerate(session):
            if minute in positions:
                cases.append((copy_state(model),alternatives(session[minute:minute+7])))
                ids.append((sid,minute))
            model.minute(events)
    return cases,np.array(ids)


def evaluate(model,cases,weight,save_first=False):
    scores=[];first=[]
    for i,(state,inputs) in enumerate(cases):
        row=[]
        for events in inputs:
            m=branch(model,state,weight)
            recorded=np.array([m.minute(v)[0] for v in events])
            row.append(recorded[3:].sum(axis=(0,1)))
            if i==0 and save_first:first.append(recorded)
        scores.append(row)
    return np.array(scores),np.array(first)


def select(scores,model,cases):
    contrast=scores[:,0].mean(axis=0)-scores[:,1:5].mean(axis=(0,1))
    unit=int(np.argmax(contrast))
    key=int(np.argmax(model.weight[unit,:3]))
    candidates=np.array([i for i in range(24) if i not in SIGNAL_CHANNELS])
    control=int(candidates[np.argmin(abs(model.weight[unit,candidates]-model.weight[unit,key]))])
    activity=np.array([inputs[0].sum(axis=(0,1)) for _,inputs in cases]).mean(axis=0)
    return dict(neuron=unit,key_channel=key,control_channel=control,selection_contrast=float(contrast[unit]),
        key_weight=float(model.weight[unit,key]),key_initial_weight=float(model.initial_weight[unit,key]),
        control_weight=float(model.weight[unit,control]),key_mean_input_count=float(activity[key]),
        control_mean_input_count=float(activity[control]))


def summary(rows):
    result=[]
    for (seed,arm,intervention),g in rows.groupby(['seed','arm','intervention']):
        wide=g.pivot(index=['session','anchor'],columns='kind',values='response')
        for kind in KINDS[1:]:
            t=wide.target.to_numpy();c=wide[kind].to_numpy()
            result.append(dict(seed=seed,arm=arm,intervention=intervention,control=kind,n=len(t),
                target_mean=t.mean(),control_mean=c.mean(),mean_difference=(t-c).mean(),
                paired_order_accuracy=np.mean((t>c)+.5*(t==c)),
                target_response_rate=np.mean(t>0),control_response_rate=np.mean(c>0)))
    return pd.DataFrame(result)


def intervals(rows):
    base=rows[rows.intervention=='none']
    wide=base.pivot(index=['seed','arm','session','anchor'],columns='kind',values='response').reset_index()
    results=[];rng=np.random.default_rng(11901)
    for control in ['same_last','reverse','other_gap','simultaneous']:
        wide['contrast']=wide.target-wide[control]
        wide['pair_win']=(wide.target>wide[control])+.5*(wide.target==wide[control])
        for metric in ['contrast','pair_win']:
            means=wide.groupby(['arm','session'])[metric].mean().unstack('arm')
            for ref in ['random','shuffled_stdp']:
                d=(means.stdp-means[ref]).to_numpy()
                draws=rng.integers(0,len(d),(2000,len(d)))
                lo,hi=np.quantile(d[draws].mean(axis=1),[.025,.975])
                results.append(dict(control=control,metric=metric,left='stdp',right=ref,
                    difference=d.mean(),low95=lo,high95=hi,session_blocks=len(d)))
    return pd.DataFrame(results)


def main(args):
    cfg=json.loads((ROOT/'config.json').read_text())
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    protocol=(ROOT/'MECHANISM_PROTOCOL.md').read_text(encoding='utf-8')
    (out/'protocol.md').write_text(protocol,encoding='utf-8')
    files=[ROOT/'config.json',ROOT/'MECHANISM_PROTOCOL.md',Path(__file__)]+list((ROOT/'src').glob('*.py'))
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    data={name:make_data(n,seed,cfg) for name,n,seed in [('train',40,10101),('selection',16,20202),('test',32,30303)]}
    shuffled_train=shuffled(*data['train'])
    for name,(spikes,anchors) in data.items():
        np.savez_compressed(out/f'{name}_inputs.npz',packed_spikes=np.packbits(spikes,axis=-1),anchors=anchors)
    checks={'per_channel_count_match':True,'training_shuffle_count_match':True,'weights_frozen':True}
    all_rows=[];selection_rows=[];weight_rows=[];omission=[];start=time.monotonic()
    for name,(spikes,anchors) in data.items():
        counts=np.array([[spikes[s,t+lag,:,ch].sum() for ch,lag in enumerate([0,1,3])] for s,starts in enumerate(anchors) for t in starts])
        omission.append(dict(split=name,patterns=len(counts),all_three_encoded_fraction=float((counts>0).all(axis=1).mean()),
            mean_signal_spikes=float(counts.sum(axis=1).mean())))
    pd.DataFrame(omission).to_csv(out/'encoding_coverage.csv',index=False)
    for seed in cfg['seeds']:
        for arm in ARMS:
            model=ContinuousSNN(cfg,seed)
            audit=train(model,shuffled_train if arm=='shuffled_stdp' else data['train'][0],arm!='random')
            model.save(out/f'{arm}_{seed}_network.npz')
            np.savez_compressed(out/f'{arm}_{seed}_plasticity_audit.npz',**audit)
            for unit in range(64):
                for channel in range(24):
                    weight_rows.append(dict(seed=seed,arm=arm,neuron=unit,channel=channel,
                        initial=model.initial_weight[unit,channel],final=model.weight[unit,channel],
                        ltp=audit['ltp'][unit,channel],ltd=audit['ltd'][unit,channel],
                        normalization_and_clipping=audit['normalization_and_clipping'][unit,channel]))
            weights,theta=model.weight.copy(),model.theta.copy()
            cal_cases,cal_ids=contexts(model,*data['selection'])
            cal_scores,_=evaluate(model,cal_cases,weights)
            choice=select(cal_scores,model,cal_cases)
            choice.update(seed=seed,arm=arm)
            # Selection lock is written before this model's test is evaluated.
            (out/f'{arm}_{seed}_selection_lock.json').write_text(json.dumps(choice,indent=2))
            selection_rows.append(choice)
            test_cases,test_ids=contexts(model,*data['test'])
            neuron,key,control=[choice[k] for k in ['neuron','key_channel','control_channel']]
            interventions={'none':weights.copy(),'remove_key':weights.copy(),'remove_control':weights.copy(),'restore_key_initial':weights.copy()}
            interventions['remove_key'][neuron,key]=0
            interventions['remove_control'][neuron,control]=0
            interventions['restore_key_initial'][neuron,key]=model.initial_weight[neuron,key]
            for name,w in interventions.items():
                scores,example=evaluate(model,test_cases,w,save_first=name=='none')
                if name=='none':
                    np.savez_compressed(out/f'{arm}_{seed}_responses.npz',selection=cal_scores,selection_ids=cal_ids,
                                        test=scores,test_ids=test_ids,example_inputs=test_cases[0][1],example_spikes=example)
                for i,(sid,anchor) in enumerate(test_ids):
                    for k,kind in enumerate(KINDS):
                        all_rows.append(dict(seed=seed,arm=arm,intervention=name,session=int(sid),anchor=int(anchor),
                            kind=kind,neuron=neuron,response=int(scores[i,k,neuron])))
            assert np.array_equal(model.weight,weights) and np.array_equal(model.theta,theta)
            print(f'{arm} seed {seed}: unit {neuron}, selected contrast {choice["selection_contrast"]:.4f}; {time.monotonic()-start:.1f}s',flush=True)
            pd.DataFrame(selection_rows).to_csv(out/'selected_units.csv',index=False)
    rows=pd.DataFrame(all_rows)
    rows.to_csv(out/'test_responses.csv',index=False)
    summary(rows).to_csv(out/'response_summary.csv',index=False)
    intervals(rows).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(out/'connection_updates.csv',index=False)
    checks.update(models=9,test_background_sessions=32,test_cases_per_model=320,
                  elapsed_seconds=time.monotonic()-start,status='complete')
    (out/'verification.json').write_text(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/sequence_mechanism')
    main(p.parse_args())
