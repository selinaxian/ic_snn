"""Fixed network initialization; vary encoding draws or only within-minute positions."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.market_data import ROOT,load
from src.encoding import encode
from src.network import ContinuousSNN
from run import stream
from audit_utils import fit_predict,perturb_measure

def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else float('nan')

def permute_minutes(e,seed):
    rng=np.random.default_rng(seed)
    order=np.argsort(rng.random(e.shape),axis=1)
    result=np.take_along_axis(e,order,axis=1)
    assert np.array_equal(result.sum(1),e.sum(1))
    return result

def apply_saved(path,blocks):
    z=np.load(path);parts=[]
    for j,b in enumerate(blocks):
        v=(b-z[f'mean_{j}'])/z[f'scale_{j}'];v[:,~z[f'keep_{j}']]=0;parts.append(v)
    return np.maximum(np.concatenate(parts,axis=1)@z['coef']+z['target_mean'],0)

def main():
    out=ROOT/'results/encoding_stability_audit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load();fold=folds[0]
    protocol=dict(network_seed=104,encoding_seeds=[104,211,307],permutation_seeds=[211,307],
        variants=['base','rate_211','rate_307','timing_211','timing_307'],epochs=cfg['epochs'],
        scope='fixed first fold, one initialization, 3 rate draws including baseline, 2 count-preserving permutations',
        common_probe='baseline encoded calibration/test input for every trained network',
        native_probe='each network receives its own encoding at calibration/test',
        training='same realization repeated across epochs, original adaptation rules for random and stdp',
        readout='same prior audit: full12x8x64 plus raw, calibration only, fixed lambda0.01',
        target='next minute absolute close-to-close log return bp; no horizon/encoding/model tuning',
        comparisons='weights and updates by neuron identity; nearest weight cosine is many-to-one descriptive only; common and native responses; total prediction and increment over raw',
        dates=fold)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    valid=np.isfinite(x).all(1);groups=[]
    for _,g in bars.groupby('segment_id',sort=False):
        ids=g.index.to_numpy()[valid[g.index]]
        if len(ids) and bars.trading_date.iloc[ids[0]] in fold['train_dates']+fold['validation_dates']+fold['forecast_dates']:groups.append(ids)
    train=[g for g in groups if bars.trading_date.iloc[g[0]] in fold['train_dates']]
    evaluate=[g for g in groups if bars.trading_date.iloc[g[0]] not in fold['train_dates']]
    testgroups=[g for g in evaluate if bars.trading_date.iloc[g[0]] in fold['forecast_dates']]
    active=np.concatenate(groups);testids=np.concatenate(testgroups)
    ends=np.concatenate([g[11:] for g in evaluate]);meta=bars.iloc[ends]
    scale=np.load(ROOT/'results/first_fold/fold_00/factor_scale.npy')
    y=np.full(len(bars),np.nan)
    for g in evaluate:y[g[:-1]]=abs(np.log(bars.close.to_numpy()[g[1:]]/bars.close.to_numpy()[g[:-1]]))*1e4
    y=y[ends];cal=meta.trading_date.isin(fold['validation_dates']).to_numpy()&np.isfinite(y)
    test=meta.trading_date.isin(fold['forecast_dates']).to_numpy();ix=ends[:,None]+np.arange(-11,1)
    raw=np.clip(x[ix]/scale,-1,1).reshape(len(ends),-1)
    rawpred=fit_predict([raw],cal,test,y,out/'raw_readout.npz')
    base=np.zeros((len(bars),8,24),bool)
    base[active]=encode(np.clip(x[active]/scale,-1,1),bars.bar_id.to_numpy()[active],104,8,cfg['p_max'])
    refs={};summary=[];predrows=[];inputrows=[];t0=time.monotonic()
    for variant in protocol['variants']:
        ev=base.copy()
        if variant.startswith('rate'):
            ev[active]=encode(np.clip(x[active]/scale,-1,1),bars.bar_id.to_numpy()[active],int(variant.split('_')[1]),8,cfg['p_max'])
        elif variant.startswith('timing'):ev[active]=permute_minutes(base[active],int(variant.split('_')[1]))
        inputrows.append(dict(variant=variant,changed_cells=float((ev[active]!=base[active]).mean()),
            changed_minute_channel_counts=float((ev[active].sum(1)!=base[active].sum(1)).mean())))
        np.savez_compressed(out/f'{variant}_input.npz',bar_ids=active,packed=np.packbits(ev[active],axis=2))
        for pool in ['random','stdp']:
            m=ContinuousSNN(cfg,104);initial=m.weight.copy()
            for _ in range(cfg['epochs']):stream(m,ev,train,learn=pool=='stdp',adapt=True)
            m.save(out/f'{variant}_{pool}_network.npz')
            if variant=='base':
                old=np.load(ROOT/f'results/first_fold/fold_00/{pool}_104_network.npz')
                assert np.array_equal(m.weight,old['weight']) and np.array_equal(m.theta,old['theta'])
            w=m.weight.copy();theta=m.theta.copy()
            for probe,events in [('common',base),('native',ev)]:
                _,_,s=stream(m,events,evaluate,capture=True,total_rows=len(bars))
                assert np.array_equal(w,m.weight) and np.array_equal(theta,m.theta)
                f=s[ix].reshape(len(ends),-1)
                path=out/f'{variant}_{pool}_{probe}_readout.npz'
                pred=fit_predict([raw,f],cal,test,y,path)
                fixed=apply_saved(out/f'base_{pool}_common_readout.npz',[raw[test],f[test]])
                if variant=='base':refs[(pool,probe)]=dict(w=w.copy(),theta=theta.copy(),s=s[testids].copy(),pred=pred.copy())
                ref=refs[(pool,probe)];rw=ref['w'];delta=w-initial;rd=rw-initial
                cos=(w@rw.T)/np.outer(np.linalg.norm(w,axis=1),np.linalg.norm(rw,axis=1))
                top=np.argsort(w,axis=1)[:,-3:];rtop=np.argsort(rw,axis=1)[:,-3:]
                overlap=np.mean([len(set(a)&set(b))/3 for a,b in zip(top,rtop)])
                good=np.isfinite(y[test]);measure=perturb_measure(ref['s'],s[testids])
                summary.append(dict(variant=variant,pool=pool,probe=probe,weight_relative_l2=float(np.linalg.norm(w-rw)/np.linalg.norm(rw)),
                    weight_update_cosine=float(np.sum(delta*rd)/(np.linalg.norm(delta)*np.linalg.norm(rd))) if np.linalg.norm(delta)*np.linalg.norm(rd)>0 else np.nan,
                    weight_same_neuron_cosine=float(np.diag(cos).mean()),weight_nearest_cosine=float(cos.max(1).mean()),
                    top3_overlap=overlap,theta_difference_l2=float(np.linalg.norm(theta-ref['theta'])),
                    prediction_correlation=corr(pred,ref['pred']),increment_correlation=corr(pred-rawpred,ref['pred']-rawpred),
                    prediction_difference_mae=float(abs(pred-ref['pred']).mean()),increment_rms=float(np.sqrt(np.mean((pred-rawpred)**2))),
                    fixed_readout_difference_mae=float(abs(fixed-ref['pred']).mean()),
                    rmse_bp=float(np.sqrt(np.mean((pred[good]-y[test][good])**2))),raw_rmse_bp=float(np.sqrt(np.mean((rawpred[good]-y[test][good])**2))),
                    **measure))
                predrows.append(pd.DataFrame(dict(timestamp=meta.datetime[test].astype(str).to_numpy(),variant=variant,pool=pool,probe=probe,
                    actual_abs_1m_bp=y[test],prediction=pred,raw_prediction=rawpred,fixed_baseline_readout_prediction=fixed)))
                np.savez_compressed(out/f'{variant}_{pool}_{probe}_test_trace.npz',bar_ids=testids,spikes=s[testids])
            print(variant,pool,'complete',round(time.monotonic()-t0,1),'seconds',flush=True)
    pd.DataFrame(summary).to_csv(out/'stability_summary.csv',index=False)
    pd.DataFrame(inputrows).to_csv(out/'input_checks.csv',index=False)
    pd.concat(predrows,ignore_index=True).to_csv(out/'minute_predictions.csv',index=False)
    sources=[Path(__file__),ROOT/'src/encoding.py',ROOT/'src/network.py',ROOT/'src/stdp.py',ROOT/'audit_utils.py']
    (out/'verification.json').write_text(json.dumps(dict(complete=True,trained_networks=10,baseline_exactly_reproduced=True,
        fixed_count_assertions_passed=True,evaluation_frozen=True,unique_test_minutes=int(test.sum()),labeled_test_minutes=int(np.isfinite(y[test]).sum()),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf-8')

if __name__=='__main__':main()
