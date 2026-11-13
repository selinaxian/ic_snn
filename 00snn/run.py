"""Run chronological half-session SNN folds. Existing outputs are never overwritten."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.market_data import ROOT,FEATURES,load,scale_from_training
from src.encoding import encode
from src.network import ContinuousSNN
from src.readout import summarize,fit,predict
from src.evaluation import metrics,paired


def stream(model,events,groups,learn=False,adapt=False,capture=False,total_rows=0):
    if capture:
        counts=np.full((total_rows,model.cfg['hidden']),np.nan)
        state=counts.copy();records=np.zeros((total_rows,model.cfg['steps_per_minute'],model.cfg['hidden']),np.uint8)
    for ids in groups:
        model.reset_state()
        for i in ids:
            spikes,v=model.minute(events[i],learn,adapt)
            if capture:
                records[i]=spikes;counts[i]=spikes.sum(axis=0);state[i]=v
    if capture:return counts,state,records


def main(args):
    cfg=json.loads((ROOT/'config.json').read_text())
    if args.seeds:cfg['seeds']=[int(s) for s in args.seeds.split(',')]
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    inputs=list((ROOT/'data').glob('*'))+[ROOT/'config.json']+list((ROOT/'src').glob('*.py'))+[Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    bars,x,samples,y,folds=load()
    if args.folds is not None:folds=folds[:args.folds]
    (out/'folds.json').write_text(json.dumps(folds,indent=2))
    allpred=[];allminutes=[];diagnostics=[];started=time.monotonic()
    valid=np.isfinite(x).all(axis=1)
    basegroups=[g.index.to_numpy()[valid[g.index]] for _,g in bars.groupby('segment_id',sort=False)]
    basegroups=[g for g in basegroups if len(g)]
    for fold in folds:
        f=fold['fold'];dest=out/f'fold_{f:02d}';dest.mkdir()
        scale=scale_from_training(bars,x,fold['train_dates'])
        np.save(dest/'factor_scale.npy',scale)
        groups=[g for g in basegroups if bars.trading_date.iloc[g[0]]<=max(fold['forecast_dates'])]
        train_groups=[g for g in groups if bars.trading_date.iloc[g[0]] in fold['train_dates']]
        train=samples.trading_date.isin(fold['train_dates']).to_numpy()
        val=samples.trading_date.isin(fold['validation_dates']).to_numpy()
        test=samples.trading_date.isin(fold['forecast_dates']).to_numpy()
        active=train|val|test;meta=samples[active].reset_index(drop=True);yy=y[active]
        train,val,test=train[active],val[active],test[active]
        end=meta.bar_id.to_numpy(int)
        # Live score eligibility uses only past data, never horizon completeness.
        live=np.concatenate([g[cfg['readout_minutes']-1:] for g in groups if bars.trading_date.iloc[g[0]] in fold['forecast_dates']])
        ix=end[:,None]+np.arange(-11,1)
        raw=np.clip(x[ix]/scale,-1,1).reshape(len(end),-1)
        raw_model=fit(raw[train],yy[train],raw[val],yy[val],cfg['ridge_alphas'])
        np.savez_compressed(dest/'raw_history_readout.npz',**raw_model)
        forecasts={'raw_history':predict(raw_model,raw[test])}
        live_raw=np.clip(x[live[:,None]+np.arange(-11,1)]/scale,-1,1).reshape(len(live),-1)
        live_forecasts={'raw_history':predict(raw_model,live_raw)}
        for pool in ['random','stdp']:
            ps=[];pls=[]
            for seed in cfg['seeds']:
                model=ContinuousSNN(cfg,seed)
                ev=np.zeros((len(bars),cfg['steps_per_minute'],24),bool)
                ev[valid]=encode(np.clip(x[valid]/scale,-1,1),bars.bar_id.to_numpy()[valid],seed,cfg['steps_per_minute'],cfg['p_max'])
                for epoch in range(cfg['epochs']):
                    stream(model,ev,train_groups,learn=pool=='stdp',adapt=True)
                before=model.weight.copy();theta=model.theta.copy()
                counts,state,spikes=stream(model,ev,groups,capture=True,total_rows=len(bars))
                assert np.array_equal(before,model.weight) and np.array_equal(theta,model.theta)
                if pool=='random':assert np.array_equal(model.weight,model.initial_weight)
                rep=summarize(counts,state,end,cfg)
                readout=fit(rep[train],yy[train],rep[val],yy[val],cfg['ridge_alphas'])
                ps.append(predict(readout,rep[test]));pls.append(predict(readout,summarize(counts,state,live,cfg)))
                model.save(dest/f'{pool}_{seed}_network.npz')
                np.savez_compressed(dest/f'{pool}_{seed}_readout.npz',**readout)
                trace_ids=np.concatenate([g for g in groups if bars.trading_date.iloc[g[0]] in fold['forecast_dates']])
                np.savez_compressed(dest/f'{pool}_{seed}_forecast_trace.npz',bar_ids=trace_ids,
                                    spikes=spikes[trace_ids],normalized_voltage=state[trace_ids],representation_bar_ids=live,
                                    representations=summarize(counts,state,live,cfg))
                # Save/load prediction equality on one complete session.
                reload_model=ContinuousSNN(cfg,seed).load(dest/f'{pool}_{seed}_network.npz')
                first=next(g for g in groups if bars.trading_date.iloc[g[0]] in fold['forecast_dates'])
                for i in first:
                    z,v=reload_model.minute(ev[i])
                    assert np.array_equal(z,spikes[i]) and np.allclose(v,state[i],atol=0,rtol=0)
                w=model.weight;cos=(w@w.T)/np.outer(np.linalg.norm(w,axis=1),np.linalg.norm(w,axis=1))
                diagnostics.append(dict(fold=f,pool=pool,seed=seed,training_minutes=sum(map(len,train_groups)),
                    epochs=cfg['epochs'],test_minutes=len(trace_ids),weight_change_l2=float(np.linalg.norm(w-model.initial_weight)),
                    mean_weight_cosine=float(cos[~np.eye(len(w),dtype=bool)].mean()),
                    theta_mean=float(model.theta.mean()),mean_firing_rate=float(spikes[trace_ids].mean()),
                    active_neurons=int((spikes[trace_ids].sum(axis=(0,1))>0).sum()),reload_exact=True,weights_frozen=True))
                print(f'fold {f:02d} {pool} seed {seed} complete ({time.monotonic()-started:.1f}s)',flush=True)
            forecasts[pool]=np.mean(ps,axis=0);live_forecasts[pool]=np.mean(pls,axis=0)
        # Persist a fold before progressing; safe partial results if interrupted.
        rows=[];live_rows=[]
        for name,p in forecasts.items():
            q=meta[test].copy();q['fold']=f;q['model']=name
            for j,target in enumerate(cfg['targets']):q[target]=p[:,j];q['actual_'+target]=yy[test,j]
            rows.append(q)
            l=bars.iloc[live][['bar_id','segment_id','contract_id','trading_date','datetime']].copy()
            l['fold']=f;l['model']=name;l['in_common_evaluation']=np.isin(live,end[test])
            for j,target in enumerate(cfg['targets']):l[target]=live_forecasts[name][:,j]
            live_rows.append(l)
        pd.concat(rows).to_csv(dest/'common_predictions.csv',index=False)
        pd.concat(live_rows).to_csv(dest/'minute_signals.csv',index=False)
        allpred.extend(rows);allminutes.extend(live_rows)
        pd.DataFrame(diagnostics).to_csv(out/'network_diagnostics.csv',index=False)
    pred=pd.concat(allpred,ignore_index=True);live=pd.concat(allminutes,ignore_index=True)
    pred.to_csv(out/'common_predictions.csv',index=False);live.to_csv(out/'minute_signals.csv',index=False)
    summary=metrics(pred,cfg['targets']);summary.to_csv(out/'metrics.csv',index=False)
    intervals=paired(pred,cfg['targets']);intervals.to_csv(out/'paired_ic_intervals.csv',index=False)
    completed=dict(folds=len(folds),seeds=cfg['seeds'],elapsed_seconds=time.monotonic()-started,
        common_points=int((pred.model=='stdp').sum()),live_points=int((live.model=='stdp').sum()),
        input_factor_and_label_rebuild_verified=True,frozen_weights_and_reload_verified=True,
        status='complete',scope='Historical diagnostic; not new unseen data; no strategy backtest')
    (out/'complete.json').write_text(json.dumps(completed,indent=2))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/continuous_run')
    p.add_argument('--folds',type=int,default=None);p.add_argument('--seeds',default=None)
    main(p.parse_args())
