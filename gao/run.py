"""Full previous-day/next-day pipeline; writes a new immutable run directory."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import ROOT,load
from src.encoding import fit_scaler,encode
from src.network import GaoNetwork
from src.strategy import paper_candidates,executable_trades,random_timing_reference,NAMES
from src.evaluation import event_labels,summarize_events

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(args):
    cfg=json.loads((ROOT/'config.json').read_text());out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    models=out/'models';models.mkdir();traces=out/'traces';traces.mkdir()
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    sources=[ROOT/'config.json',ROOT/'data/bars.csv',Path(__file__),*list((ROOT/'src').glob('*.py'))]
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in sources},indent=2),encoding='utf-8')
    b,x,dates,groups=load();pairs=list(zip(dates[:-1],dates[1:]))
    if args.days is not None:pairs=pairs[:args.days]
    pd.DataFrame(pairs,columns=['train_date','test_date']).to_csv(out/'date_splits.csv',index=False)
    rows=[];diags=[];start=time.monotonic();steps=cfg['network']['steps_per_minute']
    for seed in cfg['seeds']:
        for startidx in range(0,len(pairs),cfg['batch_days']):
            batch=pairs[startidx:startidx+cfg['batch_days']];means=[];scales=[];train_ev=[[],[]];test_ev=[[],[]]
            for train_date,test_date in batch:
                mu,sd=fit_scaler(x[np.concatenate(groups[train_date])]);means.append(mu);scales.append(sd)
                for half in range(2):
                    g=groups[train_date][half];train_ev[half].append(encode(x[g],g,mu,sd,seed,steps,cfg['encoding']))
                    g=groups[test_date][half];test_ev[half].append(encode(x[g],g,mu,sd,seed,steps,cfg['encoding']))
            train_ev=[np.array(v) for v in train_ev];test_ev=[np.array(v) for v in test_ev]
            for pool in ['random','stdp']:
                net=GaoNetwork(cfg['network'],seed,len(batch));train_out=np.zeros(len(batch),int)
                if pool=='stdp':
                    for _ in range(cfg['network']['epochs']):
                        for ev in train_ev:
                            _,sp=net.simulate(ev,learn=True);train_out+=sp.sum((1,2)).astype(int)
                wi=net.w_in.copy();wo=net.w_out.copy()
                outputs=[];hiddens=[]
                for half,ev in enumerate(test_ev):
                    hidden,spikes=net.simulate(ev);outputs.append(spikes);hiddens.append(hidden)
                    for j,(_,date) in enumerate(batch):
                        ids=groups[date][half]
                        for k,bar in enumerate(ids):rows.append(dict(seed=seed,pool=pool,train_date=batch[j][0],date=date,bar_id=int(bar),
                            timestamp=str(b.datetime.iloc[bar]),contract=b.contract_id.iloc[bar],segment_id=int(b.segment_id.iloc[bar]),
                            output_count=int(spikes[j,k].sum()),event=bool(spikes[j,k].any()),
                            hidden_h1_count=int(hidden[j,k,:,0].sum()),hidden_h2_count=int(hidden[j,k,:,1].sum())))
                assert np.array_equal(wi,net.w_in) and np.array_equal(wo,net.w_out)
                if pool=='random':assert np.array_equal(wi,net.initial_in) and np.array_equal(wo,net.initial_out)
                for j,(td,date) in enumerate(batch):
                    net.save(models/f'{date}_{seed}_{pool}.npz',j,mean=means[j],scale=scales[j],train_date=td,test_date=date)
                    ids=np.concatenate(groups[date]);s=np.concatenate([o[j] for o in outputs])
                    np.savez_compressed(traces/f'{date}_{seed}_{pool}.npz',bar_ids=ids,output_spikes=s,
                        hidden_counts=np.concatenate([h[j].sum(1) for h in hiddens]))
                    diags.append(dict(seed=seed,pool=pool,train_date=td,test_date=date,train_output_spikes=int(train_out[j]),
                        test_output_spikes=int(s.sum()),test_event_minutes=int(s.any(1).sum()),
                        input_weight_change=float(np.linalg.norm(wi[j]-net.initial_in[j])),
                        output_weight_change=float(np.linalg.norm(wo[j]-net.initial_out[j])),
                        input_weights_at_cap=float((wi[j]>=cfg['network']['weight_max']).mean()),
                        output_weights_at_zero=float((wo[j]<=cfg['network']['weight_min']).mean())))
                # One independent model replay per batch verifies batched computation and serialization.
                probe=GaoNetwork(cfg['network'],seed);probe.w_in[0]=wi[0];probe.w_out[0]=wo[0]
                hh,ss=probe.simulate(test_ev[0][:1]);assert np.array_equal(ss[0],outputs[0][0]) and np.array_equal(hh[0],hiddens[0][0])
            print('models',seed,startidx+len(batch),'/',len(pairs),'elapsed',round(time.monotonic()-start,1),flush=True)
            pd.DataFrame(diags).to_csv(out/'network_diagnostics.csv',index=False)
    signals=pd.DataFrame(rows);signals.to_csv(out/'minute_signals.csv',index=False)
    testdates=[p[1] for p in pairs];testbars=b[b.trading_date.isin(testdates)]
    labeled,metrics=summarize_events(signals,event_labels(testbars,cfg['event_evaluation']['window']))
    labeled.to_csv(out/'event_evaluation.csv',index=False);metrics.to_csv(out/'event_metrics.csv',index=False)
    paper=[];execution=[];daily=[];naive=[]
    for (seed,pool),g in signals.groupby(['seed','pool']):
        ids=g.loc[g.event,'bar_id'].to_numpy(int);r=paper_candidates(testbars,ids,cfg['strategy']['holding_steps'])
        for a in r:a.update(seed=int(seed),pool=pool)
        paper.extend(r)
        r,d=executable_trades(testbars,ids,cfg,cfg['strategy']['holding_steps'])
        for a in r+d:a.update(seed=int(seed),pool=pool)
        execution.extend(r);daily.extend(d)
        nr=random_timing_reference(testbars,g,cfg['strategy']['random_timing_repeats'],int(seed)+9000,cfg['strategy']['holding_steps'])
        for a in nr:a.update(seed=int(seed),pool=pool)
        naive.extend(nr);print('strategies',seed,pool,'complete',flush=True)
    pd.DataFrame(paper).to_csv(out/'paper_style_trades.csv',index=False)
    pd.DataFrame(execution).to_csv(out/'cost_scenario_trades.csv',index=False)
    day=pd.DataFrame(daily);day.to_csv(out/'cost_scenario_daily.csv',index=False)
    pd.DataFrame(naive).to_csv(out/'random_timing_reference.csv',index=False)
    summaries=[]
    for (seed,pool,strategy),g in day.groupby(['seed','pool','strategy']):
        tr=[r for r in execution if r['seed']==seed and r['pool']==pool and r['strategy']==strategy]
        gross=sum(r['gross_yuan'] for r in tr);costs=sum(r['slippage_yuan']+r['fee_yuan'] for r in tr)
        summaries.append(dict(seed=seed,pool=pool,strategy=strategy,days=len(g),trades=len(tr),gross_yuan=gross,costs_yuan=costs,
            net_yuan=float(g.daily_net.sum()),return_on_initial=float(g.capital.iloc[-1]/cfg['cost_scenario']['capital_yuan']-1),
            win_rate=float(np.mean([r['net_yuan']>0 for r in tr])) if tr else np.nan,
            daily_sharpe=float(g.daily_return.mean()/g.daily_return.std(ddof=1)*np.sqrt(252)) if g.daily_return.std(ddof=1)>0 else np.nan,
            max_closed_daily_drawdown=float(g.closed_equity_drawdown.min())))
    pd.DataFrame(summaries).to_csv(out/'strategy_metrics.csv',index=False)
    assert not signals.duplicated(['seed','pool','bar_id']).any()
    (out/'complete.json').write_text(json.dumps(dict(complete=True,pairs=len(pairs),seeds=cfg['seeds'],trained_models=len(pairs)*len(cfg['seeds']),
        frozen_random_models=len(pairs)*len(cfg['seeds']),minute_rows=len(signals),unique_minutes=int(signals.bar_id.nunique()),
        elapsed_seconds=time.monotonic()-start,feature_rebuild_verified=True,test_weights_frozen=True,batched_vs_single_replay_exact=True,
        scope='Gao-inspired multifactor minute adaptation, not exact published replication; cost scenario not historical fee audit'),indent=2),encoding='utf-8')
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/baseline');p.add_argument('--days',type=int);main(p.parse_args())
