"""Fixed deterministic crossing contrast; no raw inputs in supervised readout."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.market_data import ROOT,load,FEATURES
from src.network import ContinuousSNN
from src.encoding import encode
from src.crossing_encoding import fit_thresholds,encode_session
from run import stream
from audit_utils import fit_predict
from encoding_stability_audit import apply_saved

def main():
    out=ROOT/'results/deterministic_crossing_audit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'config.json').read_text());bars,x,_,_,folds=load();fold=folds[0]
    (out/'protocol.json').write_text(json.dumps(dict(
        threshold='per signed channel positive training values q80 trigger; reset min(q60,0.8*trigger); disable <20 positives',
        boundary='first valid minute establishes state without firing; state reset at original segment boundaries',
        pulse='one event at step0 of existing8 steps, unit amplitude, no gain/threshold adjustments',
        dynamics='unchanged tau_mem=tau_trace=16 steps=2minutes; minute retention exp(-8/16); theta decay unchanged',
        seeds=cfg['seeds'],epochs=cfg['epochs'],readout='ONLY12x8x64 spikes; fixed ridge penalty .01; calibration only',
        target='next-minute absolute close log return bp',comparison='same dates, seeds, dynamics, readout; original frozen Poisson models reused',
        limitation='coding contrast also changes event semantics and density; not pure randomness ablation',dates=fold),indent=2),encoding='utf-8')
    groups=[]
    for _,g in bars.groupby('segment_id',sort=False):
        ids=g.index.to_numpy();ids=ids[np.isfinite(x[ids]).all(1)]
        if len(ids) and bars.trading_date.iloc[ids[0]] in fold['train_dates']+fold['validation_dates']+fold['forecast_dates']:groups.append(ids)
    train=[g for g in groups if bars.trading_date.iloc[g[0]] in fold['train_dates']]
    evaluate=[g for g in groups if bars.trading_date.iloc[g[0]] not in fold['train_dates']]
    tg=[g for g in evaluate if bars.trading_date.iloc[g[0]] in fold['forecast_dates']]
    tri=np.concatenate(train);active=np.concatenate(groups);ti=np.concatenate(tg)
    scale=np.load(ROOT/'results/first_fold/fold_00/factor_scale.npy');values=np.clip(x/scale,-1,1)
    threshold=fit_thresholds(values[tri]);np.savez_compressed(out/'encoding_thresholds.npz',scale=scale,**threshold)
    events=np.zeros((len(bars),8,24),bool)
    for g in groups:
        events[g]=encode_session(values[g],threshold)
        assert np.array_equal(events[g],encode_session(values[g],threshold))
        n=len(g)//2;assert np.array_equal(events[g[:n]],encode_session(values[g[:n]],threshold))
    assert not events[:,1:].any() and (events.sum(1)<=1).all()
    np.savez_compressed(out/'crossing_inputs.npz',bar_ids=active,packed=np.packbits(events[active],axis=2))
    channels=[f'positive_{f}' for f in FEATURES]+[f'negative_{f}' for f in FEATURES]
    pd.DataFrame(dict(channel=channels,high=threshold['high'],reset=threshold['low'],positive_training_count=threshold['positive_count'],
        train_events=events[tri].sum((0,1)),test_events=events[ti].sum((0,1)))).to_csv(out/'channel_events.csv',index=False)
    ends=np.concatenate([g[11:] for g in evaluate]);ix=ends[:,None]+np.arange(-11,1);meta=bars.iloc[ends]
    yfull=np.full(len(bars),np.nan)
    for g in evaluate:yfull[g[:-1]]=abs(np.log(bars.close.to_numpy()[g[1:]]/bars.close.to_numpy()[g[:-1]]))*1e4
    y=yfull[ends];test=meta.trading_date.isin(fold['forecast_dates']).to_numpy();cal=~test&np.isfinite(y)
    rows=[];diagnostics=[];inputs=[];epochrows=[];started=time.monotonic()
    for seed in cfg['seeds']:
        for coding in ['crossing','poisson']:
            ev=events
            if coding=='poisson':
                ev=np.zeros_like(events);ev[active]=encode(values[active],bars.bar_id.to_numpy()[active],seed,8,cfg['p_max'])
            inputs.append(dict(seed=seed,coding=coding,train_events_per_minute=float(ev[tri].sum()/len(tri)),
                train_active_minute_fraction=float(ev[tri].any((1,2)).mean()),test_events_per_minute=float(ev[ti].sum()/len(ti))))
            for pool in ['random','stdp']:
                m=ContinuousSNN(cfg,seed);initial=m.weight.copy()
                if coding=='crossing':
                    for epoch in range(cfg['epochs']):
                        _,_,train_s=stream(m,ev,train,learn=pool=='stdp',adapt=True,capture=True,total_rows=len(bars))
                        epochrows.append(dict(seed=seed,pool=pool,epoch=epoch+1,spikes=int(train_s[tri].sum()),
                            active_neurons=int((train_s[tri].sum((0,1))>0).sum()),weight_change=float(np.linalg.norm(m.weight-initial))))
                    m.save(out/f'{coding}_{pool}_{seed}_network.npz')
                else:m.load(ROOT/f'results/first_fold/fold_00/{pool}_{seed}_network.npz')
                w=m.weight.copy();theta=m.theta.copy()
                _,_,s=stream(m,ev,evaluate,capture=True,total_rows=len(bars))
                assert np.array_equal(w,m.weight) and np.array_equal(theta,m.theta)
                if pool=='random':assert np.array_equal(w,initial)
                f=s[ix].reshape(len(ends),-1);path=out/f'{coding}_{pool}_{seed}_readout.npz'
                pred=fit_predict([f],cal,test,y,path)
                assert np.allclose(pred,apply_saved(path,[f[test]]),atol=1e-12,rtol=0)
                # Single isolated input from rest, holding this model's threshold fixed.
                isolated=[]
                for ch in range(24):
                    m.reset_state();probe=np.zeros((8,24),bool);probe[0,ch]=1
                    isolated.append(int(m.minute(probe)[0].sum()))
                # Complete segment replay after reset must be deterministic.
                _,_,repeat=stream(m,ev,[tg[0]],capture=True,total_rows=len(bars));assert np.array_equal(repeat[tg[0]],s[tg[0]])
                if coding=='crossing':
                    reloaded=ContinuousSNN(cfg,seed).load(out/f'{coding}_{pool}_{seed}_network.npz')
                    _,_,again=stream(reloaded,ev,[tg[0]],capture=True,total_rows=len(bars));assert np.array_equal(again[tg[0]],s[tg[0]])
                diagnostics.append(dict(coding=coding,pool=pool,seed=seed,test_spikes_per_minute=float(s[ti].sum()/len(ti)),
                    test_active_minute_fraction=float(s[ti].any((1,2)).mean()),active_test_neurons=int((s[ti].sum((0,1))>0).sum()),
                    weight_change_l2=float(np.linalg.norm(w-initial)),theta_mean=float(theta.mean()),
                    isolated_channels_producing_output=int((np.array(isolated)>0).sum()),prediction_std=float(pred.std())))
                rows.append(pd.DataFrame(dict(timestamp=meta.datetime[test].astype(str).to_numpy(),date=meta.trading_date[test].to_numpy(),
                    coding=coding,pool=pool,seed=seed,prediction=pred,actual_abs_1m_bp=y[test])))
                np.savez_compressed(out/f'{coding}_{pool}_{seed}_test_trace.npz',bar_ids=ti,spikes=s[ti])
                print(coding,pool,seed,'done',round(time.monotonic()-started,1),flush=True)
    d=pd.concat(rows,ignore_index=True);d.to_csv(out/'seed_predictions.csv',index=False)
    ens=d.groupby(['coding','pool','timestamp','date'],sort=False)[['prediction','actual_abs_1m_bp']].mean().reset_index();ens.to_csv(out/'minute_predictions.csv',index=False)
    metrics=[]
    for (coding,pool),g in ens.groupby(['coding','pool']):
        g=g.dropna();error=g.prediction-g.actual_abs_1m_bp
        metrics.append(dict(coding=coding,pool=pool,points=len(g),rmse_bp=float(np.sqrt(np.mean(error**2))),mae_bp=float(abs(error).mean())))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'network_diagnostics.csv',index=False)
    pd.DataFrame(inputs).to_csv(out/'input_diagnostics.csv',index=False)
    pd.DataFrame(epochrows).to_csv(out/'training_diagnostics.csv',index=False)
    comparisons=[];wide=ens.pivot(index=['timestamp','date'],columns=['coding','pool'],values='prediction');label=ens.groupby(['timestamp','date']).actual_abs_1m_bp.first()
    for labelname,reference in [('crossing_random',('crossing','random')),('poisson_stdp',('poisson','stdp'))]:
        delta=(wide['crossing','stdp']-label)**2-(wide[reference]-label)**2
        z=delta.dropna().groupby(level='date').agg(['sum','count']);idx=np.random.default_rng(501).integers(0,len(z),(2000,len(z)))
        bs=z['sum'].to_numpy()[idx].sum(1)/z['count'].to_numpy()[idx].sum(1);lo,hi=np.quantile(bs,[.025,.975])
        comparisons.append(dict(comparison='crossing_stdp minus '+labelname,mse_delta_bp2=float(delta.mean()),low=lo,high=hi))
    pd.DataFrame(comparisons).to_csv(out/'paired_mse.csv',index=False)
    files=[Path(__file__),ROOT/'src/crossing_encoding.py',ROOT/'src/network.py',ROOT/'src/stdp.py',ROOT/'audit_utils.py']
    (out/'verification.json').write_text(json.dumps(dict(complete=True,new_models=6,reused_models=6,test_minutes=int(test.sum()),labeled_minutes=int(np.isfinite(y[test]).sum()),
        deterministic_encoding_and_prefix=True,replay_and_reload_exact=True,weights_and_theta_frozen=True,readout_raw_features=0,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2),encoding='utf-8')

if __name__=='__main__':main()
