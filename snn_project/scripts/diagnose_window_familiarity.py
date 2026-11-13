"""Fixed-sample descriptive audit; cross-fitted residuals are NOT live signals.

No reservoir retraining, label selection, horizon change or parameter sweep.
Only numpy/pandas are required. Run from snn_project.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from walk_forward_spike_readout import load_data


def design(train, query, columns):
    """Train-only linear splines and pairwise linear interactions."""
    a, b, za, zb = [], [], [], []
    for col in columns:
        x, q = train[col].to_numpy(), query[col].to_numpy()
        mean, sd = x.mean(), max(x.std(), 1e-8)
        x, q = (x-mean)/sd, (q-mean)/sd
        za.append(x); zb.append(q)
        a.append(x); b.append(q)
        for knot in np.quantile(x, [.1,.25,.5,.75,.9]):
            a.append(np.maximum(x-knot,0)); b.append(np.maximum(q-knot,0))
    for i in range(len(za)):
        for j in range(i):
            a.append(za[i]*za[j]); b.append(zb[i]*zb[j])
    if not a:
        return np.zeros((len(train),0)),np.zeros((len(query),0))
    return np.column_stack(a),np.column_stack(b)


def fit_predict(train, query, columns, targets):
    x,q = design(train,query,columns)
    y = train[targets].to_numpy()
    clocks = sorted(train.clock.unique())
    mapping = {c:i for i,c in enumerate(clocks)}
    ti = train.clock.map(mapping).to_numpy()
    qi = query.clock.map(mapping).to_numpy()
    assert np.isfinite(qi).all()
    mx=np.array([x[ti==i].mean(axis=0) for i in range(len(clocks))])
    my=np.array([y[ti==i].mean(axis=0) for i in range(len(clocks))])
    if not columns:
        return my[qi]
    xc,yc = x-mx[ti],y-my[ti]
    coef=np.linalg.solve(xc.T@xc + np.eye(x.shape[1])*len(x)*.001,xc.T@yc)
    return my[qi]+(q-mx[qi])@coef


def effect_interval(x,y,days):
    # Fixed cross-fitted nuisance fits; day-cluster bootstrap, exploratory only.
    ids=np.unique(days)
    numer=np.array([np.sum(x[days==d]*y[days==d]) for d in ids])
    denom=np.array([np.sum(x[days==d]**2) for d in ids])
    draws=np.random.default_rng(20261110).integers(0,len(ids),(2000,len(ids)))
    vals=.1*numer[draws].sum(axis=1)/denom[draws].sum(axis=1)
    return .1*numer.sum()/denom.sum(),*np.quantile(vals,[.025,.975])


def main(source, output):
    output.mkdir(parents=True,exist_ok=False)
    d=pd.read_csv(source)
    assert len(d)==13800 and not d.timestamp.duplicated().any()
    d['date']=d.timestamp.str[:10]; d['clock']=d.timestamp.str[11:16]
    # Independently verify bar alignment and label/session boundaries.
    sessions,_,_=load_data()
    market={}
    for times,bars,_ in sessions:
        for i in range(5,len(bars)):
            key=times[i].isoformat(sep=' ')
            y=1e4*np.log(bars[i+1,3]/bars[i,3]) if i+1<len(bars) else np.nan
            r=1e4*np.log(bars[i-4:i+1,1].max()/bars[i-4:i+1,2].min())
            market[key]=(i,y,r)
    verified=np.array([market[t] for t in d.timestamp])
    assert np.allclose(verified[:,1],d.actual_1m_bp,equal_nan=True)
    assert np.allclose(verified[:,2],d.past_5m_range_bp)
    d['session_bar_index']=verified[:,0].astype(int)
    d['abs_1m_bp']=d.actual_1m_bp.abs()
    d['log_range']=np.log1p(d.past_5m_range_bp)
    summaries, effects, matches, profiles, residuals=[],[],[],[],[]
    stages={'clock':[], 'clock_range':['log_range'],
            'clock_range_density':['log_range','random_firing_fraction','stdp_firing_fraction']}
    targets=['random_familiarity','stdp_familiarity','abs_1m_bp']
    for fold,g0 in d.groupby('fold'):
        g=g0.dropna(subset=['abs_1m_bp']).copy().reset_index(drop=True)
        date_ids={day:i for i,day in enumerate(sorted(g.date.unique()))}
        block=g.date.map(date_ids).to_numpy()//4  # five contiguous four-day holdouts
        for stage,columns in stages.items():
            pred=np.empty((len(g),3))
            for k in range(5):
                mask=block==k
                pred[mask]=fit_predict(g[~mask],g[mask],columns,targets)
            rem=g[targets].to_numpy()-pred
            for j,target in enumerate(targets):
                y=g[target].to_numpy()
                summaries.append(dict(fold=fold,stage=stage,target=target,
                    crossfit_r2=1-np.sum(rem[:,j]**2)/np.sum((y-y.mean())**2)))
            for j,pool in enumerate(['random','stdp']):
                val,lo,hi=effect_interval(rem[:,j],rem[:,2],g.date.to_numpy())
                effects.append(dict(fold=fold,stage=stage,pool=pool,
                    bp_per_point1_residual_score=val,low95=lo,high95=hi,
                    residual_score_sd=rem[:,j].std(),minutes=len(g),days=g.date.nunique()))
            if stage=='clock_range_density':
                r=g[['fold','timestamp']].copy()
                for j,target in enumerate(targets):
                    r[target+'_diagnostic_residual']=rem[:,j]
                residuals.append(r)
        # Does STDP familiarity add to random familiarity, conditionally?
        columns=stages['clock_range_density']+['random_familiarity']
        pred=np.empty((len(g),2))
        for k in range(5):
            mask=block==k
            pred[mask]=fit_predict(g[~mask],g[mask],columns,targets[1:])
        rem=g[targets[1:]].to_numpy()-pred
        val,lo,hi=effect_interval(rem[:,0],rem[:,1],g.date.to_numpy())
        effects.append(dict(fold=fold,stage='clock_range_density_random_score',pool='stdp',
            bp_per_point1_residual_score=val,low95=lo,high95=hi,
            residual_score_sd=rem[:,0].std(),minutes=len(g),days=g.date.nunique()))
        for pool in ['random','stdp']:
            col=pool+'_familiarity'
            neighbor=pd.to_datetime(g[pool+'_nearest_training_time'])
            same=neighbor.dt.strftime('%H:%M').to_numpy()==g.clock.to_numpy()
            for label,mask in [('all',np.ones(len(g),bool)),('first_15_bars',g.session_bar_index<=14),('later',g.session_bar_index>14)]:
                q=g[mask]
                profiles.append(dict(fold=fold,pool=pool,phase=label,minutes=len(q),
                    mean_familiarity=q[col].mean(),top20_share=(q[col]>=.8).mean(),
                    same_clock_neighbor_share=same[mask].mean(),mean_density=q[pool+'_firing_fraction'].mean()))
            # Exact same clock, unlike score, similar nuisance ranks. No labels used to match.
            nuis=['log_range','random_firing_fraction','stdp_firing_fraction']
            ranks=g[nuis].rank(pct=True).to_numpy()
            for i in np.flatnonzero(g[col].to_numpy()>=.8):
                candidates=np.flatnonzero((g.clock==g.clock[i]) & (g[col]<=.2))
                if not len(candidates): continue
                dif=np.abs(ranks[candidates]-ranks[i])
                ok=(dif<=.15).all(axis=1)
                if not ok.any(): continue
                candidates=candidates[ok]; dif=dif[ok]
                j=candidates[np.argmin(np.sum(dif**2,axis=1))]
                matches.append(dict(fold=fold,pool=pool,high_time=g.timestamp[i],low_time=g.timestamp[j],
                    high_score=g[col][i],low_score=g[col][j],high_range=g.past_5m_range_bp[i],
                    low_range=g.past_5m_range_bp[j],high_abs_bp=g.abs_1m_bp[i],low_abs_bp=g.abs_1m_bp[j],
                    max_nuisance_percentile_gap=np.max(np.abs(ranks[i]-ranks[j]))))
        print('diagnosed fold',fold,flush=True)
    pd.DataFrame(summaries).to_csv(output/'conditional_explanation.csv',index=False)
    pd.DataFrame(effects).to_csv(output/'conditional_future_effect.csv',index=False)
    pd.DataFrame(profiles).to_csv(output/'startup_profiles.csv',index=False)
    pd.DataFrame(matches).to_csv(output/'same_clock_matched_pairs.csv',index=False)
    pd.concat(residuals).to_csv(output/'diagnostic_residuals_NOT_LIVE.csv',index=False)
    # Preserve causal existing signals; no retrospective residual is mixed into these.
    d.to_csv(output/'minute_signals_with_session_audit.csv',index=False)
    audit=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               rows=len(d),missing_labels=int(d.actual_1m_bp.isna().sum()),
               verified='all labels/ranges rechecked from source bars; no cross-session labels',
               method='5 contiguous four-day cross-fit blocks within each historical test fold; exact clock fixed effects plus train-only linear splines, pairwise interactions; ridge 0.001; no tuning',
               inference='2000 day-cluster bootstrap draws with nuisance fits fixed; descriptive intervals, not final hypothesis tests',
               limitations='Same clock confounds clock with startup age. Controls may mediate STDP effects. Existing score only; not a test of all raster information. Diagnostic fits use other test days including later days, so residuals are NOT deployable.',
               matches='high >=0.8 vs low <=0.2, exact clock, each nuisance rank gap <=0.15; nearest nuisance distance; low windows may be reused; descriptive only')
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('experiments/whole_window_familiarity/whole_window_signals.csv'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.source,args.output)
