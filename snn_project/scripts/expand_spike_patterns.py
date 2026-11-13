"""Budgeted frequent pattern expansion and time-aligned parent/child comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mine_local_spike_patterns import MAX_LAG, MIN_SUPPORT, PAIR_SEEDS, item_matrix, streams
from snn_core.market_data import load_sessions

NEURONS = 216
MAX_SIZE = 216
BEAM = 24
MAX_EXPANDED = 240


def seeds(items, max_lag):
    values = items.astype(np.float32)
    result = []
    for lag in range(1, max_lag + 1):
        count = values[:, :NEURONS].T @ values[:, lag*NEURONS:(lag+1)*NEURONS]
        np.fill_diagonal(count, 0)
        coords = np.argwhere(count >= MIN_SUPPORT)
        order = np.argsort(-count[tuple(coords.T)], kind='stable')[:PAIR_SEEDS]
        result.extend((int(a), int(lag*NEURONS+b)) for a,b in coords[order])
    return result


def discover(items, max_lag=MAX_LAG, max_expanded=MAX_EXPANDED):
    frontier = seeds(items, max_lag)
    stats, edges = [], []
    visited = set()
    expanded = 0
    # Each edge records a refinement before the size/beam budget is applied.
    while frontier and expanded < max_expanded:
        candidates = {}
        queried = frequent = closures = 0
        levels = [len(p) for p in frontier]
        for parent in frontier[:max_expanded-expanded]:
            if parent in visited or len(parent) == MAX_SIZE:
                continue
            visited.add(parent)
            expanded += 1
            hits = items[:,parent].all(axis=1)
            support = int(hits.sum())
            count = items[hits].sum(axis=0)
            extras = count >= MIN_SUPPORT
            extras[list(parent)] = False
            queried += items.shape[1]-len(parent)
            frequent += int(extras.sum())
            equal = np.flatnonzero(extras & (count == support))
            if len(equal):
                closure = tuple(sorted((*parent,*map(int,equal))))
                if len(closure) <= MAX_SIZE:
                    closures += 1
                    candidates[closure] = (support, parent)
                    continue
            for item in np.flatnonzero(extras):
                child = tuple(sorted((*parent,int(item))))
                if child not in visited:
                    candidates.setdefault(child,(int(count[item]),parent))
        # Keep representative branches by training frequency only.
        ordered = sorted(candidates, key=lambda p:(-candidates[p][0], -len(p), p))
        kept = ordered[:BEAM]
        for child in kept:
            support,parent = candidates[child]
            edges.append((parent,child,support))
        stats.append(dict(round=len(stats)+1,min_parent_size=min(levels),max_parent_size=max(levels),
                          expanded_total=expanded,extensions_checked=queried,frequent_extensions=frequent,
                          closure_jumps=closures,distinct_candidates=len(candidates),
                          kept=len(kept),beam_discarded=len(candidates)-len(kept),
                          largest_candidate=max(map(len,candidates),default=0)))
        frontier = kept
    return edges,stats,('expansion_budget' if frontier else 'frontier_exhausted')


def fmt(pattern):
    return ' '.join(f'N{i%NEURONS}@{i//NEURONS}' for i in pattern)


def day_interval(a,b,days):
    unique, group = np.unique(days,return_inverse=True)
    if len(unique)<5 or not (a.any() and b.any()):
        return np.nan,np.nan
    counts = np.array([np.bincount(group,weights=x,minlength=len(unique)) for x in (a[:,0],b[:,0],a[:,1],b[:,1])])
    draws = np.random.default_rng(43).integers(len(unique),size=(400,len(unique)))
    sc,sp,nc,np_ = counts[:,draws].sum(axis=2)
    valid=(nc>0)&(np_>0)
    if not valid.any():
        return np.nan,np.nan
    return np.quantile(sc[valid]/nc[valid]-sp[valid]/np_[valid],[.025,.975])


def main(output):
    data=Path('data/ic_holdout_input.csv')
    sessions=list(streams(data,Path('experiments/segmented_v1/reservoir.npz')))
    dates=np.concatenate([[t[:10] for t in ts] for ts,_,_ in sessions])
    split=np.concatenate([[s]*len(ts) for ts,s,_ in sessions])
    steps=np.concatenate([np.arange(len(ts)) for ts,_,_ in sessions])
    remaining=np.concatenate([np.arange(len(ts)-1,-1,-1) for ts,_,_ in sessions])
    train=split=='train'
    frozen, search, stopping = {}, [], {}
    for name in ('random','stdp'):
        items=item_matrix([rasters[name] for _,_,rasters in sessions])
        edges,stats,reason=discover(items[train])
        frozen[name]=(items,edges)
        search.extend(dict(pool=name,**row) for row in stats)
        stopping[name]=reason
        print(name,'frozen edges',len(edges),'largest',max((len(c) for _,c,_ in edges),default=0),reason,flush=True)

    # Only now compute return labels. Row k corresponds to bar k+1.
    returns=[]
    for (_,bars),(stamps,_,_) in zip(load_sessions(data),sessions,strict=True):
        assert len(bars)==len(stamps)+1
        values=np.full(len(stamps),np.nan)
        values[:-5]=1e4*np.log(bars[6:,3]/bars[1:-5,3])
        returns.append(values)
    y=np.concatenate(returns)
    coverage, comparison, patterns=[],[],[]
    for pool,(items,edges) in frozen.items():
        for i,(parent,child,support) in enumerate(edges,1):
            lag=max(child)//NEURONS
            p=items[:,parent].all(axis=1)&(remaining>=lag)
            c=items[:,child].all(axis=1)
            assert np.all(p[c])
            assert c[train].sum()==support
            patterns.append(dict(pool=pool,edge=i,parent=fmt(parent),child=fmt(child),
                                 parent_size=len(parent),child_size=len(child),decision_lag=lag))
            for stage in ('train','validation','diagnostic'):
                mask=split==stage
                parents=np.flatnonzero(p&mask)
                children=np.flatnonzero(c&mask)
                coverage.append(dict(pool=pool,edge=i,split=stage,parent_size=len(parent),child_size=len(child),
                                     parent_hits=len(parents),child_hits=len(children),
                                     retention=len(children)/len(parents) if len(parents) else np.nan,
                                     child_days=len(set(dates[children])),
                                     child_first_10_fraction=np.mean(steps[children]+lag<10) if len(children) else np.nan))
                anchors=parents[remaining[parents]>=lag+5]
                labels=y[anchors+lag]
                assert np.isfinite(labels).all()
                present=c[anchors]
                absent=~present
                cm=labels[present].mean() if present.any() else np.nan
                pm=labels[absent].mean() if absent.any() else np.nan
                lo,hi=day_interval(np.column_stack((labels*present,present)),
                                   np.column_stack((labels*absent,absent)),dates[anchors])
                comparison.append(dict(pool=pool,edge=i,split=stage,child_size=len(child),
                                       child_n=int(present.sum()),parent_only_n=int(absent.sum()),
                                       child_days=len(set(dates[anchors[present]])),
                                       parent_only_days=len(set(dates[anchors[absent]])),
                                       child_mean_bp=cm,parent_only_mean_bp=pm,delta_bp=cm-pm,
                                       exploratory_ci_low=lo,exploratory_ci_high=hi))
    output.mkdir(parents=True)
    for name,rows in [('search_progress',search),('patterns',patterns),('trigger_retention',coverage),
                      ('aligned_return_comparison',comparison)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(beam=BEAM,max_expanded=MAX_EXPANDED,
        max_size=MAX_SIZE,min_support=MIN_SUPPORT,stopping=stopping),indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

