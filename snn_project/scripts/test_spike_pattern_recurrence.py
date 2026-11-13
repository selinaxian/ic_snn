"""Fixed-motif recurrence test against independent neuron day permutations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mine_local_spike_patterns import streams

DRAWS = 999
SEED = 43


def adjust_bh(p):
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def pattern_items(label):
    return [(int(neuron[1:]), int(lag)+4)
            for neuron, lag in (item.split('@') for item in label.split())]


def counts_for_pattern(packed, permutations, items):
    # Same permutation for a neuron's entire trajectory, including all its motif lags.
    bits = None
    for neuron, lag in items:
        selected = packed[neuron, lag, permutations[neuron]]
        if bits is None:
            bits = selected.copy()
        else:
            bits &= selected
    return np.bitwise_count(bits).sum(axis=(1,2,3))


def main(output):
    base = Path('experiments/decision_aligned_position_test')
    catalog = pd.read_csv(base/'patterns.csv')
    observed = pd.read_csv(base/'pattern_summary.csv').set_index(['pool','shape','age','split'])
    sessions = list(streams(Path('data/ic_holdout_input.csv'),Path('experiments/segmented_v1/reservoir.npz')))
    rows = []
    for split in ('validation','diagnostic'):
        day_map = {}
        for times, stage, rasters in sessions:
            if stage == split:
                day_map.setdefault(times[0][:10], []).append((tuple(t[11:16] for t in times),rasters))
        days = list(day_map.values())
        clocks = [ts for ts,_ in days[0]]
        assert len(clocks)==2 and all([ts for ts,_ in day] == clocks for day in days)
        n = len(days)
        rng = np.random.default_rng(SEED)
        permutations = rng.random((216,DRAWS,n)).argsort(axis=2)
        permutations = np.concatenate((np.broadcast_to(np.arange(n),(216,1,n)),permutations),axis=1)
        for pool, patterns in catalog.groupby('pool'):
            spikes = np.stack([[rasters[pool] for _,rasters in day] for day in days])
            # neuron x relative minute x day x session x packed decision time
            windows = np.stack([spikes[:,:,lag:lag+spikes.shape[2]-4,:] for lag in range(5)])
            packed = np.packbits(windows.transpose(4,0,1,2,3),axis=-1)
            for row in patterns.itertuples(index=False):
                counts = counts_for_pattern(packed,permutations,pattern_items(row.items))
                actual = int(counts[0])
                assert actual == observed.loc[(pool,row.shape,row.age,split),'minutes']
                null = counts[1:]
                lo,hi = np.quantile(null,[.025,.975])
                mean = float(null.mean())
                rows.append(dict(pool=pool,shape=row.shape,age=row.age,items=row.items,split=split,
                    days=n,observed=actual,coverage=actual/(n*2*(spikes.shape[2]-4)),
                    null_mean=mean,null_025=lo,null_975=hi,excess=actual-mean,
                    lift=actual/mean if mean else np.nan,p=(1+(null>=actual).sum())/(DRAWS+1)))
            print(split,pool,len(patterns),flush=True)
    result=pd.DataFrame(rows)
    result['q_bh']=result.groupby('split').p.transform(lambda p: adjust_bh(p.to_numpy()))
    output.mkdir(parents=True)
    result.to_csv(output/'recurrence_tests.csv',index=False)
    result[result.pool.eq('stdp') & result['shape'].isin([35,37,38,43]) & result.age.isin([0,2])].to_csv(
        output/'return_candidates.csv',index=False)
    print(result.groupby(['split','pool']).agg(tests=('p','size'),significant=('q_bh',lambda x:(x<=.05).sum()),
        median_lift=('lift','median')).to_string())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
