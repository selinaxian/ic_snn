"""Evaluate four frozen STDP structures on ICL8 days after 2026-09-01."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mine_local_spike_patterns import streams
from snn_core.market_data import load_sessions

FIELDS = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
CUTOFF = '2026-09-01'
SHAPES = (35, 37, 38, 43)


def evaluate(table):
    days, day = np.unique(table.day, return_inverse=True)
    _, clock = np.unique(table.clock, return_inverse=True)
    y = table.actual_bp.to_numpy()
    now, late = table.age0.to_numpy(), table.age2.to_numpy()
    masks = [now, late, ~now]
    shape = (len(days), clock.max()+1)
    sums, counts = np.zeros((3,*shape)), np.zeros((3,*shape))
    for i, mask in enumerate(masks):
        np.add.at(sums[i], (day[mask],clock[mask]), y[mask])
        np.add.at(counts[i], (day[mask],clock[mask]), 1)

    def differences(weight):
        s, c = np.einsum('d,gdc->gc',weight,sums), np.einsum('d,gdc->gc',weight,counts)

        means = np.divide(s.sum(axis=1), c.sum(axis=1), out=np.full(3,np.nan), where=c.sum(axis=1)>0)
        comparable = c[2]>0
        control = s[2,comparable]/c[2,comparable]
        excess = (s[0,comparable]-control*c[0,comparable]).sum()/c[0,comparable].sum() if c[0,comparable].sum() else np.nan
        return means[0]-means[1], excess, int(c[0,comparable].sum())

    delta, excess, matched = differences(np.ones(len(days)))
    rng = np.random.default_rng(43)
    samples = np.array([differences(np.bincount(idx,minlength=len(days)))[:2]
                        for idx in rng.integers(len(days),size=(1000,len(days)))])
    ci = [np.nanquantile(a,[.025,.975]).tolist() if np.isfinite(a).any() else [None,None] for a in samples.T]
    return dict(age0_n=int(now.sum()),age2_n=int(late.sum()),both_n=int((now&late).sum()),
                age0_days=int(table.loc[now,'day'].nunique()),age2_days=int(table.loc[late,'day'].nunique()),
                age0_mean_bp=float(y[now].mean()),age2_mean_bp=float(y[late].mean()),
                age0_minus_age2_bp=delta,age0_minus_nontrigger_same_clock_bp=excess,
                age0_with_clock_control=matched,position_ci=ci[0],same_clock_ci=ci[1])


def main(parquet, output):
    old = pd.read_csv('data/ic_holdout_input.csv',usecols=['date',*FIELDS]).set_index('date')
    source = pd.read_parquet(parquet,columns=['datetime',*FIELDS])
    source['date'] = source.pop('datetime').dt.strftime('%Y-%m-%d %H:%M:%S')
    source = source.set_index('date').sort_index()
    assert source.index.is_unique
    overlap=old.index.intersection(source.index)
    assert len(overlap)==len(old)
    np.testing.assert_allclose(old.loc[overlap,FIELDS],source.loc[overlap,FIELDS],rtol=0,atol=1e-8)
    fresh = source[source.index.str[:10]>CUTOFF]
    assert not fresh.empty
    for day, bars in fresh.groupby(fresh.index.str[:10]):
        grid = pd.date_range(day+' 09:31',day+' 11:30',freq='min').append(
            pd.date_range(day+' 13:01',day+' 15:00',freq='min'))
        np.testing.assert_array_equal(bars.index.to_numpy(),grid.strftime('%Y-%m-%d %H:%M:%S').to_numpy())

    catalog=pd.read_csv('experiments/decision_aligned_position_test/patterns.csv')
    frozen=catalog[(catalog.pool=='stdp') & catalog['shape'].isin(SHAPES) & catalog.age.isin([0,2])]
    assert len(frozen)==8
    output.mkdir(parents=True)
    input_file=output/'new_minutes.csv'
    fresh.to_csv(input_file)
    frozen.to_csv(output/'frozen_patterns.csv',index=False)
    tables=[]
    for (timestamps,_,rasters),(_, bars) in zip(streams(input_file,Path('experiments/segmented_v1/reservoir.npz')),
                   load_sessions(input_file),strict=True):
        z=rasters['stdp']
        end=np.arange(4,len(z)-5)
        values=1e4*np.log(bars[end+6,3]/bars[end+1,3])
        stamps=np.asarray(timestamps)[end]
        for shape, motifs in frozen.groupby('shape'):
            row=dict(shape=shape,timestamp=stamps,day=stamps.astype('U10'),
                     clock=[t[11:16] for t in stamps],actual_bp=values)
            for motif in motifs.itertuples():
                bits=[(int(neuron[1:]),int(offset)) for neuron,offset in (x.split('@') for x in motif.items.split())]
                row[f'age{motif.age}']=np.logical_and.reduce([z[end+lag,neuron] for neuron,lag in bits])
            tables.append(pd.DataFrame(row))
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(output/'minute_outcomes.csv',index=False)
    result={str(shape):evaluate(group) for shape,group in table.groupby('shape')}
    result={'first_day':fresh.index[0][:10],'last_day':fresh.index[-1][:10],
            'days':int(table.day.nunique()),'minutes_per_shape':len(table)//len(SHAPES),
            'verified_prior_bars':len(overlap),'input_sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),
            'patterns':result}
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parquet',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.parquet,args.output)


