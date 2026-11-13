"""Mine and compare decision-aligned five-minute IC spike motifs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expand_spike_patterns import BEAM, MAX_EXPANDED, NEURONS, day_interval, discover
from mine_local_spike_patterns import MAX_LAG, MIN_SUPPORT, draw, streams
from snn_core.market_data import load_sessions

STAGES = ('train', 'validation', 'diagnostic')


def key_at(structure, age):
    last = max(structure) // NEURONS
    return tuple(i + (MAX_LAG - age - last) * NEURONS for i in structure)


def describe(key):
    return ' '.join(f'N{i % NEURONS}@{i // NEURONS - MAX_LAG:+d}' for i in key)


def main(output):
    data = Path('data/ic_holdout_input.csv')
    sessions = list(streams(data, Path('experiments/segmented_v1/reservoir.npz')))
    timestamps, stages, session_steps = [], [], []
    for ts, stage, _ in sessions:
        timestamps.extend(ts[MAX_LAG:])
        stages.extend([stage] * (len(ts)-MAX_LAG))
        session_steps.extend(range(MAX_LAG, len(ts)))
    timestamps, stages, session_steps = map(np.asarray, (timestamps, stages, session_steps))
    dates = timestamps.astype('U10')
    train = stages == 'train'
    frozen, progress, protocol = {}, [], {}
    for pool in ('random', 'stdp'):
        windows = np.concatenate([
            np.lib.stride_tricks.sliding_window_view(rasters[pool], MAX_LAG+1, axis=0).transpose(0,2,1)
            for _,_,rasters in sessions
        ])
        items = windows.reshape(len(windows), -1)
        # Mine shapes on the five-step grid, then count each valid absolute placement separately.
        half = MAX_EXPANDED // 2
        general, general_stats, general_stop = discover(items[train], max_expanded=half)
        short, short_stats, short_stop = discover(items[train, :3*NEURONS], max_lag=2, max_expanded=half)
        shapes = sorted({child for _, child, _ in general + short})
        variants = []
        for i, shape in enumerate(shapes, 1):
            for age in range(MAX_LAG - max(shape)//NEURONS + 1):
                key = key_at(shape, age)
                hits = items[:, key].all(axis=1)
                if hits[train].sum() >= MIN_SUPPORT:
                    variants.append((i, age, key, hits))
        frozen[pool] = (windows, variants)
        progress.extend(dict(pool=pool, search=kind, **row)
                        for kind, stats in (('general', general_stats), ('short', short_stats)) for row in stats)
        protocol[pool] = {'general_stop': general_stop, 'short_stop': short_stop,
                          'shapes': len(shapes), 'fixed_patterns': len(variants)}
        print(pool, protocol[pool], flush=True)

    # Returns enter only after both motif catalogs and all trigger masks are fixed.
    returns = []
    for (_, bars), (ts, _, _) in zip(load_sessions(data), sessions, strict=True):
        assert len(bars) == len(ts)+1
        values = np.full(len(ts), np.nan)
        values[:-5] = 1e4*np.log(bars[6:,3]/bars[1:-5,3])
        returns.extend(values[MAX_LAG:])
    y = np.asarray(returns)
    output.mkdir(parents=True)
    patterns, summary, coverage, comparisons, occurrences = [], [], [], [], []
    for pool, (windows, variants) in frozen.items():
        by_shape, any_hit = {}, np.zeros(len(y), dtype=bool)
        for shape, age, key, hits in variants:
            by_shape.setdefault(shape,{})[age] = (key,hits)
            any_hit |= hits
            patterns.append(dict(pool=pool,shape=shape,age=age,spikes=len(key),items=describe(key)))
            for i in np.flatnonzero(hits):
                occurrences.append((pool,shape,age,timestamps[i],stages[i]))
            for stage in STAGES:
                rows = np.flatnonzero(hits & (stages==stage))
                summary.append(dict(pool=pool,shape=shape,age=age,split=stage,minutes=len(rows),
                                    days=len(set(dates[rows])),clocks=len({t[11:16] for t in timestamps[rows]}),
                                    coverage=len(rows)/(stages==stage).sum(),
                                    first_10_minutes=int((session_steps[rows]<10).sum())))
        for stage in STAGES:
            mask = stages==stage
            coverage.append(dict(pool=pool,split=stage,eligible_minutes=int(mask.sum()),
                                 matched=int((mask & any_hit).sum()),coverage=any_hit[mask].mean()))
        for shape, ages in by_shape.items():
            if not {0,2}.issubset(ages):
                continue
            key, recent = ages[0]
            _, older = ages[2]
            for stage in STAGES:
                eligible = (stages==stage) & np.isfinite(y) & (recent | older)
                idx = np.flatnonzero(eligible)
                # Keep all trigger minutes; count simultaneous matches in both groups and report overlap.
                a,b = recent[idx],older[idx]
                values=y[idx]
                ma=values[a].mean() if a.any() else np.nan
                mb=values[b].mean() if b.any() else np.nan
                lo,hi=day_interval(np.column_stack((values*a,a)),np.column_stack((values*b,b)),dates[idx])
                comparisons.append(dict(pool=pool,shape=shape,split=stage,spikes=len(key),
                    age0_n=int(a.sum()),age2_n=int(b.sum()),both=int((a&b).sum()),
                    age0_days=len(set(dates[idx[a]])),age2_days=len(set(dates[idx[b]])),
                    age0_bp=ma,age2_bp=mb,delta_bp=ma-mb,ci_low=lo,ci_high=hi))
        candidates = [(shape,age,key,hits) for shape,age,key,hits in variants if age==0]
        candidates.sort(key=lambda row:-int(row[3][train].sum()))
        # Pick two distinct shapes and show two fixed placements when available, before considering returns.
        picked=[row for row in candidates if 2 in by_shape[row[0]]][:2] or candidates[:2]
        for shape,_,_,_ in picked:
            for age in (0,2):
                if age not in by_shape[shape]:
                    continue
                key,hits=by_shape[shape][age]
                examples, clocks, seen_days = [],set(),set()
                for stage in STAGES:
                    rows=np.flatnonzero(hits & (stages==stage))
                    if len(rows):
                        distinct=[i for i in rows if timestamps[i][11:16] not in clocks and dates[i] not in seen_days]
                        i=distinct[0] if distinct else rows[0]
                        clocks.add(timestamps[i][11:16])
                        seen_days.add(dates[i])
                        examples.append((windows[i],timestamps[i],key))
                draw(output/f'{pool}_shape_{shape}_age_{age}.png',f'{pool.upper()} #{shape} | {describe(key)}',
                     examples,NEURONS)
    for name, rows in [('patterns',patterns),('coverage',coverage),('pattern_summary',summary),
                       ('position_return_comparison',comparisons),('search_progress',progress)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    pd.DataFrame(occurrences,columns=['pool','shape','age','decision','split']).to_csv(output/'occurrences.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(beam=BEAM,max_expanded=MAX_EXPANDED,
        min_support=MIN_SUPPORT,pools=protocol),indent=2),encoding='utf8')
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)


