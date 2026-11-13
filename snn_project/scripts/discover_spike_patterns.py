"""Discover recurring IC spike rasters without consulting future labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from run_diagnostics import load_features
from snn_core.market_data import load_sessions

THRESHOLD = 0.40
MIN_DAYS = 5
MAX_PATTERNS = 6


def jaccard(x, reference):
    dot = x @ reference.T
    return dot / np.maximum(x.sum(axis=1)[:, None] + reference.sum(axis=1) - dot, 1)


def discover(x, train, days):
    train_ids = np.flatnonzero(train)
    # Thin only prototype candidates; all training windows can support a pattern.
    seeds = train_ids[np.arange(len(train_ids)) % 5 == 0]
    reference = x[train]
    day_ids = pd.factorize(days[train])[0]
    support = np.zeros(len(seeds), dtype=int)
    for start in range(0, len(seeds), 64):
        batch = seeds[start:start+64]
        similarities = jaccard(x[batch], reference)
        for i, seed in enumerate(batch):
            eligible = (similarities[i] >= THRESHOLD) & (days[train] != days[seed])
            support[start+i] = len(np.unique(day_ids[eligible]))
    chosen = []
    for idx in np.argsort(-support, kind='stable'):
        if support[idx] < MIN_DAYS - 1 or len(chosen) == MAX_PATTERNS:
            break
        seed = seeds[idx]
        if chosen and np.any(jaccard(x[[seed]], x[chosen]) >= THRESHOLD):
            continue
        chosen.append(seed)
    if not chosen:
        return np.empty(0, dtype=int), np.full(len(x), -1), np.full(len(x), np.nan)
    similarities = jaccard(x, x[chosen])
    assignment = similarities.argmax(axis=1)
    best = similarities[np.arange(len(x)), assignment]
    assignment[best < THRESHOLD] = -1
    # Require recurring support after the exclusive assignment as well.
    keep = [i for i in range(len(chosen)) if len(set(days[train & (assignment == i)])) >= MIN_DAYS]
    if not keep:
        return np.empty(0, dtype=int), np.full(len(x), -1), np.full(len(x), np.nan)
    chosen = np.asarray(chosen)[keep]
    similarities = jaccard(x, x[chosen])
    assignment = similarities.argmax(axis=1)
    best = similarities[np.arange(len(x)), assignment]
    assignment[best < THRESHOLD] = -1
    return chosen, assignment, best


def market_context(data):
    context = {}
    for times, bars in load_sessions(data):
        for i in range(30, len(times)-5):
            o, h, l, c, v, oi = bars[i]
            recent = bars[i-29:i+1]
            steps = 1e4 * np.diff(np.log(bars[i:i+6, 3]))
            path = np.abs(steps).sum()
            context[times[i].isoformat(sep=' ')] = {
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v, 'open_interest': oi,
                'past_30m_return_bp': 1e4*np.log(c/bars[i-30,3]),
                'past_30m_range_bp': 1e4*np.log(recent[:,1].max()/recent[:,2].min()),
                'past_30m_volume': recent[:,4].sum(),
                'past_30m_oi_change': oi-bars[i-30,5],
                'future_abs_bp': abs(steps.sum()), 'future_variance_bp2': steps @ steps,
                'future_path_efficiency': abs(steps.sum())/path if path else 0,
            }
    return context


def draw(path, rasters, ids, table, title):
    image = Image.new('RGB', (360*len(ids)+35, 620), '#f7f9fc')
    pen = ImageDraw.Draw(image)
    def text(x,y,value,size):
        pen.text((x,y), value, fill='#14233a', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size))
    text(20,15,title,26)
    text(20,50,'30 minutes left to right | 216 neurons top to bottom',16)
    prototype = rasters[ids[0]]
    for col, idx in enumerate(ids):
        row = table.iloc[idx]
        x, top = 20 + col*360, 144
        tag = 'Prototype' if col == 0 else row['split']
        text(x,80,f'{tag} | {row.timestamp[5:16]} | J={row.overlap:.2f}',16)
        text(x,105,f'Past return {row.past_30m_return_bp:+.1f} bp | Vol {row.past_30m_volume:,.0f}',15)
        pen.rectangle((x,top,x+300,top+432),fill='white',outline='#bcc9db')
        for minute, neuron in np.argwhere(rasters[idx]):
            left, y = x+10*minute+2, top+2*neuron
            color = '#2872bf' if prototype[minute, neuron] else '#dd6b27'
            pen.rectangle((left,y,left+4,y+1),fill=color)
        text(x,584,f'Future 5m return {row.future_return_bp:+.1f} bp',15)
    image.save(path)


def summarize(table):
    # First occurrence per pattern and day avoids counting runs of overlapping windows.
    # Membership and this selection use only information available at the window end.
    matches = table[table.pattern >= 0].drop_duplicates(['pattern', 'day'])
    metrics = ['future_return_bp','future_abs_bp','future_variance_bp2','future_path_efficiency']
    rows = []
    for (split, pattern), g in matches.groupby(['split','pattern']):
        # Compare against all eligible windows at the same clock minute in the same split.
        universe = table[table.split == split]
        typical = universe.groupby('clock')[metrics].mean()
        delta = g[metrics].to_numpy() - typical.loc[g.clock,metrics].to_numpy()
        row = dict(split=split, pattern=pattern, days=len(g), clock_count=g.clock.nunique(),
                   early_fraction=g.early.mean(), mean_jaccard=g.overlap.mean(),
                   positive_fraction=(g.future_return_bp>0).mean(),
                   median_past_range_bp=g.past_30m_range_bp.median(),
                   median_past_volume=g.past_30m_volume.median(),
                   median_past_oi_change=g.past_30m_oi_change.median())
        for j, name in enumerate(metrics):
            row[name] = g[name].mean()
            row[name+'_minus_clock_mean'] = delta[:,j].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main(output):
    data = Path('data/ic_holdout_input.csv')
    _, random, stdp, _, times, groups, future = load_features(
        data, Path('experiments/segmented_v1'), window_signature=lambda raster: raster)
    dates = np.array([t[:10] for t in times])
    clocks = np.array([t[11:16] for t in times])
    splits = np.array(['train','validation','diagnostic'])[groups]
    discoveries = {}
    for kind, rasters in [('random', random), ('stdp', stdp)]:
        x = rasters.reshape(len(rasters),-1).astype(np.float32)
        discoveries[kind] = discover(x, groups==0, dates)
        print(kind, 'prototypes frozen', len(discoveries[kind][0]), flush=True)

    output.mkdir(parents=True)
    by_time = market_context(data)
    context = pd.DataFrame([by_time[t] for t in times])
    coverage, summaries = [], []
    for kind, rasters in [('random', random), ('stdp', stdp)]:
        chosen, assignment, best = discoveries[kind]
        table = context.assign(timestamp=times, day=dates, clock=clocks, split=splits,
                               pattern=assignment+1, overlap=best, future_return_bp=future)
        table.loc[assignment<0,'pattern'] = -1
        table['early'] = table.clock.between('10:01','10:05') | table.clock.between('13:31','13:35')
        table.to_csv(output/f'{kind}_occurrences.csv',index=False)
        np.savez(output/f'{kind}_prototypes.npz', spikes=rasters[chosen], timestamps=times[chosen],
                 threshold=THRESHOLD)
        summaries.append(summarize(table).assign(pool=kind))
        for split, g in table.groupby('split'):
            matched=g[g.pattern>=0]
            coverage.append(dict(pool=kind,split=split,windows=len(g),matched=len(matched),
                                 coverage=len(matched)/len(g),early_fraction=matched.early.mean(),
                                 patterns=matched.pattern.nunique()))
        for pattern, seed in enumerate(chosen,1):
            subset=table[(table.pattern==pattern)&(table.day!=dates[seed])]
            ids=[seed]
            for split in ('train','validation','diagnostic'):
                eligible=subset[subset.split==split]
                if len(eligible):
                    ids.append(eligible.overlap.idxmax())
            draw(output/f'{kind}_pattern_{pattern}.png',rasters,ids,table,f'{kind.upper()} | Pattern {pattern}')
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(output/'pattern_summary.csv',index=False)
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)



