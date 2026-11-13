"""Bounded SPADE-style frequent itemset mining on continuous IC spike trains."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from snn_core.encoding import encode_factor_thresholds
from snn_core.market_data import load_sessions, ohlcv_features
from snn_core.snncube import SNNCube

MIN_SUPPORT = 20
MAX_LAG = 4
PAIR_SEEDS = 24
PER_DURATION = 3


def streams(path, state_path):
    with np.load(state_path) as state:
        pools = {}
        for name, key in [('random', 'initial'), ('stdp', 'learned')]:
            cube = SNNCube(11, 43, 2.5, side=6, depression=1, tau=2)
            for attr in ('pre', 'post', 'sign', 'ports'):
                np.testing.assert_array_equal(getattr(cube, attr), state[attr])
            cube.weights[:] = state[key]
            pools[name] = cube
        for times, bars in load_sessions(path):
            day = times[0].date().isoformat()
            split = 'train' if day <= '2026-05-11' else 'validation' if day <= '2026-06-30' else 'diagnostic'
            stamps = [t.isoformat(sep=' ') for t in times[1:]]
            x = ohlcv_features(bars, True)[:, :6]
            encoded = encode_factor_thresholds(x, state['center'], state['scale'])
            yield stamps, split, {name: cube.simulate(encoded) for name, cube in pools.items()}


def item_matrix(rasters):
    # Row i holds every (lag, neuron) event starting at i. End padding cannot match.
    blocks = []
    for spikes in rasters:
        padded = np.pad(spikes, ((0, MAX_LAG), (0, 0)))
        blocks.append(np.column_stack([padded[lag:lag+len(spikes)] for lag in range(MAX_LAG+1)]))
    return np.vstack(blocks)


def mine(items, neurons):
    numeric = items.astype(np.float32)
    pairs, catalog, tri = [], [], {}
    for lag in range(1, MAX_LAG+1):
        counts = (numeric[:, :neurons].T @ numeric[:, lag*neurons:(lag+1)*neurons]).astype(int)
        np.fill_diagonal(counts, 0)
        candidates = [(int(counts[a,b]), (int(a), int(lag*neurons+b)))
                      for a,b in np.argwhere(counts >= MIN_SUPPORT)]
        candidates.sort(key=lambda row: (-row[0], row[1]))
        pairs.extend(candidates[:PAIR_SEEDS])
        catalog.append((lag, len(candidates)))
    for _, pair in pairs:
        occurrence = items[:, pair].all(axis=1)
        counts = items[occurrence].sum(axis=0)
        for third in np.flatnonzero(counts >= MIN_SUPPORT):
            if third % neurons in {item % neurons for item in pair}:
                continue
            key = tuple(sorted((*pair, int(third))))
            tri[key] = int(counts[third])
    selected = []
    for lag in range(1, MAX_LAG+1):
        selected.extend([row for row in pairs if row[1][-1]//neurons == lag][:PER_DURATION])
        triples = sorted(((count,key) for key,count in tri.items() if key[-1]//neurons == lag),
                         key=lambda row: (-row[0], row[1]))
        selected.extend(triples[:PER_DURATION])
    return selected, catalog, len(tri)


def draw(path, pattern, examples, neurons):
    image = Image.new('RGB', (max(650, 320*len(examples)), 535), '#f7f9fc')
    pen = ImageDraw.Draw(image)
    def text(x, y, s, size):
        pen.text((x,y),s,fill='#17253b',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size))
    text(18,14,pattern,22)
    text(18,47,'5 minute slots | Orange: motif spikes | Blue: other activity',16)
    for col,(raster, stamp, key) in enumerate(examples):
        x, y = 18+320*col, 112
        text(x,82,stamp,17)
        pen.rectangle((x,y,x+260,y+388),fill='white',outline='#b9c7da')
        for lag in range(MAX_LAG+1):
            for neuron in np.flatnonzero(raster[lag]):
                px,py=x+25+52*lag,y+int(386*neuron/(neurons-1))
                color='#e15a21' if lag*neurons+neuron in key else '#377eb8'
                pen.rectangle((px,py,px+(12 if color=='#e15a21' else 5),py+1),fill=color)
            text(x+21+52*lag,505,str(lag),14)
    image.save(path)


def main(output):
    sessions = list(streams(Path('data/ic_holdout_input.csv'), Path('experiments/segmented_v1/reservoir.npz')))
    neurons = 216
    all_stamps = np.array([t for times, _, _ in sessions for t in times])
    all_splits = np.array([split for times, split, _ in sessions for _ in times])
    session_id = np.repeat(np.arange(len(sessions)), [len(times) for times, _, _ in sessions])
    positions = np.concatenate([np.arange(len(times)) for times, _, _ in sessions])
    output.mkdir(parents=True)
    summary, occurrences, searches = [], [], []
    for pool in ('random','stdp'):
        rasters = [entry[2][pool] for entry in sessions]
        items = item_matrix(rasters)
        selected, catalog, triples = mine(items[all_splits=='train'], neurons)
        searches.extend(dict(pool=pool,lag=lag,frequent_pairs=count,triples_total=triples) for lag,count in catalog)
        coverage = np.zeros(len(items), dtype=bool)
        for i,(support,key) in enumerate(selected,1):
            start = np.flatnonzero(items[:,key].all(axis=1))
            duration = key[-1]//neurons
            end = start+duration
            np.testing.assert_array_equal(session_id[start],session_id[end])
            assert (all_splits[end]=='train').sum()==support
            label = ' '.join(f'N{item%neurons}@{item//neurons}' for item in key)
            coverage[end] = True
            for s,e in zip(start,end):
                occurrences.append((pool,i,label,all_splits[e],all_stamps[s],all_stamps[e],positions[e]))
            for split in ('train','validation','diagnostic'):
                match = end[all_splits[end]==split]
                stamps = all_stamps[match]
                clocks = [t[11:16] for t in stamps]
                summary.append(dict(pool=pool,pattern=i,items=label,duration_min=duration,split=split,
                                    occurrences=len(match),days=len({t[:10] for t in stamps}),
                                    clocks=len(set(clocks)),first_10_min=int((positions[match]<10).sum())))
            if i in (1,4,13,16):
                cases, used = [], set()
                for split in ('train','validation','diagnostic'):
                    candidates = start[all_splits[start]==split]
                    if not len(candidates):
                        continue
                    distinct = [s for s in candidates if all_stamps[s+duration][11:16] not in used]
                    s = distinct[0] if distinct else candidates[0]
                    pos = positions[s]
                    block = rasters[session_id[s]][pos:pos+MAX_LAG+1]
                    block = np.pad(block, ((0,MAX_LAG+1-len(block)),(0,0)))
                    stamp = all_stamps[s+duration]
                    used.add(stamp[11:16])
                    cases.append((block,stamp,key))
                draw(output/f'{pool}_{i:02}.png',f'{pool.upper()} #{i} | {label}',cases,neurons)
        for split in ('train','validation','diagnostic'):
            mask = all_splits==split
            print(pool,split,'patterns',len(selected),'covered',int(coverage[mask].sum()),'of',int(mask.sum()),flush=True)
    pd.DataFrame(summary).to_csv(output/'pattern_summary.csv',index=False)
    pd.DataFrame(searches).to_csv(output/'search_summary.csv',index=False)
    pd.DataFrame(occurrences,columns=['pool','pattern','items','split','start','confirmed','session_step']).to_csv(
        output/'occurrences.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

