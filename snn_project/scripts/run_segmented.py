"""Paired deterministic IC reservoir experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from snn_core.encoding import encode_factor_thresholds
from snn_core.market_data import load_sessions, ohlcv_features
from snn_core.metrics import evaluate, paired_interval
from snn_core.ridge_readout import predict_ridge
from snn_core.snncube import SNNCube
from snn_core.spike_features import segmented_firing_rates


def main(data, output):
    sessions = [(times, bars, ohlcv_features(bars, True)[:, :6])
                for times, bars in load_sessions(data) if len(times) >= 36]

    def split(times):
        day = times[0].date().isoformat()
        return 0 if day <= '2026-05-11' else 1 if day <= '2026-06-30' else 2

    training = np.concatenate([x for times, _, x in sessions if split(times) == 0])
    center = np.median(training, axis=0)
    scale = np.maximum(training.std(axis=0), 1e-10)
    streams = []
    for times, bars, x in sessions:
        standardized = (x - center) / scale
        encoded = encode_factor_thresholds(x, center, scale)
        streams.append((times, bars, standardized, encoded))

    frozen = SNNCube(11, 43, 2.5, side=6, depression=1, tau=2)
    learned = SNNCube(11, 43, 2.5, side=6, depression=1, tau=2)
    for times, _, _, encoded in streams:
        if split(times) == 0:
            learned.simulate(encoded, learn=True)
    print('STDP trained; extracting paired features', flush=True)
    parts = [[], [], []]
    times_out, y, groups = [], [], []
    activity = np.zeros((3, 2))
    steps = np.zeros(3)
    for times, bars, x, encoded in streams:
        ends = np.arange(30, len(times) - 5)
        group = split(times)
        parts[0].append(x[ends[:, None] + np.arange(-30, 0)].reshape(len(ends), -1))
        for i, model in enumerate((frozen, learned)):
            spikes = model.simulate(encoded)
            parts[i + 1].append(segmented_firing_rates(spikes, ends))
            activity[group, i] += spikes[:, model.internal].sum()
        steps[group] += len(encoded) * frozen.internal.sum()
        groups.extend([group] * len(ends))
        times_out.extend(times[end].isoformat(sep=' ') for end in ends)
        y.extend(1e4 * np.log(bars[ends + 5, 3] / bars[ends, 3]))
    raw, random, stdp = map(np.vstack, parts)
    y, group, dates = np.asarray(y), np.asarray(groups), np.asarray(times_out)
    train, query = group == 0, group != 0
    models = {'raw': raw, 'raw_random': np.column_stack((raw, random)),
              'raw_stdp': np.column_stack((raw, stdp))}
    pred = {'zero': np.zeros(query.sum())}
    for name, x in models.items():
        pred[name] = predict_ridge(x[train], y[train], x[query])
    report = {'samples': np.bincount(group).tolist(),
              'internal_spike_rates_random_stdp': (activity / steps[:, None]).tolist(),
              'mean_abs_weight_change': float(np.abs(frozen.weights - learned.weights).mean()),
              'metrics': {}}
    for name, index in [('validation', 1), ('diagnostic', 2)]:
        mask = group[query] == index
        target = y[query][mask]
        dates_in_split = dates[query][mask]
        days = np.array([stamp[:10] for stamp in dates_in_split])
        report['metrics'][name] = {
            'range': [min(days), max(days), len(set(days))],
            'models': {key: evaluate(target, p[mask]) for key, p in pred.items()},
            'stdp_rank_ic_delta_ci': {
                key: paired_interval(target, pred['raw_stdp'][mask], pred[key][mask], days).tolist()
                for key in ('raw', 'raw_random')
            }
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'split', 'actual_bp', *pred])
        writer.writerows(zip(dates[query], group[query], y[query], *pred.values()))
    np.savez(output / 'reservoir.npz', pre=learned.pre, post=learned.post, sign=learned.sign,
             ports=learned.ports, initial=frozen.weights, learned=learned.weights, center=center, scale=scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.output)


