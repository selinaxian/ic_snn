"""Threshold representation, STDP reservoir and deSNN regression experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from snn_core.desnn import desnn_regression, desnn_signature
from snn_core.encoding import threshold_encode
from snn_core.market_data import load_sessions
from snn_core.metrics import evaluate
from snn_core.ridge_readout import predict_ridge
from snn_core.snncube import SNNCube as Cube


def experiment(samples, name, output, side, persistence):
    # Each entry contains a time-by-variable stream, output indices and labels.
    training = [values for group, values, _, _, _ in samples if group == 0]
    change = np.concatenate([np.diff(values, axis=0) for values in training])
    scale = np.maximum(change.std(axis=0), 1e-10)
    cube = Cube(2 * len(scale), 43, 2.5, side, 1)
    random = Cube(2 * len(scale), 43, 2.5, side, 1)
    streams = [(group, threshold_encode(values, scale), ends, labels, timestamps)
               for group, values, ends, labels, timestamps in samples]
    for group, encoded, _, _, _ in streams:
        if group == 0:
            cube.simulate(encoded, learn=True)
    features = {(kind, mod): [] for kind in ('random', 'stdp') for mod in (0.5, 0.8)}
    groups, actual, timestamps = [], [], []
    firing = np.zeros(2)
    neuron_steps = 0
    for group, encoded, ends, labels, times in streams:
        for index, (kind, model) in enumerate((('random', random), ('stdp', cube))):
            spikes = model.simulate(encoded)
            firing[index] += spikes[:, model.internal].sum()
            for mod in (0.5, 0.8):
                features[kind, mod].extend(desnn_signature(spikes[max(0, end - 59):end + 1], mod)
                                           for end in ends)
        neuron_steps += len(encoded) * cube.internal.sum()
        groups.extend([group] * len(ends))
        actual.extend(labels)
        timestamps.extend(times)
    group, y = np.asarray(groups), np.asarray(actual)
    train = group == 0
    summary = {'samples': {str(g): int((group == g).sum()) for g in np.unique(group)},
               'internal_spike_rate_random_stdp': (firing / neuron_steps).tolist(), 'metrics': {}}
    predictions = {'persistence': np.asarray(persistence)}
    for (kind, mod), data in features.items():
        data = np.asarray(data)
        query = ~train
        for method, fit in [('desnn', desnn_regression), ('ridge', predict_ridge)]:
            pred = np.full(len(y), np.nan)
            pred[query] = fit(data[train], y[train], data[query])
            predictions[f'{kind}_{mod}_{method}'] = pred
    for split, idx in [('validation', 1), ('test', 2)]:
        mask = group == idx
        if mask.any():
            summary['metrics'][split] = {key: evaluate(y[mask], pred[mask]) for key, pred in predictions.items()}
    summary['target_unit'] = 'price' if name == 'qqqx' else 'bp'
    for metrics in summary['metrics'].values():
        for entry in metrics.values():
            entry['rmse'] = entry.pop('rmse_bp')
    with (output / f'{name}_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'timestamp_or_sample', 'actual', *predictions])
        for idx in np.flatnonzero(~train):
            writer.writerow([group[idx], timestamps[idx], y[idx], *(p[idx] for p in predictions.values())])
    return summary


def daily(directory):
    labels = np.loadtxt(directory / 'tar_class_labels.csv', ndmin=1)
    samples, persistence = [], []
    for i, label in enumerate(labels, 1):
        values = np.loadtxt(directory / f'sam{i}_qqqx.csv', delimiter=',')[-60:]
        group = 0 if i <= 25 else 2
        samples.append((group, values, [len(values) - 1], [label], [i]))
        persistence.append(values[-1, -1])
    return samples, persistence


def ic(path, extra):
    samples, persistence = [], []
    for times, values in load_sessions(path):
        if len(times) < 65:
            continue
        day = times[0].strftime('%Y-%m-%d')
        group = 0 if day <= '2026-05-11' else 1 if day <= '2026-06-30' else 2
        data = np.log(values[:, :4])
        if extra:
            data = np.column_stack((data, np.log1p(values[:, 4:])))
        ends = np.arange(59, len(times) - 5)
        labels = 1e4 * np.log(values[ends + 5, 3] / values[ends, 3])
        samples.append((group, data, ends, labels, [times[end].isoformat(sep=' ') for end in ends]))
        persistence.extend(np.zeros(len(ends)))
    return samples, persistence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ic', type=Path, required=True)
    parser.add_argument('--share-price', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    result = {}
    for name, data, side in [('qqqx', daily(args.share_price), 10),
                             ('ic_ohlc', ic(args.ic, False), 6),
                             ('ic_ohlc_volume_oi', ic(args.ic, True), 6)]:
        print(name, flush=True)
        result[name] = experiment(data[0], name, args.output, side, data[1])
        (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


