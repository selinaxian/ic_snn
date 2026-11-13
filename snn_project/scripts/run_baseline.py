import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from snn_core.desnn import baseline_desnn_signature as signature, predict_knn_mean as predict_knn
from snn_core.encoding import encode_gao_rate as encode
from snn_core.market_data import load_sessions, ohlcv_features as channels
from snn_core.metrics import evaluate
from snn_core.ridge_readout import predict_ridge
from snn_core.snncube import SNNCube as Cube


def run(args):
    sessions = load_sessions(args.data)
    days = sorted({times[0].date() for times, _ in sessions})
    if len(days) < 10:
        raise ValueError('Need at least ten trading days')
    splits = [set(days[:int(.7 * len(days))]), set(days[int(.7 * len(days)):int(.85 * len(days))]),
              set(days[int(.85 * len(days)):])]
    if args.holdout:
        splits = [set(d for d in days if d <= datetime(2026, 5, 11).date()),
                  set(d for d in days if datetime(2026, 5, 11).date() < d <= datetime(2026, 6, 30).date()),
                  set(d for d in days if d > datetime(2026, 6, 30).date())]
        if not all(splits):
            raise ValueError("Holdout mode requires data before and after June 30")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'config': vars(args) | {'data': str(args.data), 'output': str(args.output)},
               'day_ranges': [[str(min(group)), str(max(group)), len(group)] for group in splits], 'variants': {}}
    prediction_rows = []
    for extra in (False, True):
        name = 'ohlc_volume_oi' if extra else 'ohlc'
        prepared = [(times, values, channels(values, extra)) for times, values in sessions if len(times) >= args.window + args.horizon + 1]
        train = np.concatenate([x for times, _, x in prepared if times[0].date() in splits[0]])
        mean, scale = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-10)
        cube = Cube(train.shape[1], args.seed, args.gain, args.side, args.depression)
        initial = cube.weights.copy()
        random_cube = Cube(train.shape[1], args.seed, args.gain, args.side, args.depression)
        print(f'{name}: training STDP', flush=True)
        for i, (times, _, x) in enumerate(prepared):
            if times[0].date() in splits[0]:
                cube.simulate(encode(x, mean, scale, args.steps, args.seed + i), learn=True)
        print(f'{name}: extracting frozen states', flush=True)
        features = [[], [], [], [], []]
        labels, groups, stamps = [], [], []
        spike_totals = np.zeros(2)
        internal_totals = np.zeros(2)
        internal_steps = 0
        simulated = 0
        for i, (times, values, x) in enumerate(prepared):
            encoded = encode(x, mean, scale, args.steps, args.seed + i)
            trained = cube.simulate(encoded)
            random = random_cube.simulate(encoded)
            spike_totals += [trained.sum(), random.sum()]
            simulated += trained.size
            internal_totals += [trained[:, cube.internal].sum(), random[:, cube.internal].sum()]
            internal_steps += len(trained) * cube.internal.sum()
            group = next(j for j, days_in_group in enumerate(splits) if times[0].date() in days_in_group)
            for end in range(args.window, len(times) - args.horizon, args.stride):
                interval = slice((end - args.window) * args.steps, end * args.steps)
                features[0].append(((x[end - args.window:end] - mean) / scale)[:, :x.shape[1] // 2].ravel())
                features[1].append(signature(random[interval]))
                features[2].append(signature(trained[interval]))
                features[3].append(random[interval].mean(axis=0))
                features[4].append(trained[interval].mean(axis=0))
                labels.append(1e4 * np.log(values[end + args.horizon, 3] / values[end, 3]))
                groups.append(group)
                stamps.append(times[end].isoformat(sep=' '))
        y, group = np.asarray(labels), np.asarray(groups)
        train_mask = group == 0
        report = {'windows': [int((group == i).sum()) for i in range(3)],
                  'spike_rate_stdp_random': (spike_totals / simulated).tolist(),
                  'internal_spike_rate_stdp_random': (internal_totals / internal_steps).tolist(),
                  'mean_abs_weight_change': float(np.mean(np.abs(cube.weights - initial))),
                  'fraction_weights_changed': float(np.mean(cube.weights != initial)), 'metrics': {}}
        names = ('raw_knn', 'random_cube', 'stdp_cube')
        features = [np.asarray(f) for f in features]
        ridge_names = ('raw_ridge', 'random_signature_ridge', 'stdp_signature_ridge',
                       'random_count_ridge', 'stdp_count_ridge',
                       'raw_random_ridge', 'raw_stdp_ridge')
        ridge_features = features + [np.column_stack((features[0], features[i])) for i in (1, 2)]
        all_names = (() if args.skip_knn else names) + ridge_names
        report['mean_in_degree'] = len(cube.pre) / cube.size
        for split, index in [('validation', 1), ('test', 2)]:
            mask = group == index
            predictions = {'zero': np.zeros(mask.sum())}
            for key, f in zip(names, features) if not args.skip_knn else ():
                f = np.asarray(f)
                predictions[key] = predict_knn(f[train_mask], y[train_mask], f[mask])
            report['metrics'][split] = {key: evaluate(y[mask], p) for key, p in predictions.items()}
            for key, f in zip(ridge_names, ridge_features):
                predictions[key] = predict_ridge(f[train_mask], y[train_mask], f[mask])
            report['metrics'][split].update({key: evaluate(y[mask], predictions[key]) for key in ridge_names})
            for j, idx in enumerate(np.flatnonzero(mask)):
                prediction_rows.append([name, split, stamps[idx], y[idx], *(predictions[key][j] for key in all_names)])
        summary['variants'][name] = report
        np.savez(args.output / f'{name}_reservoir.npz', pre=cube.pre, post=cube.post, sign=cube.sign,
                 ports=cube.ports, max_weight=cube.max_weight, weights=cube.weights, initial=initial, mean=mean, scale=scale,
                 readout_x=np.asarray(features[2])[train_mask], readout_y=y[train_mask])
        (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{name}: complete', flush=True)
    with (args.output / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['variant', 'split', 'timestamp', 'actual_bp', *(key + '_bp' for key in all_names)])
        writer.writerows(prediction_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/final.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--gain', type=float, default=2.5)
    parser.add_argument('--side', type=int, default=4, choices=(4, 6))
    parser.add_argument('--depression', type=float, default=1.1)
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--skip-knn', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.window, args.horizon, args.steps, args.stride) < 1 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error('window, horizon, stride, steps and gain must be positive')
    run(args)


