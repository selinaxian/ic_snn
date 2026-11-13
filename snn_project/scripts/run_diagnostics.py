"""Fixed-target and within-segment order diagnostics for segmented_v1."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from snn_core.encoding import encode_factor_thresholds
from snn_core.market_data import load_sessions, ohlcv_features
from snn_core.metrics import evaluate, paired_interval
from snn_core.ridge_readout import predict_ridge
from snn_core.snncube import SNNCube
from snn_core.spike_features import segmented_firing_rates


def load_features(path, baseline, window_signature=None):
    with np.load(baseline / 'reservoir.npz') as state:
        center, scale = state['center'], state['scale']
        pools = []
        for key in ('initial', 'learned'):
            pool = SNNCube(11, 43, 2.5, side=6, depression=1, tau=2)
            for attr in ('pre', 'post', 'sign', 'ports'):
                np.testing.assert_array_equal(getattr(pool, attr), state[attr])
            pool.weights[:] = state[key]
            pools.append(pool)
    raw, random, stdp, windows, dates, groups, actual = [], [], [], [], [], [], []
    for times, bars in load_sessions(path):
        if len(times) < 36:
            continue
        day = times[0].date().isoformat()
        group = 0 if day <= '2026-05-11' else 1 if day <= '2026-06-30' else 2
        x = ohlcv_features(bars, True)[:, :6]
        encoded = encode_factor_thresholds(x, center, scale)
        ends = np.arange(30, len(times) - 5)
        offsets = ends[:, None] + np.arange(-30, 0)
        raw.append(((x - center) / scale)[offsets].reshape(len(ends), -1))
        fixed_spikes, learned_spikes = (pool.simulate(encoded) for pool in pools)
        if window_signature is None:
            random.append(segmented_firing_rates(fixed_spikes, ends))
            stdp.append(segmented_firing_rates(learned_spikes, ends))
            windows.append(learned_spikes[offsets])
        else:
            random.append(np.stack([window_signature(fixed_spikes[idx]) for idx in offsets]))
            stdp.append(np.stack([window_signature(learned_spikes[idx]) for idx in offsets]))
        dates.extend(times[end].isoformat(sep=' ') for end in ends)
        groups.extend([group] * len(ends))
        actual.extend(1e4 * np.log(bars[ends + 5, 3] / bars[ends, 3]))
    return (*map(np.vstack, (raw, random, stdp)), np.vstack(windows) if windows else None,
            np.asarray(dates), np.asarray(groups), np.asarray(actual))


def shuffle_within_segments(windows, seed=43):
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(windows)
    for lo, hi in ((0, 15), (15, 25), (25, 30)):
        order = rng.random((len(windows), hi - lo)).argsort(axis=1)
        shuffled[:, lo:hi] = np.take_along_axis(windows[:, lo:hi], order[:, :, None], axis=1)
        np.testing.assert_array_equal(shuffled[:, lo:hi].sum(axis=1), windows[:, lo:hi].sum(axis=1))
    return shuffled


def main(data, baseline, output):
    raw, random, stdp, windows, dates, group, y = load_features(data, baseline)
    train, query = group == 0, group != 0
    old = pd.read_csv(baseline / 'predictions.csv')
    np.testing.assert_array_equal(old.timestamp.to_numpy(), dates[query])
    np.testing.assert_array_equal(old.split.to_numpy(), group[query])
    np.testing.assert_allclose(old.actual_bp.to_numpy(), y[query], atol=1e-10)
    predictions = {'signed': {}, 'absolute': {}}
    for model, x in [('raw', raw), ('raw_random', np.column_stack((raw, random))),
                     ('raw_stdp', np.column_stack((raw, stdp)))]:
        for target, labels in [('signed', y), ('absolute', np.abs(y))]:
            predictions[target][model] = predict_ridge(x[train], labels[train], x[query])
        np.testing.assert_allclose(predictions['signed'][model], old[model], atol=1e-9)
    print('Label probe complete; baseline predictions match', flush=True)
    for model, spikes in [('ordered', windows), ('shuffled', shuffle_within_segments(windows))]:
        # Same standardized ridge objective; float32 limits peak memory for the 7,308 columns.
        x = np.empty((len(raw), raw.shape[1] + stdp.shape[1] + spikes.shape[1] * spikes.shape[2]),
                     dtype=np.float32)
        x[:, :raw.shape[1]] = raw
        x[:, raw.shape[1]:raw.shape[1] + stdp.shape[1]] = stdp
        x[:, raw.shape[1] + stdp.shape[1]:] = spikes.reshape(len(spikes), -1)
        predictions['signed'][model] = predict_ridge(x[train], y[train].astype(np.float32), x[query])
        del x
        print(f'{model} probe complete', flush=True)

    report = {}
    for target, pred in predictions.items():
        labels = (y if target == 'signed' else np.abs(y))[query]
        comparisons = [('raw_stdp', 'raw'), ('raw_stdp', 'raw_random')]
        if target == 'signed':
            comparisons += [('ordered', 'shuffled'), ('ordered', 'raw_stdp'),
                            ('shuffled', 'raw_stdp')]
        report[target] = {}
        for split, code in [('validation', 1), ('diagnostic', 2)]:
            mask = group[query] == code
            actual = labels[mask]
            days = np.array([stamp[:10] for stamp in dates[query][mask]])
            metrics = {name: evaluate(actual, values[mask]) for name, values in pred.items()}
            report[target][split] = {
                'days': len(set(days)), 'samples': int(mask.sum()),
                'models': metrics,
                'paired_rank_ic': {
                    f'{left}_minus_{right}': {
                        'delta': metrics[left]['rank_ic'] - metrics[right]['rank_ic'],
                        'ci_95': paired_interval(actual, pred[left][mask], pred[right][mask], days).tolist(),
                    } for left, right in comparisons
                },
            }
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        columns = [(f'{target}_{model}', values) for target, pred in predictions.items()
                   for model, values in pred.items()]
        writer.writerow(['timestamp', 'split', 'actual_bp', *[name for name, _ in columns]])
        writer.writerows(zip(dates[query], group[query], y[query], *[p for _, p in columns]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ic_holdout_input.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('experiments/segmented_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.baseline, args.output)

