"""Relabel the fixed deSNN neighbor experiment with five-minute realized variance."""
import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from run_diagnostics import load_features
from snn_core.desnn import desnn_regression, desnn_signature
from snn_core.market_data import load_sessions
from snn_core.metrics import evaluate, paired_interval


def variance_labels(path):
    labels, stamps = [], []
    for times, bars in load_sessions(path):
        if len(times) < 36:
            continue
        squared = (1e4 * np.diff(np.log(bars[:, 3]))) ** 2
        ends = np.arange(30, len(times) - 5)
        labels.extend(np.lib.stride_tricks.sliding_window_view(squared, 5)[ends].sum(axis=1))
        stamps.extend(times[end].isoformat(sep=' ') for end in ends)
    return np.asarray(labels), np.asarray(stamps)


def main(data, baseline, previous, output):
    raw, random, stdp, _, dates, group, y = load_features(
        data, baseline, window_signature=partial(desnn_signature, mod=0.5))
    variance, stamps = variance_labels(data)
    np.testing.assert_array_equal(stamps, dates)
    train, query = group == 0, group != 0
    history, actual = variance[train], variance[query]
    old = pd.read_csv(previous / 'predictions.csv')
    np.testing.assert_array_equal(old.timestamp.to_numpy(), dates[query])
    np.testing.assert_array_equal(old.split.to_numpy(), group[query])
    np.testing.assert_allclose(old.actual_bp, y[query], atol=1e-9)
    raw = (raw - raw[train].mean(axis=0)) / np.maximum(raw[train].std(axis=0), 1e-6)

    scores, matches, weights = {}, {}, {}
    for name, x in [('raw', raw), ('random_pool', random), ('stdp_pool', stdp)]:
        reference, current = x[train], x[query]
        signed, idx = desnn_regression(reference, y[train], current, return_neighbors=True)
        np.testing.assert_allclose(signed, old[name], atol=1e-9)
        distances = np.square(current[:, None, :] - reference[idx]).sum(axis=2)
        weight = np.exp(-0.5 * (distances - distances.min(axis=1, keepdims=True)))
        weight /= weight.sum(axis=1, keepdims=True)
        np.testing.assert_allclose((weight * y[train][idx]).sum(axis=1), signed, atol=1e-9)
        scores[name] = (weight * history[idx]).sum(axis=1)
        matches[name], weights[name] = idx, weight
        print(f'{name}: original returns verified; variance relabeled', flush=True)

    uniform = np.random.default_rng(43).integers(len(history), size=(len(actual), 3))
    report = {'target': 'sum of next five squared one-minute log returns', 'unit': 'bp^2', 'splits': {}}
    for name, split in [('validation', 1), ('diagnostic', 2)]:
        mask = group[query] == split
        target = actual[mask]
        days = np.asarray([stamp[:10] for stamp in dates[query][mask]])
        models = {}
        for key, pred in scores.items():
            neighbor_labels = history[matches[key][mask]]
            models[key] = {
                'rank_ic': evaluate(target, pred[mask])['rank_ic'],
                'mae_bp2': float(np.abs(pred[mask] - target).mean()),
                'rmse_bp2': float(np.sqrt(np.mean((pred[mask] - target) ** 2))),
                'mean_within_neighbor_std_bp2': float(neighbor_labels.std(axis=1).mean()),
                'mean_abs_neighbor_target_error_bp2': float(np.abs(neighbor_labels - target[:, None]).mean()),
            }
        uniform_labels = history[uniform[mask]]
        report['splits'][name] = {
            'days': len(set(days)), 'samples': int(mask.sum()), 'models': models,
            'uniform_mean_within_neighbor_std_bp2': float(uniform_labels.std(axis=1).mean()),
            'stdp_minus_control_rank_ic_ci': {
                key: paired_interval(target, scores['stdp_pool'][mask], scores[key][mask], days).tolist()
                for key in ('raw', 'random_pool')
            },
        }
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'split', 'actual_variance_bp2', *scores])
        writer.writerows(zip(dates[query], group[query], actual, *scores.values()))
    np.savez(output / 'fixed_neighbors.npz', train_timestamps=dates[train], query_timestamps=dates[query],
             **{f'{key}_indices': idx for key, idx in matches.items()},
             **{f'{key}_weights': weight for key, weight in weights.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ic_holdout_input.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('experiments/segmented_v1'))
    parser.add_argument('--previous', type=Path, default=Path('experiments/desnn_neighbors_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.baseline, args.previous, args.output)
