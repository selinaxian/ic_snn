"""Compare three historical neighbor spaces for the same IC target."""
import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np

from run_diagnostics import load_features
from snn_core.desnn import desnn_regression, desnn_signature
from snn_core.metrics import evaluate, paired_interval


def main(data, baseline, output):
    raw, random, stdp, _, dates, group, y = load_features(
        data, baseline, window_signature=partial(desnn_signature, mod=0.5))
    train, query = group == 0, group != 0
    history, actual = y[train], y[query]
    # Scale continuous factors using training rows only; keep deSNN weights in their native space.
    raw = (raw - raw[train].mean(axis=0)) / np.maximum(raw[train].std(axis=0), 1e-6)
    scores = {'zero': np.zeros(len(actual))}
    neighbors = {}
    for name, x in [('raw', raw), ('random_pool', random), ('stdp_pool', stdp)]:
        scores[name], neighbors[name] = desnn_regression(x[train], history, x[query], return_neighbors=True)
        print(f'{name} complete', flush=True)
    neighbors['uniform_reference'] = np.random.default_rng(43).integers(len(history), size=(len(actual), 3))
    train_times, query_times = dates[train], dates[query]
    report = {'mod': 0.5, 'drift': 0.005, 'k': 3, 'sigma': 1,
              'samples': np.bincount(group).tolist(), 'splits': {}}
    for name, split in [('validation', 1), ('diagnostic', 2)]:
        mask = group[query] == split
        days = np.array([stamp[:10] for stamp in query_times[mask]])
        target = actual[mask]
        models = {key: evaluate(target, score[mask]) for key, score in scores.items()}
        details = {}
        for key, idx in neighbors.items():
            selected = idx[mask]
            labels = history[selected]
            matched_days = train_times[selected].astype('U10')
            distinct_days = 1 + (matched_days[:, 0] != matched_days[:, 1]).astype(int)
            distinct_days += ((matched_days[:, 2] != matched_days[:, 0]) &
                              (matched_days[:, 2] != matched_days[:, 1]))
            details[key] = {
                'mean_within_neighbor_std_bp': float(labels.std(axis=1).mean()),
                'mean_abs_neighbor_target_error_bp': float(np.abs(labels - target[:, None]).mean()),
                'unanimous_nonzero_sign_fraction': float(((labels > 0).all(axis=1) | (labels < 0).all(axis=1)).mean()),
                'mean_distinct_neighbor_days': float(distinct_days.mean()),
            }
        report['splits'][name] = {
            'date_range': [min(days), max(days)], 'models': models, 'neighbors': details,
            'stdp_rank_ic_delta_ci': {
                key: paired_interval(target, scores['stdp_pool'][mask], scores[key][mask], days).tolist()
                for key in ('raw', 'random_pool')
            },
        }
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'split', 'actual_bp', *scores])
        writer.writerows(zip(query_times, group[query], actual, *scores.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ic_holdout_input.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('experiments/segmented_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.baseline, args.output)
