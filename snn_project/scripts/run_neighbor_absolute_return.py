"""Relabel saved IC neighbors with absolute five-minute net log return."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from snn_core.market_data import load_sessions
from snn_core.metrics import paired_interval, rank_ic


def main(data, experiment, output):
    signed_by_time = {}
    for times, bars in load_sessions(data):
        for end in range(30, len(times) - 5):
            signed_by_time[times[end].isoformat(sep=' ')] = 1e4 * np.log(bars[end + 5, 3] / bars[end, 3])

    previous = pd.read_csv(experiment / 'predictions.csv')
    predictions = previous[['timestamp', 'split']].copy()
    names = ('raw', 'random_pool', 'stdp_pool')
    with np.load(experiment / 'fixed_neighbors.npz') as saved:
        train = np.asarray([signed_by_time[t] for t in saved['train_timestamps']])
        query = saved['query_timestamps']
        np.testing.assert_array_equal(query, previous.timestamp.to_numpy())
        actual = np.abs([signed_by_time[t] for t in query])
        labels = {}
        for name in names:
            idx, weights = saved[f'{name}_indices'], saved[f'{name}_weights']
            np.testing.assert_allclose(weights.sum(axis=1), 1)
            labels[name] = np.abs(train[idx])
            predictions[name] = (weights * labels[name]).sum(axis=1)
    predictions['actual_abs_bp'] = actual
    uniform = np.random.default_rng(43).integers(len(train), size=(len(query), 3))
    report = {'target': 'absolute five-minute net log return', 'unit': 'bp', 'splits': {}}
    days = np.asarray([t[:10] for t in query])
    for split, code in [('validation', 1), ('diagnostic', 2)]:
        mask = predictions.split.to_numpy() == code
        target = actual[mask]
        models = {}
        for name in names:
            score = predictions[name].to_numpy()[mask]
            values = labels[name][mask]
            models[name] = {
                'rank_ic': rank_ic(target, score),
                'mae_bp': np.abs(score - target).mean(),
                'mean_within_neighbor_std_bp': values.std(axis=1).mean(),
                'mean_abs_neighbor_target_error_bp': np.abs(values - target[:, None]).mean(),
            }
        report['splits'][split] = {
            'days': len(set(days[mask])), 'samples': int(mask.sum()), 'models': models,
            'uniform_mean_within_neighbor_std_bp': np.abs(train[uniform[mask]]).std(axis=1).mean(),
            'stdp_minus_control_rank_ic_ci': {
                name: paired_interval(target, predictions.stdp_pool.to_numpy()[mask],
                                      predictions[name].to_numpy()[mask], days[mask]).tolist()
                for name in names[:2]
            },
        }
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    predictions.to_csv(output / 'predictions.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/ic_holdout_input.csv'))
    parser.add_argument('--experiment', type=Path, default=Path('experiments/desnn_neighbor_variance'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.experiment, args.output)
