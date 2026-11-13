"""Compare full spike-raster distances with saved deSNN matches."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_diagnostics import load_features
from snn_core.desnn import desnn_signature
from snn_core.metrics import evaluate, paired_interval


def smooth(spikes):
    signal = spikes.astype(np.float32)
    signal *= 0.5
    signal[:, 1:] += 0.25 * spikes[:, :-1]
    signal[:, :-1] += 0.25 * spikes[:, 1:]
    return signal


def search(train, query, method):
    reference = (smooth(train) if method == 'smooth' else train.astype(np.float32)).reshape(len(train), -1)
    ref_mass = np.einsum('ij,ij->i', reference, reference)
    matches = []
    for start in range(0, len(query), 128):
        raster = query[start:start + 128]
        x = (smooth(raster) if method == 'smooth' else raster.astype(np.float32)).reshape(len(raster), -1)
        dot = x @ reference.T
        mass = np.einsum('ij,ij->i', x, x)[:, None] + ref_mass
        if method == 'smooth':
            distance = mass - 2 * dot
        else:
            distance = 1 - dot / np.maximum(mass - dot, 1)
        idx = np.argpartition(distance, 2, axis=1)[:, :3]
        order = np.argsort(np.take_along_axis(distance, idx, axis=1), axis=1)
        matches.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(matches)


def pair_metrics(train, query, idx):
    overlap, smoothed = [], []
    for start in range(0, len(query), 128):
        sl = slice(start, start + 128)
        a, b = query[sl, None], train[idx[sl]]
        overlap.append((a & b).sum(axis=(2, 3)) / np.maximum((a | b).sum(axis=(2, 3)), 1))
        smooth_b = smooth(b.reshape(-1, *train.shape[1:])).reshape(b.shape)
        smoothed.append(np.sqrt(np.square(smooth(query[sl])[:, None] - smooth_b).sum(axis=(2, 3))))
    return np.vstack(overlap), np.vstack(smoothed)


def draw_case(output, case, i, train, query, dates, train_dates, target, history, matches, metrics):
    img = Image.new('RGB', (1510, 665), '#f7f9fc')
    d = ImageDraw.Draw(img)
    def font(size):
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    def put(x, y, value, size=17):
        d.text((x, y), value, fill='#17253b', font=font(size))
    put(35, 20, 'IC spike rasters | ' + case, 30)
    put(35, 64, '30-minute aligned grid, 216 neurons. Neighbor blue = shared spike; orange = neighbor-only spike.', 19)
    panels = [('Query', query[i], dates[i], target[i], None)]
    for method, title in [('desnn', 'deSNN #1'), ('jaccard', 'Jaccard #1'), ('smooth', 'Smoothed #1')]:
        idx = matches[method][i, 0]
        panels.append((title, train[idx], train_dates[idx], history[idx], method))
    for k, (title, raster, stamp, ret, method) in enumerate(panels):
        left, top = 40 + 375 * k, 152
        put(left, 102, f'{title} | {stamp[5:16]}', 19)
        put(left, 127, f'Next 5 min return: {ret:+.1f} bp')
        d.rectangle((left, top, left + 300, top + 432), fill='white', outline='#bcc9db')
        for t in range(5, 30, 5):
            d.line((left + 10*t, top, left + 10*t, top + 432), fill='#e1e7ef')
        for t, neuron in np.argwhere(raster):
            x, y = left + 10*t + 3, top + 2*neuron
            color = '#2872bf' if method is None or query[i, t, neuron] else '#dd6b27'
            d.rectangle((x, y, x+4, y+1), fill=color)
        put(left, 590, '-29 min', 14)
        put(left + 264, 590, '0 min', 14)
        if method:
            j, s, v = (a[i, 0] for a in metrics[method])
            put(left, 612, f'Exact overlap {j:.1%}  |  Jaccard d {1-j:.3f}', 15)
            put(left, 634, f'Smooth d {s:.3f}  |  deSNN d {v:.3f}', 15)
    img.save(output / f'{case}.png')


def main(output):
    base = Path('experiments')
    _, _, _, spikes, dates, groups, y = load_features(
        Path('data/ic_holdout_input.csv'), base/'segmented_v1')
    train, query = spikes[groups == 0], spikes[groups != 0]
    history, target = y[groups == 0], y[groups != 0]
    train_dates, query_dates = dates[groups == 0], dates[groups != 0]
    with np.load(base/'desnn_neighbor_variance/fixed_neighbors.npz') as state:
        np.testing.assert_array_equal(state['train_timestamps'], train_dates)
        np.testing.assert_array_equal(state['query_timestamps'], query_dates)
        idx = state['stdp_pool_indices']
    train_sig = np.stack([desnn_signature(x, .5) for x in train])
    query_sig = np.stack([desnn_signature(x, .5) for x in query])
    d = np.linalg.norm(train_sig[idx] - query_sig[:, None], axis=2)
    matches = {'desnn': np.take_along_axis(idx, np.argsort(d, axis=1), axis=1)}
    for method in ('jaccard', 'smooth'):
        matches[method] = search(train, query, method)
        print(method, 'search complete', flush=True)
    metrics, predictions = {}, {}
    for method, idx in matches.items():
        overlap, smoothed = pair_metrics(train, query, idx)
        signature = np.linalg.norm(train_sig[idx] - query_sig[:, None], axis=2)
        metrics[method] = overlap, smoothed, signature
        predictions[method] = history[idx].mean(axis=1)
    report = {'smoothing_kernel': [.25, .5, .25], 'k': 3, 'readout': 'equal mean', 'splits': {}}
    for name, code in [('validation', 1), ('diagnostic', 2)]:
        mask = groups[groups != 0] == code
        days = np.array([stamp[:10] for stamp in query_dates[mask]])
        report['splits'][name] = {}
        for method, (overlap, smoothed, signature) in metrics.items():
            stats = evaluate(target[mask], predictions[method][mask])
            stats.update({
                'median_nearest_overlap': float(np.median(overlap[mask, 0])),
                'median_top3_overlap': float(np.median(overlap[mask])),
                'median_nearest_smooth_distance': float(np.median(smoothed[mask, 0])),
                'median_nearest_desnn_distance': float(np.median(signature[mask, 0])),
                'mean_abs_neighbor_target_error_bp': float(np.abs(history[matches[method][mask]] - target[mask, None]).mean()),
            })
            if method != 'desnn':
                stats['delta_rank_ic_ci'] = paired_interval(target[mask], predictions[method][mask],
                                                           predictions['desnn'][mask], days).tolist()
            report['splits'][name][method] = stats
    output.mkdir(parents=True)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    with (output/'neighbor_pairs.csv').open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'query', 'neighbor_rank', 'neighbor', 'spike_overlap', 'jaccard_distance',
                         'smooth_distance', 'desnn_distance', 'actual_bp', 'neighbor_return_bp', 'prediction_bp'])
        for method, idx in matches.items():
            overlap, smoothed, signature = metrics[method]
            for i, stamp in enumerate(query_dates):
                for k, j in enumerate(idx[i]):
                    writer.writerow([method, stamp, k+1, train_dates[j], overlap[i,k], 1-overlap[i,k],
                                     smoothed[i,k], signature[i,k], target[i], history[j], predictions[method][i]])
    test = np.flatnonzero(groups[groups != 0] == 2)
    test = test[np.argsort(metrics['desnn'][2][test, 0])]
    original = np.flatnonzero(query_dates == '2026-07-01 14:01:00')[0]
    for label, i in [('original_example', original), ('close_desnn', test[round(.05*(len(test)-1))]),
                     ('distant_desnn', test[round(.95*(len(test)-1))])]:
        draw_case(output, label, i, train, query, query_dates, train_dates, target, history, matches, metrics)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
