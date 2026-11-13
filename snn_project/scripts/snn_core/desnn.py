import numpy as np


def baseline_desnn_signature(spikes):
    active = spikes.any(axis=0)
    first = spikes.argmax(axis=0)
    rank = np.searchsorted(np.sort(first[active]), first[active], side='left')
    result = np.zeros(spikes.shape[1])
    counts = spikes[:, active].sum(axis=0)
    later_steps = len(spikes) - 1 - first[active]
    result[active] = 0.9 ** rank + 0.005 * (2 * (counts - 1) - later_steps)
    return result


def predict_knn_mean(x, y, query):
    ref_norm = (x * x).sum(axis=1)
    predictions = []
    for batch in np.array_split(query, max(1, (len(query) + 127) // 128)):
        distance = (batch * batch).sum(axis=1)[:, None] + ref_norm - 2 * batch @ x.T
        neighbors = np.argpartition(distance, 2, axis=1)[:, :3]
        predictions.extend(y[neighbors].mean(axis=1))
    return np.asarray(predictions)


def desnn_signature(spikes, mod):
    active = spikes.any(axis=0)
    first = spikes.argmax(axis=0)[active]
    rank = np.searchsorted(np.sort(first), first)
    later_spikes = spikes[:, active].sum(axis=0) - 1
    later_steps = len(spikes) - 1 - first
    weights = np.zeros(spikes.shape[1])
    weights[active] = mod ** rank + 0.005 * (2 * later_spikes - later_steps)
    return weights


def desnn_regression(train, labels, query, return_neighbors=False):
    predictions, neighbor_batches = [], []
    norms = np.square(train).sum(axis=1)
    for start in range(0, len(query), 128):
        batch = query[start:start + 128]
        distance = np.square(batch).sum(axis=1)[:, None] + norms - 2 * batch @ train.T
        neighbors = np.argpartition(distance, 2, axis=1)[:, :3]
        if return_neighbors:
            neighbor_batches.append(neighbors)
        squared = np.take_along_axis(distance, neighbors, axis=1)
        weights = np.exp(-0.5 * (squared - squared.min(axis=1, keepdims=True)))
        predictions.extend((weights * labels[neighbors]).sum(axis=1) / weights.sum(axis=1))
    predictions = np.asarray(predictions)
    if return_neighbors:
        return predictions, np.vstack(neighbor_batches)
    return predictions


