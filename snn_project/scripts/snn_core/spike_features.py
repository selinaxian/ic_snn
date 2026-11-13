import numpy as np


def segmented_firing_rates(spikes, ends):
    cumulative = np.vstack((np.zeros(spikes.shape[1]), spikes.cumsum(axis=0)))
    return np.column_stack([
        (cumulative[ends - left] - cumulative[ends - right]) / (right - left)
        for left, right in ((0, 5), (5, 15), (15, 30))
    ])
