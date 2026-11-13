import numpy as np


def encode_factor_thresholds(values, center, scale):
    standardized = (values - center) / scale
    # Signed returns and bar changes retain their economic zero.
    signed_columns = [0, 1, 3, 4, 5]
    signed = values[:, signed_columns] / scale[signed_columns]
    return np.column_stack((signed > .5, signed < -.5, standardized[:, 2] > .5))


def encode_gao_rate(x, mean, scale, steps, seed):
    rate = np.clip(0.13 + 0.09 * (x - mean) / scale, 0, 1)
    rng = np.random.default_rng(seed)
    return (rng.random((len(x), steps, x.shape[1])) < rate[:, None, :]).reshape(-1, x.shape[1])


def threshold_encode(values, scale):
    change = np.diff(values, axis=0, prepend=values[:1])
    return np.column_stack((change > 0.5 * scale, change < -0.5 * scale))
