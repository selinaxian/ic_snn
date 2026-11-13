import csv
from datetime import datetime

import numpy as np


def load_sessions(path):
    fields = ('open', 'high', 'low', 'close', 'volume', 'open_interest')
    sessions = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        previous = None
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row['date'])
            values = [float(row[key]) for key in fields]
            if previous is not None and stamp <= previous:
                raise ValueError('Timestamps must be unique and sorted')
            if previous is None or stamp.date() != previous.date() or (stamp - previous).total_seconds() != 60:
                sessions.append(([], []))
            sessions[-1][0].append(stamp)
            sessions[-1][1].append(values)
            previous = stamp
    return [(times, np.asarray(values)) for times, values in sessions]


def ohlcv_features(values, extra):
    if not np.isfinite(values).all() or np.any(values[:, :4] <= 0) or np.any(values[:, 4:] < 0):
        raise ValueError('Invalid OHLC, volume or open interest')
    o, h, l, c, v, oi = values.T
    features = [np.diff(np.log(c)), np.log(c[1:] / o[1:]),
                np.log(h[1:] / l[1:]), np.log(o[1:] / c[:-1])]
    if extra:
        features.extend([np.diff(np.log1p(v)), np.diff(np.log1p(oi))])
    x = np.column_stack(features)
    return np.column_stack([x, -x])
