import numpy as np
import pandas as pd


def rank_ic(actual, predicted):
    if np.std(predicted) == 0:
        return np.nan
    return np.corrcoef(pd.Series(actual).rank(), pd.Series(predicted).rank())[0, 1]


def evaluate(y, pred):
    ic = rank_ic(y, pred)
    ic = float(ic) if np.isfinite(ic) else None
    return {'rmse_bp': float(np.sqrt(np.mean((y - pred) ** 2))),
            'rank_ic': ic, 'sign_match': float(np.mean(np.sign(y) == np.sign(pred)))}


def paired_interval(actual, trained, control, days):
    blocks = [np.flatnonzero(days == day) for day in np.unique(days)]
    rng = np.random.default_rng(2026)
    deltas = []
    for draw in rng.integers(0, len(blocks), size=(400, len(blocks))):
        idx = np.concatenate([blocks[i] for i in draw])
        deltas.append(rank_ic(actual[idx], trained[idx]) - rank_ic(actual[idx], control[idx]))
    return np.nanpercentile(deltas, [2.5, 97.5])
