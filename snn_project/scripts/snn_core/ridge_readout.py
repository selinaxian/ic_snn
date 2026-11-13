import numpy as np


def predict_ridge(x, y, query):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    design = (x - mean) / scale
    gram = design.T @ design
    gram.flat[::len(mean) + 1] += len(y) * 0.1
    weights = np.linalg.solve(gram, design.T @ (y - y.mean()))
    return y.mean() + ((query - mean) / scale) @ weights
