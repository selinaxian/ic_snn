import numpy as np

from .stdp import update_excitatory_weights


class SNNCube:
    def __init__(self, inputs, seed, gain, side=4, depression=1.1, tau=8):
        rng = np.random.default_rng(seed)
        xyz = np.indices((side, side, side)).reshape(3, -1).T
        distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
        probability = 0.8 * np.exp(-distance / 1.35)
        np.fill_diagonal(probability, 0)
        if side != 4:
            reference = np.indices((4, 4, 4)).reshape(3, -1).T
            reference_distance = np.linalg.norm(reference[:, None] - reference[None, :], axis=2)
            target = (0.8 * np.exp(-reference_distance / 1.35)).sum(axis=0).mean() - 0.8
            probability *= target / probability.sum(axis=0).mean()
        connected = rng.random(distance.shape) < probability
        self.depression = depression
        self.tau = tau
        self.pre, self.post = np.nonzero(connected)
        self.weights = gain * rng.uniform(0.07, 0.26, len(self.pre))
        inhibitory = rng.random(len(xyz)) < 0.2
        self.sign = np.where(inhibitory[self.pre], -1, 1)
        self.ports = rng.choice(len(xyz), inputs, replace=False)
        self.size = len(xyz)
        self.internal = ~np.isin(np.arange(self.size), self.ports)
        self.max_weight = max(0.45, 0.26 * gain)

    def simulate(self, inputs, learn=False):
        voltage = np.zeros(self.size)
        previous = np.zeros(self.size, dtype=bool)
        last = np.full(self.size, -10000)
        record = np.empty((len(inputs), self.size), dtype=bool)
        for t, events in enumerate(inputs):
            current = np.bincount(self.post, self.weights * self.sign * previous[self.pre], minlength=self.size)
            voltage = np.maximum(0, 0.85 * voltage + current)
            voltage[self.ports] += 1.1 * events
            available = t - last > 2
            voltage[~available] = 0
            fired = (voltage >= 1) & available
            if learn and fired.any():
                update_excitatory_weights(self, t, last, fired)
            voltage[fired] = 0
            last[fired] = t
            previous = fired
            record[t] = fired
        return record
