import numpy as np


def update_excitatory_weights(cube, t, last, fired):
    # Nearest previous spike on each side, with bounded excitatory plasticity.
    pre_age, post_age = t - last[cube.pre], t - last[cube.post]
    eligible = cube.sign > 0
    pot = eligible & fired[cube.post] & (pre_age <= 4 * cube.tau)
    dep = eligible & fired[cube.pre] & (post_age <= 4 * cube.tau)
    cube.weights[pot] += 0.002 * np.exp(-pre_age[pot] / cube.tau)
    cube.weights[dep] -= 0.002 * cube.depression * np.exp(-post_age[dep] / cube.tau)
    np.clip(cube.weights, 0.01, cube.max_weight, out=cube.weights)
