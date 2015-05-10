import numpy as np
import matplotlib.pyplot as plt

def blobel_example(N, bias='blobel', sigma=0.1):

    true = np.random.uniform(0, 2, N)
    if bias == 'blobel':
        with_bias = true - 0.05 * true**2
    else:
        with_bias = bias(true)

    eff = 1 - 0.5 * (true - 1)**2

    if sigma is not None:
        with_smearing = with_bias + np.random.normal(0, 0.1, N)
    else:
        with_smearing = with_bias

    eff_mask = np.random.random(N) < eff
    geom_mask = np.logical_and(with_smearing >= 0, with_smearing <= 2)

    measured = with_smearing[np.logical_and(geom_mask, eff_mask)]
    true = true[np.logical_and(geom_mask, eff_mask)]

    return measured, true
