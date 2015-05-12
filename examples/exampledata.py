import numpy as np


def blobel_example(N, bias='blobel', detector_eff='blobel',  sigma=0.1):

    true = np.random.uniform(0, 2, N)
    if bias == 'blobel':
        with_bias = true - 0.05 * true**2
    else:
        with_bias = bias(true)

    if sigma is not None:
        with_smearing = with_bias + np.random.normal(0, sigma, N)
    else:
        with_smearing = with_bias

    if detector_eff == 'blobel':
        eff = 1 - 0.5 * (true - 1)**2
    else:
        eff = detector_eff(true)

    eff_mask = np.random.random(N) < eff
    geom_mask = np.logical_and(with_smearing >= 0, with_smearing <= 2)

    measured = with_smearing[np.logical_and(geom_mask, eff_mask)]
    true = true[np.logical_and(geom_mask, eff_mask)]

    return measured, true


def double_gauss(N, sigma_left=0.1, sigma_right=0.2, smearing_sigma=0.1):

    true = np.random.uniform(0, 0.5, N)
    true[0:N/2] += np.random.normal(0.5, sigma_left, N/2)
    true[N/2:] += np.random.normal(1.2, sigma_right, N/2)

    measured = true + np.random.normal(0, smearing_sigma, N)
    measured += -0.1*(1-true**2)
    return measured, true


