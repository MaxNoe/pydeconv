import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev
from scipy.integrate import quad
from scipy.optimize import minimize

from deconv import Blobel



if __name__ == '__main__':

    # data = np.random.exponential(1, 100000)
    data = np.random.uniform(0, 4, int(1e7))

    truee = Blobel(
        n_bins_observed=40,
        n_bins_target=20,
        range_observed=[0, 4],
        range_target=[0, 4],
        n_knots=10,
    )
    truee.fit(data, data)

    unfolded = truee.predict(data)

    plt.matshow(truee.response_matrix_)
    plt.colorbar()
    plt.show()

    data, edges = np.histogram(data,
                               bins=truee.n_bins_target,
                               range=truee.range_target,
                               )

    bin_width = edges[1] - edges[0]

    # scale histograms so that area under curve = n_entries
    data /= bin_width
    unfolded /= bin_width

    px = np.linspace(0, 4, 1000)

    for i, coeff in enumerate(truee.spline_coefficients_):
        plt.plot(px, truee.singlespline(px, i) * coeff)

    plt.plot(px, truee.result_spline_(px))
    plt.bar(edges[:-1], data, bin_width, label='data', alpha=0.3, lw=0, color='red')
    plt.bar(edges[:-1], unfolded, bin_width, label='unfolded', alpha=0.3, lw=0, color='blue')
    plt.legend()

    print('Num Events Data:', data.sum())
    print('Num Events Unfolding:', int(unfolded.sum()))
    print('ratio:', data.sum()/unfolded.sum())

    plt.show()
