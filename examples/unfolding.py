import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev
from scipy.integrate import quad
from scipy.optimize import minimize

from ../deconv import Blobel



if __name__ == '__main__':

    data = np.random.exponential(1, 100000)
    # data = np.random.uniform(0, 2, int(1e6))

    truee = Blobel(
        n_bins_measured=40,
        n_bins_true=10,
        range_measured=[0, 4],
        range_true=[0, 4],
        n_knots=5,
    )

    truee.fit(data, data)
    unfolded = truee.predict(data)

    plt.matshow(truee.response_matrix_)
    plt.colorbar()
    plt.show()

    data, edges = np.histogram(data,
                               bins=truee.n_bins_true,
                               range=truee.range_true,
                               )

    bin_width = edges[1] - edges[0]

    print('num_splines:', len(truee.spline_coefficients_))

    for i in range(truee.n_knots):
        print(quad(lambda x: truee.singlespline(x, i), 0, 3))

    plt.bar(edges[:-1], data, bin_width, label='data', alpha=0.3, lw=0, color='red')
    plt.bar(edges[:-1], unfolded, bin_width, label='unfolded', alpha=0.3, lw=0, color='blue')
    plt.legend()

    print('Num Events Data:', data.sum())
    print('Num Events Unfolding:', int(unfolded.sum()))
    print('ratio:', data.sum()/unfolded.sum())

    plt.show()
