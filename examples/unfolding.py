import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev
from scipy.integrate import quad
from scipy.optimize import minimize


class Blobel():

    def __init__(
        self,
        n_bins_measured,
        n_bins_true,
        range_measured,
        range_true,
        n_knots,
    ):
        self.n_bins_measured = n_bins_measured
        self.n_bins_true = n_bins_true
        self.range_measured = range_measured
        self.range_true = range_true
        self.n_knots = n_knots
        self.spline_degree = 4

        self.knots = np.linspace(range_true[0], range_true[1], n_knots)
        self.knots = np.concatenate([np.ones(self.spline_degree)*range_true[0], self.knots, np.ones(self.spline_degree) * range_true[1]])

        self.n_knots = len(self.knots) - self.spline_degree - 1

        def splinefunction(x, coefficients):
            return splev(x, (self.knots, coefficients, self.spline_degree))

        self.splinefunction = splinefunction

    def singlespline(self, x, j):
        coefficients = np.zeros(self.n_knots)
        coefficients[j] = 1
        return self.splinefunction(x, coefficients)


    def fit(self, measured, true):

        response_matrix = np.empty((self.n_bins_measured, self.n_knots))

        for j in range(self.n_knots):
            entries, _ = np.histogram(measured,
                                      self.n_bins_measured,
                                      self.range_measured,
                                      weights=self.singlespline(true, j),
                                      )
            response_matrix[:, j] = entries

        self.response_matrix_ = response_matrix / response_matrix.sum(axis=1)[:, None]

    def predict(self, measured, **kwargs):

        self.entries_, edges = np.histogram(
            measured,
            self.n_bins_measured,
            self.range_measured,
        )

        def negLnL(params, x):
            if np.any(params < 0):
                return np.inf
            lambd = np.dot(self.response_matrix_, params)
            return np.sum(lambd - self.entries_ * np.log(lambd))


        self._negLnL = negLnL

        result = minimize(negLnL,
                          args=(self.entries_),
                          x0=np.ones(self.n_knots),
                          method='Powell',
                          **kwargs
                          )

        self.minimize_result_ = result
        self.spline_coefficients_ = result.x

        unfolded = np.empty(self.n_bins_true)

        edges = np.linspace(self.range_true[0],
                            self.range_true[1],
                            self.n_bins_true + 1,
                            )

        def result_spline_(x):
            return self.splinefunction(x, self.spline_coefficients_)
        self.result_spline_ = result_spline_

        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            unfolded[i], _ = quad(result_spline_, a, b)

        return unfolded


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
