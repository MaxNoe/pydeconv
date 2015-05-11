import numpy as np

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
