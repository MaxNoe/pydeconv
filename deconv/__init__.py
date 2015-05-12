import numpy as np

from scipy.interpolate import splev, splint
from scipy.optimize import minimize


class Blobel():

    def __init__(
        self,
        n_bins_observed,
        n_bins_target,
        range_observed,
        range_target,
        n_knots,
    ):
        self.n_bins_observed = n_bins_observed
        self.n_bins_target = n_bins_target
        self.range_observed = range_observed
        self.range_target = range_target
        self.n_knots = n_knots
        self.spline_degree = 4

        self.knots = np.linspace(range_true[0], range_true[1], n_knots)
        self.knots = np.concatenate([
            np.ones(self.spline_degree)*range_true[0],
            self.knots,
            np.ones(self.spline_degree) * range_true[1]
        ])

        self.n_knots = len(self.knots) - self.spline_degree - 1

        def splinefunction(x, coefficients):
            return splev(x, (self.knots, coefficients, self.spline_degree))

        self.splinefunction = splinefunction

    def singlespline(self, x, j):
        coefficients = np.zeros(self.n_knots)
        coefficients[j] = 1
        return self.splinefunction(x, coefficients)

    def fit(self, observed, target):

        response_matrix = np.empty((self.n_bins_observed, self.n_knots))

        for j in range(self.n_knots):
            entries, edges = np.histogram(observed,
                                          self.n_bins_observed,
                                          self.range_observed,
                                          weights=self.singlespline(target, j),
                                          )
            binwidth = edges[1] - edges[0]
            response_matrix[:, j] = entries

        response_matrix /= np.reshape(response_matrix.sum(axis=1), (1, -1)).T
        response_matrix *= binwidth

        self.response_matrix_ = response_matrix

    def negLnL(self, params, x):
        if np.any(params < 0):
            return np.inf
        lambd = np.dot(self.response_matrix_, params)
        return np.sum(lambd - self.entries_ * np.log(lambd))

    @timecall
    def predict(self, observed, **kwargs):

        self.entries_, edges = np.histogram(
            observed,
            self.n_bins_observed,
            self.range_observed,
        )

        result = minimize(self.negLnL,
                          args=(self.entries_),
                          x0=np.ones(self.n_knots),
                          method='Powell',
                          **kwargs
                          )

        self.minimize_result_ = result
        self.spline_coefficients_ = result.x

        edges = np.linspace(self.range_target[0],
                            self.range_target[1],
                            self.n_bins_target + 1,
                            )

        def result_spline(x):
            return self.splinefunction(x, self.spline_coefficients_)
        self.result_spline_ = result_spline

        unfolded = np.empty(self.n_bins_target)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            unfolded[i] = splint(
              a, b,
              (self.knots, self.spline_coefficients_, self.spline_degree)
            )

        return unfolded
