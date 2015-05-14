import numpy as np

from scipy.interpolate import splev, splint
from scipy.integrate import quad
from scipy.optimize import minimize

from functools import partial


class Blobel():

    def __init__(
        self,
        n_bins_observed,
        n_bins_target,
        range_observed,
        range_target,
        n_inner_knots,
    ):
        self.n_bins_observed = n_bins_observed
        self.n_bins_target = n_bins_target
        self.range_observed = range_observed
        self.range_target = range_target
        self.spline_degree = 3
        self.n_knots = n_inner_knots + 2 * self.spline_degree

        self.n_splines = self.n_knots - self.spline_degree - 1

        dist = (range_target[1] - range_target[0]) / (n_inner_knots - 1)
        self.knots = np.linspace(range_target[0] - self.spline_degree * dist,
                                 range_target[1] + self.spline_degree * dist,
                                 self.n_knots)

    def splinefunction(self, x, coefficients):
        assert self.n_splines == len(coefficients)
        return splev(x, (self.knots, coefficients, self.spline_degree), ext=1)

    def singlespline(self, x, j):
        assert j < self.n_splines
        coefficients = np.zeros(self.n_splines)
        coefficients[j] = 1
        return self.splinefunction(x, coefficients)

    def fit(self, observed, target):

        response_matrix = np.empty((self.n_bins_observed, self.n_splines))

        for j in range(self.n_splines):
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

    def predict(self, observed, **kwargs):

        self.entries_, edges = np.histogram(
            observed,
            self.n_bins_observed,
            self.range_observed,
        )

        result = minimize(self.negLnL,
                          args=(self.entries_),
                          x0=np.ones(self.n_splines),
                          method='Powell',
                          **kwargs
                          )

        self.minimize_result_ = result
        self.spline_coefficients_ = result.x

        self.result_spline_ = partial(self.splinefunction,
                                      coefficients=self.spline_coefficients_,
                                      )

        edges = np.linspace(self.range_target[0],
                            self.range_target[1],
                            self.n_bins_target+1)
        unfolded = np.empty(self.n_bins_target)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            unfolded[i], _ = quad(
                self.result_spline_,
                a, b,
            )

        return unfolded
