__author__ = 'kai'

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

import scipy.interpolate as si

from scipy.optimize import minimize
from scipy.integrate import quad


class BlobelUnfold():

    def __init__(self, n_bins_observed, n_bins_target, range_observed, range_target, n_knots):
        self.n_bins_observed = n_bins_observed
        self.n_bins_target = n_bins_target
        self.range_observed = range_observed
        self.range_target = range_target
        self.n_knots = n_knots
        self.spline_degree = 3

        self.response_matrix_ = None

        knot_distance = (range_target[1] - range_target[0]) / n_knots

        self.knots = np.linspace(range_target[0], range_target[1], num=n_knots)

        # self.knots = np.concatenate([[range_target[0]]*3,
        #                             np.linspace(range_target[0], range_target[1], n_internal_knots + 1),
        #                             [range_target[1]]*3])

        self.n_basis_functions = len(self.knots) - self.spline_degree - 1

        #natural domain according to german wikipedia
        domain_min = self.knots[self.spline_degree - 1]
        domain_max = self.knots[len(self.knots) - self.spline_degree + 1 - 1]
        self.natural_domain = [domain_min, domain_max]

    def _spline_basis_function(self, x, j):
        basis_coefficients = np.zeros(self.n_basis_functions)
        basis_coefficients[j] = 1
        assert len(basis_coefficients) == len(self.knots) - self.spline_degree - 1
        return si.splev(x, (self.knots, basis_coefficients, self.spline_degree))

    def fit(self, mc_feature, mc_target):

        # fit the matrix
        columns = len(self.knots)
        A = np.zeros([self.n_bins_observed, columns])
        # h, _ = np.histogram(mc_feature, bins=self.n_bins_observed, range=self.range_observed, density=True)
        for j in range(columns):
            weights = self._spline_basis_function(mc_target, j)
            h, _ = np.histogram(mc_feature, bins=self.n_bins_observed, range=self.range_observed, weights=weights)
            A[:, j] = h

        for i, s in enumerate(A.sum(axis=1)):
            A[i, :] /= s

        self.response_matrix_ = A

    def predict(self, measured_feature):
        measured_data_histogram_y, data_edges_y = np.histogram(measured_feature, bins=self.n_bins_observed, range=self.range_observed)
        # get new coefficients and create a new spline

        x0 = np.ones_like(self.response_matrix_[0])
        result = minimize(self._negLnL, x0=x0, args=measured_data_histogram_y)



        self.predict_tck_ = (self.knots, result.x, self.spline_degree)

        result_spline_function = lambda x: si.splev(x, self.predict_tck_)

        # integrate piecewise
        result_points = []
        for a, b in zip(data_edges_y[0:-1], data_edges_y[1:]):
            p, _ = quad(result_spline_function, a, b)/(b - a)
            result_points.append(p)

        return np.array(result_points)

    def _negLnL(self, parameter, measured_data_histogram):
        if np.any(parameter < 0):
                return np.inf
        lambd = np.dot(self.response_matrix_, parameter)
        return np.sum(lambd - measured_data_histogram * np.log(lambd))

    def __str__(self):
        return "Unfolding Parameters: \n " \
               "Knots: {}, {} \n" \
               "Natural Domain: {} \n" \
               "Spline Degree: {}".format(self.n_knots, self.knots, self.natural_domain, self.spline_degree)