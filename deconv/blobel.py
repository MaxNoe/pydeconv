__author__ = 'kai'

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

import scipy.interpolate as si

from scipy.optimize import minimize
from scipy.integrate import quad


class BlobelUnfold():

    def __init__(self, n_bins_observed, n_bins_target, range_observed, range_target, n_internal_knots):
        self.n_bins_observed = n_bins_observed
        self.n_bins_target = n_bins_target
        self.range_observed = range_observed
        self.range_target = range_target
        self.n_internal_knots = n_internal_knots
        self.spline_degree = 3
        self.fit_coefficients_ = None
        self.predict_tck_ = None
        self.response_matrix_ = None
        self.fit_spline_ = None

        knot_distance = (range_target[1] - range_target[0]) / n_internal_knots
        self.knots = np.arange(
            range_target[0] - self.spline_degree * knot_distance,
            range_target[1] + self.spline_degree * 1.1*knot_distance,
            knot_distance,
        )

        self.knots = np.concatenate([[range_target[0]]*3,
                                    np.linspace(range_target[0], range_target[1], n_internal_knots + 1),
                                    [range_target[1]]*3])

        self.n_basis_functions = len(self.knots) - self.spline_degree - 1

    def _spline_basis_function(self, x, j):
        basis_coefficients = np.zeros(self.n_basis_functions)
        basis_coefficients[j] = 1
        return si.splev(x, (self.knots, basis_coefficients, self.spline_degree))

    def fit(self, mc_feature, mc_target):

        truth_histogram, edges = np.histogram(mc_target, bins=self.n_bins_target, range=self.range_target)
        bin_center = (edges[0:-1] + edges[1:])*0.5
        spl = si.LSQUnivariateSpline(bin_center, truth_histogram, t=self.knots[4:-4], k=self.spline_degree)
        self.fit_spline_ = spl

        self.fit_coefficients_ = spl.get_coeffs()

        # # now fit a spline to it
        # degree = self.spline_degree
        #
        # # spl = InterpolatedUnivariateSpline(bin_center, truth_histogram, k=degree)
        # n_internal_knots = self.n_internal_knots
        #
        # # internal_knots = np.linspace(self.range_target[0], self.range_target[1], n_internal_knots + 2)[1:-1]

        # fit the matrix
        columns = self.n_basis_functions
        A = np.zeros([self.n_bins_observed, columns])
        for j in range(columns):
            weights = self._spline_basis_function(mc_target, j)
            h, _ = np.histogram(mc_feature, bins=self.n_bins_observed, range=self.range_observed, weights=weights, density=True)
            A[:, j] = h

        self.response_matrix_ = A

    def predict(self, measured_feature, use_lsq_start_values=True):
        measured_data_histogram_y, data_edges_y = np.histogram(measured_feature, bins=self.n_bins_observed, range=self.range_observed)
        # get new coefficients and create a new spline


        x0 = np.ones_like(self.response_matrix_[0])
        if use_lsq_start_values:
            x0 = self.fit_coefficients_
        result_coefficients = minimize(self._negLnL, x0=x0, args=measured_data_histogram_y)


        self.predict_tck_ = (self.knots, result_coefficients.x, self.spline_degree)

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