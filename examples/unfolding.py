import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev
from scipy.integrate import quad
from scipy.optimize import minimize


class Blobel():

    def __init__(
        self,
        bins_measured,
        bins_true,
        range_measured,
        range_true,
        n_knots,
    ):
        self.bins_measured = bins_measured
        self.bins_true = bins_true
        self.range_measured = range_measured
        self.range_true = range_true
        self.n_knots = n_knots

        self.knots = np.linspace(range_true[0], range_true[1], n_knots)

        def splinefunction(x, coefficients):
            return splev(x, (self.knots, coefficients, 3))

        self.splinefunction = splinefunction

    def singlespline(self, x, j):
        coefficients = np.zeros(self.n_knots)
        coefficients[j] = 1
        return self.splinefunction(x, coefficients)


    def fit(self, measured, true):

        response_matrix = np.empty((self.bins_measured, self.n_knots))

        for j in range(self.n_knots):
            print(self.bins_measured)
            entries, _ = np.histogram(measured,
                                      self.bins_measured,
                                      self.range_measured,
                                      weights=self.singlespline(true, j),
                                      )
            response_matrix[:, j] = entries / entries.sum()

        print(response_matrix.shape)

        self.response_matrix_ = response_matrix

    def predict(self, measured, **kwargs):

        self.entries_, edges = np.histogram(
            measured,
            self.bins_measured,
            self.range_measured,
        )

        def negLnL(params, x):
            lambd = np.dot(self.response_matrix_, params)
            print(lambd.shape, self.entries_.shape, self.bins_measured)
            return np.sum(lambd - self.entries_ * np.log(lambd))

        self._negLnL = negLnL

        result = minimize(negLnL,
                          args=(self.entries_),
                          x0=np.ones(self.n_knots),
                          method='Powell',
                          **kwargs
                          )

        return result


if __name__ == '__main__':
    from exampledata import blobel_example

    bias = lambda x: x
    sigma = 0.1

    meas, true = blobel_example(1000000, bias=bias, sigma=sigma)

    truee = Blobel(20, 10, [0, 2], [0, 2], 10)

    truee.fit(meas, true)

    meas, true = blobel_example(1000000, bias=bias, sigma=sigma)

    result = truee.predict(meas)

