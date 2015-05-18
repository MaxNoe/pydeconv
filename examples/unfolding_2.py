__author__ = 'kai'

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

from examples.exampledata import blobel_example, double_gauss
from deconv.blobel import BlobelUnfold

import scipy.interpolate as si

def main():


    mc_feature, mc_target = double_gauss(1000000)
    measured_data_y, _ = double_gauss(1000000, smearing_sigma=0.05)

    # mc_feature, mc_target = blobel_example(1000000, detector_eff=lambda x: x)
    # measured_data_y, _ = blobel_example(1000000, detector_eff=lambda x: x)


    value_range = [0, 2]
    target_range = [0, 2]

    n_bins_observed = 20
    n_bins_target = 15
    n_knots = 12
    blobel = BlobelUnfold(n_bins_observed, n_bins_target, value_range, value_range, n_knots)
    print(blobel)
    xs = np.linspace(value_range[0], value_range[1], num=1000)
    s = np.zeros_like(xs)
    for i in range(blobel.n_basis_functions):
        s += blobel._spline_basis_function(xs, i)
        plt.plot(xs, blobel._spline_basis_function(xs, i))

    plt.axvspan(blobel.natural_domain[0], blobel.natural_domain[1], facecolor='0.7', alpha=0.5, label="Natural Domain")
    plt.plot(xs, s, label="sum of basis functions")
    # blobel.fit(mc_feature, mc_target)
    # result_points = blobel.predict(measured_data_y, use_lsq_start_values=False)
    #
    # plt.ylim([0, 120000])
    # plt.hist(mc_target, range=target_range, bins=n_bins_target, histtype='step', label="Target MC Distribution")
    # plt.hist(mc_feature, range=value_range, bins=n_bins_observed, histtype='step', label="Feature MC Distribution")
    #
    # plt.hist(measured_data_y, range=value_range, bins=n_bins_observed, histtype='step', label="Feature Measurement Distribution")
    #
    # px = np.linspace(0, 2.2, 100)
    #
    # plt.plot(px, blobel.fit_spline_(px), "--", color="gray",  label="LSQ Spline Fit to MC")
    # knots, coefficients, degree = blobel.predict_tck_
    #
    # for i, c in enumerate(coefficients):
    #     plt.plot(px, blobel._spline_basis_function(px, i)*c)
    #
    # plt.plot(px, si.splev(px, tck=blobel.predict_tck_), 'b-', lw=1, label="Blobel Fit")
    #
    # # r = np.linspace(0, 2, 20, endpoint=True)
    # # plt.plot((r[0:-1] + r[1:])*0.5, result_points[:-1], 'bo',  label="resulting points")
    plt.ylim([-3, 3])
    plt.legend()
    # plt.savefig('fit.png')

    # plt.matshow(blobel.response_matrix_)
    # plt.colorbar()
    # plt.savefig('mat.png')
    plt.show()


if __name__ == '__main__':
    main()
