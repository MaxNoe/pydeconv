__author__ = 'kai'

import matplotlib.pyplot as plt
# plt.style.use('ggplot')

import numpy as np

from examples.exampledata import blobel_example, double_gauss
from deconv.blobel import BlobelUnfold

import scipy.interpolate as si

def main():


    mc_feature, mc_target = double_gauss(1000000)
    measured_data_y, _ = double_gauss(1000000, smearing_sigma=0.01)

    # mc_feature, mc_target = blobel_example(1000000, detector_eff=lambda x: x)
    # measured_data_y, _ = blobel_example(1000000, detector_eff=lambda x: x)


    target_range = [0, 2]
    observed_range = [0, 2]
    n_bins_observed = 40
    n_bins_target = 16

    blobel = BlobelUnfold(n_bins_observed, n_bins_target, observed_range, target_range, 8)

    blobel.fit(mc_feature, mc_target)
    print("fitted")
    result_points = blobel.predict(measured_data_y, use_lsq_start_values=False)
    print("predicted")

    plt.matshow(blobel.response_matrix_)
    plt.colorbar()
    plt.show()
    plt.clf()

    plt.ylim([0, 120000])
    plt.hist(mc_target, range=target_range, bins=n_bins_target, histtype='step', label="Target MC Distribution")
    plt.hist(mc_feature, range=observed_range, bins=n_bins_observed, histtype='step', label="Feature MC Distribution")

    plt.hist(measured_data_y, range=observed_range, bins=n_bins_observed, histtype='step', label="Feature Measurement Distribution")

    px = np.linspace(0, 2.2, 100)

    # plt.plot(px, blobel.fit_spline_(px), "--", color="gray",  label="LSQ Spline Fit to MC")
    plt.plot(px, si.splev(px, tck=blobel.predict_tck_, ext=1), 'b-', lw=1, label="Blobel Fit")

    # r = np.linspace(0, 2, 20, endpoint=True)
    # plt.plot((r[0:-1] + r[1:])*0.5, result_points[:-1], 'bo',  label="resulting points")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
