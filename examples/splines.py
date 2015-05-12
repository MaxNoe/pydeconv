import numpy as np
import scipy.interpolate as si

import matplotlib.pyplot as plt
plt.style.use('ggplot')

N = 20
degree = 3

# create some sample data and plot it
x = np.linspace(0, 2*np.pi, N)
y = np.sin(x) + np.random.normal(0, 0.1, N)
plt.plot(x, y, "rx", label="Sampled Data")

# define knots and fit a spline to the data
n_internal_knots = 8
internal_knots = np.linspace(0, 2*np.pi, n_internal_knots + 2)[1:-1]
tck = si.splrep(x, y, t=internal_knots, k=degree)


# plot the new spline as well
px = np.linspace(0, 2*np.pi, 1000)
plt.plot(px, si.splev(px, tck=tck), "b-", label="Spline fitted to sample data")
plt.plot(px, np.sin(px), "--", color="black")

# now plot b-spline basis functions and save them in a list
knots, coefficients, degree = tck
basis_functions = []
for i in range(len(coefficients)):
    basis_coeffs = np.zeros_like(coefficients)
    basis_coeffs[i] = coefficients[i]
    basis_functions.append(si.splev(px, (knots, basis_coeffs, degree)))
    plt.plot(px, si.splev(px, (knots, basis_coeffs, degree)), color="0.7")

basis_functions = np.array(basis_functions)
plt.plot(px, basis_functions.sum(axis=0), color="orange", linestyle='dashed',  label="Sum of basis functions")

plt.ylim([-1.5, 1.5])
plt.legend()
plt.title("Visualization of splines")
plt.show()


