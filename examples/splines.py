import numpy as np
from scipy.interpolate import splev
import matplotlib.pyplot as plt
from matplotlib.style import use
use('ggplot')
plt.rcParams['font.size'] = 14

def b_spline(x, knots, degree, idx):
    coeffs = np.zeros(len(knots) - degree - 1)
    coeffs[idx] = 1
    return splev(x, (knots, coeffs, degree), ext=1)

natural_domain = [0, 1]
degree = 3
n_inner_knots = 10

knots = np.arange(n_inner_knots + 2 * degree)
knots = knots / (np.diff(natural_domain)[0] * (n_inner_knots - 1))
knots -= natural_domain[0] + degree * np.diff(natural_domain) / (n_inner_knots - 1)

coefficients = np.ones(len(knots) - degree - 1)

x = np.linspace(-0.2, 1.2, 1000)

plt.figure()
plt.title('equidistant knots around inner knots')
for i in range(len(coefficients)):
    plt.plot(x, b_spline(x, knots, degree, i))
plt.plot(x, splev(x, (knots, coefficients, degree), ext=1), 'k', lw=1.5, label='sum')
plt.legend()

plt.plot(knots, np.zeros_like(knots), 'ro', mew=0, alpha=0.3)
plt.ylim(-0.05, 1.1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()

inner_knots = np.linspace(natural_domain[0], natural_domain[1], n_inner_knots)
knots = np.empty(n_inner_knots + 2*degree)
knots[degree:-degree] = inner_knots
knots[:degree] = inner_knots[0]
knots[-degree:] = inner_knots[-1]

plt.figure()
plt.title('repeat outer inner knots degree times')
for i in range(len(coefficients)):
    plt.plot(x, b_spline(x, knots, degree, i))
plt.plot(x, splev(x, (knots, coefficients, degree), ext=1), 'k', lw=1.5, label='sum')
plt.legend()

y_knots = np.zeros_like(knots)
y_knots[:degree] = -np.arange(1, degree + 1) * 0.02
y_knots[-degree:] = -np.arange(1, degree + 1) * 0.02
plt.plot(knots, y_knots, 'ro', mew=0, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show()
