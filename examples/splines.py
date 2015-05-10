import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

N = 20
degree = 3

x = np.linspace(0, 2*np.pi, N)
y = np.sin(x) + np.random.normal(0, 0.1, N)

knots = np.linspace(0, 2*np.pi, 14)[1:-1]

tck = si.splrep(x, y, k=degree, w=0.1*np.ones(N), t=knots)
tcklist = list(tck)

print(len(tck[0]))
print(len(tck[1]))

px = np.linspace(0, 2*np.pi, 1000)
plt.plot(x, y, "rx")
plt.plot(px, si.splev(px, tck), "b-")
plt.plot(px, np.sin(px), "r--")

for i in range(len(tck[1])):
    coeffs = np.zeros_like(tck[1])
    coeffs[i] = tck[1][i]

    plt.plot(px, si.splev(px, (tck[0], coeffs, tck[2])))

plt.show()

