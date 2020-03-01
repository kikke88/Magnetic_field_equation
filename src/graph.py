import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('mpisublSpectrum.data')

fig = plt.figure()

x = np.array([np.float64(x) for x in range(0, 160000)])
y = np.exp(0.000055 * x, dtype = np.float64)
#y += 10**4
plt.plot(y, label='exp^0.000055*x, 1 step in x dim ~ 10^-4 s')

plt.plot(data, label = 'η = 0.1, τ = 10^-4')
plt.yscale('log')

plt.title('Magnetic field energies (log scale)')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.legend()
plt.show()
