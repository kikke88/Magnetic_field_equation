import matplotlib.pyplot as plt
import numpy as np

#rng = np.arange(40, 0.00001)

data = np.loadtxt('z.txt')
#print(data)
fig = plt.figure()

plt.plot(data)
plt.yscale('log')
#plt.xscale('log')

plt.show()
