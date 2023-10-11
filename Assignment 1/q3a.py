import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
a = 1.5
x_0 = 0.8
k = 50

# Tent map function
def T(x):
    if x < 0.5:
        return a * x
    if x <= 1:
        return a * (1 - x)

x = np.linspace(0, 1, 1000)
y = x

t = [T(i) for i in x]

orbits = np.zeros((2*k+1, 2))       # holds points for the orbit line
orbits[0] = [x_0, 0]                # set ICs
x_k = [x_0]                         # x_k holds all x_k

# calculate x_k+1 and orbit points
for i in range(k):
    x_k.append(T(x_k[-1]))
    orbits[2*i+1] = [x_k[i], x_k[i+1]]
    orbits[2*i+2] = [x_k[i+1], x_k[i+1]]

plt.plot(x, t)
plt.plot(x, y)
plt.plot(orbits[:, 0], orbits[:, 1])
plt.title(r'Orbit when $\mathregular{x_0}$=0.8, k=50')
plt.xlabel(r'$\mathregular{x_k}$')
plt.ylabel(r'$\mathregular{x_k+1}$')
plt.show()

plt.plot(x_k)
plt.title('Time response')
plt.xlabel('k')
plt.ylabel(r'$\mathregular{x_k}$')
plt.show()