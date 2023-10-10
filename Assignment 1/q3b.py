import numpy as np
import matplotlib.pyplot as plt

a = 1.5
x_0 = 0.8
k = 100

def T2(x):
    if x < 1 / (2*a):
        return a**2 * x
    if x < 0.5:
        return a - a**2 * x
    if x < 1 - (1 / (2*a)):
        return a - a**2 + a**2 * x
    if x <= 1:
        return a**2 - a**2 * x
    
x = np.linspace(0, 1, 1000)
y = x

t = [T2(i) for i in x]

orbits = np.zeros((2*k+1, 2))
orbits[0] = [x_0, 0]
x_k = [x_0]

for i in range(k):
    x_k.append(T2(x_k[-1]))
    orbits[2*i+1] = [x_k[i], x_k[i+1]]
    orbits[2*i+2] = [x_k[i+1], x_k[i+1]]

plt.plot(x, y)
plt.plot(x, t)
plt.plot(orbits[:, 0], orbits[:, 1])
plt.title(r'Orbit when $\mathregular{x_0}$=0.8, k=100')
plt.xlabel(r'$\mathregular{x_k}$')
plt.ylabel(r'$\mathregular{x_k+1}$')
plt.show()

plt.plot(x_k)
plt.title(r'Time response when $\mathregular{x_0}$=0.8')
plt.xlabel('k')
plt.ylabel(r'$\mathregular{x_k}$')
plt.show()