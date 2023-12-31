# Vlad Pac

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def V(x, y):
    return 2*x**2 + 2*x*y + y**2

dt = 0.1
X = np.arange(-5, 5, dt)
Y = np.arange(-5, 5, dt)
X, Y = np.meshgrid(X, Y)

Z = V(X, Y)

# Surface plot for V(x, y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_title('Surface Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Contour Map
levels = 20
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, levels)
ax.clabel(contour, inline=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Contour Map')
plt.show()