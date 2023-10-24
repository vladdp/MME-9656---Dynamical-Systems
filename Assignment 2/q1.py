import numpy as np
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
surf = ax.plot_surface(X, Y, Z)
plt.show()

# Contour Map
levels = 20
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, levels)
ax.clabel(contour, inline=True)
ax.set_title('Contour Map')
plt.show()