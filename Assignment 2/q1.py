import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def V(x, y):
    return 2*x**2 + 2*x*y + y**2

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)

Z = V(X, Y)

# Surface plot for V(x, y)
surf = ax.plot_surface(X, Y, Z)
plt.show()