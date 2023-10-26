# Vlad Pac

import numpy as np
from numpy import arctan, log, cos, pi

from matplotlib import cm
import matplotlib.pyplot as plt

gamma = 0.7
dt = 0.1

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = np.arange(-1+dt, 1, dt)
Y = np.arange(-1+dt, 1, dt)
X, Y = np.meshgrid(X, Y)

# a1 = (2 / pi) * arctan( (gamma*pi*X) / 2)
# a2 = (2 / pi) * arctan( (gamma*pi*Y) / 2)

# log_cos_a1 = log(cos(0.5*pi*a1))
# log_cos_a2 = log(cos(0.5*pi*a2))

# Example 3 from textbook
# Z = -a1*a2 - (4 / (gamma*np.pi**2)) * (log_cos_a1 + log_cos_a2)

# Example 4 from textbook
# Z = -(a1**2 + a2**2) - (4/(gamma*pi**2))*(log_cos_a1 + log_cos_a2)


# I don't know why the equation does not work with a1 and a2
# The course notes only uses X and Y and that seems to work
Z = -X*Y - (4/(gamma*pi**2))*(log(cos(0.5*pi*X)) + log(cos(0.5*pi*Y)))

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_title(r'Surface plot $\gamma$=0.7')
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel('x')
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel('y')
ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_zlabel('z')
ax.set_zlim(-0.1, 1)
plt.show()

fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, 20)
ax.clabel(contour, inline=True)
ax.set_title(r'Contour Map $\gamma$=0.7')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

gamma = 2

Z = -X*Y - (4/(gamma*pi**2))*(log(cos(0.5*pi*X)) + log(cos(0.5*pi*Y)))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_title(r'Surface plot $\gamma$=2.0')
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel('x')
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel('y')
ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_zlabel('z')
ax.set_zlim(-0.1, 1)
plt.show()

fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, 20)
ax.set_title(r'Surface plot $\gamma$=2.0')
ax.clabel(contour, inline=True)
ax.set_title(r'Contour Map $\gamma$=2.0')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()