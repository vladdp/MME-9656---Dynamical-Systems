import numpy as np
import matplotlib.pyplot as plt

dt = 0.01

Omega_0 = 1.99E-7           # Angular velocity of Earth's rotation [rad/s]
k = 6.6743E-11              # Gravitational constant [Nm/kg^2]
M_EARTH = 5.972E24          # Mass of Earth (kg)
M_SUN = 1.989E30            # Mass of Sun (kg)
R_EARTHSUN = 147.87E9       # Distance from the Earth to the Sun [m] 
R_EARTHL2 = 1.5E9           # Distance from the Earth to Lagrange Point L2 [m]

C_0 = k * (M_SUN / (R_EARTHSUN + R_EARTHL2)**3 + M_EARTH / R_EARTHL2**3)
R_0 = 12742                 # Diameter of the Earth [km]

C = C_0 / Omega_0**2
# print(C)

X1_bar = np.arange(-.25, 0.25, dt)
X2_bar = np.arange(-.4, 0.4, dt)

X1_bar, X2_bar = np.meshgrid(X1_bar, X2_bar)

u = np.zeros(X1_bar.shape)
v = np.zeros(X2_bar.shape)

rows, cols = X2_bar.shape

for i in range(rows):
    for j in range(cols):
        x1 = X1_bar[i, j]
        x2 = X2_bar[i, j]
        ux_bar = -2*(2*C + 1)*x1 - 2*(np.sqrt(2*C+1))*x2
        if np.abs(ux_bar) >= 1:
            ux_bar = ux_bar / np.abs(ux_bar)
        u[i, j] = x2
        v[i, j] = (2*C+1)*x1+ux_bar

fig, ax = plt.subplots()

ax.streamplot(X1_bar, X2_bar, u, v)

minor_ticks = np.arange(-.25, .25, 0.05)
major_ticks = np.arange(-.4, .4, .1)

ax.set_xticks(minor_ticks)
ax.set_yticks(major_ticks)

ax.set_xlabel(r'$\bar{X}_1$')
ax.set_ylabel(r'$\bar{X}_2$')
plt.grid()
plt.show()