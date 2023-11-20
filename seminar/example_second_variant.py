import numpy as np
import matplotlib.pyplot as plt

S_SB = 0                    # Surface area of solar batteries
p_sunlight = 0.9E-6         # specific pressure of sunlight on a black body

D_EARTH = 12742000          # Diameter of Earth [m]
Omega_0 = 1.99E-7           # Angular velocity of Earth's rotation [rad/s]
G = 6.6743E-11              # Gravitational constant [Nm/kg^2]
M_EARTH = 5.972E24          # Mass of Earth (kg)
M_SUN = 1.989E30            # Mass of Sun (kg)
R_EARTHSUN = 147.87E9       # Distance from the Earth to the Sun [m] 
R_EARTHL2 = 1.5E9           # Distance from the Earth to Lagrange Point L2 [m]

C_0 = G * (M_SUN / (R_EARTHSUN + R_EARTHL2)**3 + M_EARTH / R_EARTHL2**3)
C = C_0 / Omega_0**2

# Initial Conditions
x_0 = 0.5
y_0 = 0.8
z_0 = 0.5

# This is of dimensionless form
time = 3 * (2*np.pi)
dt = 0.01
num_points = (int) (time / dt)

x1 = np.zeros(num_points)
x2 = np.zeros(num_points)
y1 = np.zeros(num_points)
y2 = np.zeros(num_points)
z1 = np.zeros(num_points)
z2 = np.zeros(num_points)

ux = np.zeros(num_points)
uy = np.zeros(num_points)
uz = np.zeros(num_points)

x1[0] = x_0
x2[0] = 0
y1[0] = y_0
y2[0] = 0
z1[0] = z_0
z2[0] = 0

ux[0] = 0
uy[0] = 0
uz[0] = 0

omega = np.sqrt(C-2)
# print(omega)
k = 1.4

def V1(y1, y2):
    return omega**2*y1**2 + y2**2 - omega**2

def V2(z1, z2):
    return omega**2*z1**2 + z2**2 - omega**2

def V3(y2, z2):
    return y2**2 + z2**2 - omega**2

for i in range(1, num_points):
    x1[i] = x1[i-1] + x2[i-1] * dt
    x2[i] = x2[i-1] + ((2*C+1)*x1[i-1] + 2*y2[i-1] + ux[i-1]) * dt
    ux[i] = -2*(2*C+1)*x1[i-1] - 2*y2[i-1] - 2*np.sqrt(2*C+1)*x2[i-1]

    y1[i] = y1[i-1] + y2[i-1] * dt
    y2[i] = y2[i-1] + ((1-C)*y1[i-1] - 2*x2[i-1] + uy[i-1]) * dt
    uy[i] = -omega**2*y1[i-1] + (C-1)*y1[i-1] - k*(V1(y1[i-1], y2[i-1]) + V3(y2[i-1], z2[i-1]))*y2[i-1]

    z1[i] = z1[i-1] + z2[i-1] * dt
    z2[i] = z2[i-1] + (-C*z1[i-1] + uz[i-1]) * dt
    uz[i] = -omega**2*z1[i-1] + C*z1[i-1] - k*(V2(z1[i-1], z2[i-1]) + V3(y2[i-1], z2[i-1]))*z2[i-1]


fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].plot(x1, x2)
axs[0].set_xlabel('X1')
axs[0].set_ylabel('X2')
axs[0].set_aspect('auto')

axs[1].plot(y1, z1)
axs[1].set_xlabel('Y')
axs[1].set_ylabel('Z')
axs[1].set_aspect('auto')

axs[2].plot(uy, uz)
axs[2].set_xlabel('Uy')
axs[2].set_ylabel('Uz')
axs[2].set_aspect('auto')

fig.tight_layout()
plt.show()