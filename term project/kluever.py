import numpy as np
from numpy import sin, cos, arccos, arctan
import matplotlib.pyplot as plt

# Simulation parameters
days = 216.3
sim_time = 60*60*24*days
dt = 1000
num_points = (int) (sim_time / dt)
# print(num_points)

# Initial parameters
r_Earth = 6371000               # Earth radius [m]
mu = 3.986e14                   # Earth's gravitational constant [m^3/s^2]
g = 9.80665                     # Gravitational acceleration Earth
P = 10000                       # input power [W]
m = 1200                        # initial mass [kg]
I_sp = 3300                     # specific impulse
eta = 0.65                      # engine efficiency
T2W = 3.4e-5                    # thrust to weight ratio
m_dot = -(2*eta*P)/(g*I_sp)**2  # mass rate of change

def rot_x(pq, theta):
    R_x = [ [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)] ]
    return np.matmul(R_x, pq)

def rot_y(pq, theta):
    R_y = [ [cos(theta), 0, sin(theta)],
          [0, 1, 0],
          [-sin(theta), 0, cos(theta)] ]
    return np.matmul(R_y, pq)

def rot_z(pq, theta):
    R_z = [ [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0], 
            [0, 0, 1] ]
    return np.matmul(R_z, pq)

# Initialize arrays for variables
pos = np.zeros((num_points, 3))
vel = np.zeros((num_points, 3))
acc = np.zeros((num_points, 3))
h = np.zeros((num_points, 3))

r = np.zeros(num_points)
p = np.zeros(num_points)
a = np.zeros(num_points)
da = np.zeros(num_points)
e = np.zeros(num_points)
de = np.zeros(num_points)
i = np.zeros(num_points)
di = np.zeros(num_points)
loan = np.zeros(num_points)
omega = np.zeros(num_points)
nu = np.zeros(num_points)

phi = np.zeros(num_points)
phi_a_star = 0
phi_e_star = np.zeros(num_points)
beta = np.zeros(num_points)
beta_star = np.zeros(num_points)
alpha = np.zeros(num_points)
theta = np.zeros(num_points)

c_a = np.zeros((num_points, 3))
c_e = np.zeros((num_points, 3))
c = np.zeros((num_points, 3))

gamma = np.zeros(num_points)

G_e = np.zeros(num_points)
G_i = np.zeros(num_points)

a_T = np.zeros(num_points)
u_RTN = np.zeros((num_points, 3))
u = np.zeros((num_points, 3))

# Set initial conditions
a[0] = 550000 + r_Earth             # semi-major axis [m]
e[0] = 0                            # eccentricity
i[0] = np.deg2rad(28.5)             # inclination
loan[0] = 0                         # longitude of the ascending node
omega[0] = 0                        # argument of periapsis
nu[0] = 0                           # true anomaly

# Calculate starting conditions
r[0] = 550000 + r_Earth
p[0] = r[0]*(1+e[0]*cos(nu[0]))

r_p = r[0]*cos(nu[0])
r_q = r[0]*sin(nu[0])
pos[0] = [r_p, r_q, 0]

pos[0] = rot_z(pos[0], omega[0])
pos[0] = rot_x(pos[0], i[0])
pos[0] = rot_z(pos[0], loan[0])

v_p = np.sqrt(mu/p[0]) * (-sin(nu[0]))
v_q = np.sqrt(mu/p[0]) * (e[0] + cos(nu[0]))
vel[0] = [v_p, v_q, 0]

vel[0] = rot_z(vel[0], omega[0])
vel[0] = rot_x(vel[0], i[0])
vel[0] = rot_z(vel[0], loan[0])

h[0] = np.cross(pos[0], vel[0])

phi_e_star[0] = arctan((np.linalg.norm(pos[0])*sin(nu[0])) /
                       (2*a[0]*(e[0]+cos(nu[0]))))
gamma[0] = arccos((r_p*v_p) / (np.linalg.norm(pos[0])*np.linalg.norm(vel[0])))

c_a[0] = [sin(gamma[0]+phi_a_star), cos(gamma[0]+phi_a_star), 0]
c_e[0] = [sin(gamma[0]+phi_e_star[0]), cos(gamma[0]+phi_e_star[0]), 0]
# the c unit vectors have IC [1, 0, 0]. Is this expected?
# print(c_a[0])
# print(c_e[0])

phi[0] = phi_e_star[0]
alpha[0] = gamma[0] + phi[0]

c[0] = [sin(alpha[0]), cos(alpha[0]), 0]

theta[0] = omega[0] + nu[0]
beta_star[0] = (np.pi/2)*cos(theta[0])

a_T[0] = T2W * m
# print(a_T)

# For t < 120 days
K_e0_a = 0
K_e1_a = -1.5e-3
K_e2_a = 0
K_i0_a = -0.33
K_i1_a = -4.3e-3

# For t >= 120 days
K_e0_b = -0.18
K_e1_b = -1.0e-2
K_e2_b = 7e-5
K_i0_b = -0.85
K_i1_b = -5.2e-3

t = np.arange(0, sim_time-dt, dt)

G_a = 1
G_e[0] = K_e0_a + K_e1_a * t[0] + K_e2_a * t[0]**2
G_i[0] = K_i0_a + K_i1_a * t[0]

beta[0] = G_i[0] * cos(theta[0])
# print(beta[0])

u_RTN[0] = [sin(alpha[0])*cos(beta[0]), cos(alpha[0])*cos(beta[0]), sin(beta[0])]

u[0] = rot_z(u_RTN[0], omega[0])
u[0] = rot_x(u[0], i[0])
u[0] = rot_z(u[0], loan[0])
# print(u[0])

# acc[0] = -((mu*pos[0])/np.linalg.norm(pos[0])**3) + a_T[0]*u[0]
# print(acc[0])
acc[0] = -((mu*pos[0])/np.linalg.norm(pos[0])**3)
# print(acc[0])

da[0] = ((2*a[0]**2*np.linalg.norm(vel[0]))/mu) * a_T[0]*cos(phi[0])
de[0] = (a_T[0]/np.linalg.norm(vel[0])) * \
        (2*(e[0]+cos(nu[0]))*cos(phi[0])+r[0]/a[0]*sin(nu[0])*sin(phi[0]))
di[0] = (a_T[0]*r[0]/np.linalg.norm(h[0]))*cos(theta[0])*sin(beta[0])
# print(a[0], da[0])
# print(e[0], de[0])
# print(i[0], di[0])

for i in range(1, num_points):
    vel[i] = vel[i-1] + acc[i-1] * dt
    pos[i] = pos[i-1] + vel[i-1] * dt
    h[i] = np.cross(pos[i], vel[i])

    r[i] = np.linalg.norm(pos[i])

plt.plot(t, r)
plt.show()