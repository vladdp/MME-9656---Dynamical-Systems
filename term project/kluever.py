import numpy as np
from numpy import sin, cos, arccos, arctan
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Simulation parameters
days = 216.3
sim_time = 60*60*24*days
dt = 1
num_points = (int) (sim_time / dt)
# print(num_points)

# Initial parameters
r_Earth = 6371000               # Earth radius [m]
mu = 3.986e14                   # Earth's gravitational constant [m^3/s^2]
g = 9.80665                     # Gravitational acceleration Earth
P = 10000                       # input power [W]
I_sp = 3300                     # specific impulse
eta = 0.65                      # engine efficiency
T2W = 3.4e-5                    # thrust to weight ratio
m_dot = -(2*eta*P)/(g*I_sp)**2  # mass rate of change
# print(m_dot)

# Rotation matrix functions
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
m = np.zeros(num_points)
m[0] = 1200

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
e_vec = np.zeros((num_points, 3))
K = np.array([0, 0, 1])
n = np.zeros((num_points, 3))

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

phi_e_star[0] = arctan((norm(pos[0])*sin(nu[0])) /
                       (2*a[0]*(e[0]+cos(nu[0]))))
gamma[0] = arccos((r_p*v_p) / (norm(pos[0])*norm(vel[0])))

c_a[0] = [sin(gamma[0]+phi_a_star), cos(gamma[0]+phi_a_star), 0]
c_e[0] = [sin(gamma[0]+phi_e_star[0]), cos(gamma[0]+phi_e_star[0]), 0]
# the c unit vectors have IC [1, 0, 0]. Is this expected?
# print(c_a[0])
# print(c_e[0])

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

# Weighting functions
G_a = 1
G_e[0] = K_e0_a + K_e1_a * t[0] + K_e2_a * t[0]**2
G_i[0] = K_i0_a + K_i1_a * t[0]

# phi[0] = G_e[0] * phi_e_star[0]
alpha[0] = gamma[0] + phi[0]


c[0] = [sin(alpha[0]), cos(alpha[0]), 0]

theta[0] = omega[0] + nu[0]
beta_star[0] = (np.pi/2)*cos(theta[0])

a_T[0] = T2W * m[0]
# a_T[0] = T2W * m[0] * g
# print(a_T)

beta[0] = G_i[0] * cos(theta[0])
# print(beta[0])

u_RTN[0] = [sin(alpha[0])*cos(beta[0]), cos(alpha[0])*cos(beta[0]), sin(beta[0])]

# u[0] = rot_z(u_RTN[0], omega[0])
# u[0] = rot_x(u[0], i[0])
# u[0] = rot_z(u[0], loan[0])
u[0] = rot_x(u_RTN[0], i[0])
u[0] = rot_z(u_RTN[0], theta[0])

# print(u[0])

# acc[0] = -((mu*pos[0])/np.linalg.norm(pos[0])**3) + a_T[0]*u[0]
# print(acc[0])
acc[0] = -((mu*pos[0])/norm(pos[0])**3)
# print(acc[0])

da[0] = ((2*a[0]**2*norm(vel[0]))/mu) * a_T[0]*cos(phi[0])
de[0] = (a_T[0]/norm(vel[0])) * \
        (2*(e[0]+cos(nu[0]))*cos(phi[0])+r[0]/a[0]*sin(nu[0])*sin(phi[0]))
di[0] = (a_T[0]*r[0]/norm(h[0]))*cos(theta[0])*sin(beta[0])
# print(a[0], da[0])
# print(e[0], de[0])
# print(i[0], di[0])

t_120 = 60*60*24*120 / dt
# print(t_120, len(t))

# The duration of the simulation
test_time = 6000

for k in range(1, test_time):
    vel[k] = vel[k-1] + acc[k-1] * dt
    pos[k] = pos[k-1] + vel[k-1] * dt
    h[k] = np.cross(pos[k], vel[k])

    r[k] = norm(pos[k])

    a[k] = a[k-1] + da[k-1] * dt
    e[k] = e[k-1] + de[k-1] * dt
    i[k] = i[k-1] + di[k-1] * dt

    e_vec[k] = ((norm(vel[k])**2-mu/r[k])*pos[k]-np.dot(pos[k], vel[k])*vel[k])/mu
    p[k] = r[k] * (1 + norm(e_vec[k])*cos(nu[k]))
    n[k] = np.cross(K, h[k])
    omega[k] = arccos((np.dot(n[k], e_vec[k])) / (norm(n)*norm(e_vec[k])))
    if e_vec[k, 2] < 0:
        omega[k] = 2*np.pi - omega[k]
    nu[k] = arccos((np.dot(e_vec[k], pos[k])) / (norm(e_vec[k])*r[k]))
    if np.dot(pos[k], vel[k]) < 0:
        nu[k] = 2*np.pi - nu[k]
    loan[k] = arccos(n[k, 0]/norm(n[k]))
    if n[k, 1] < 0:
        loan[k] = 2*np.pi - loan[k]

    theta[k] = omega[k] + nu[k]

    if k < t_120:
        G_e[k] = K_e0_a + K_e1_a * t[k] + K_e2_a * t[k]**2
        G_i[k] = K_i0_a + K_i1_a * t[k]
    else:
        G_e[k] = K_e0_b + K_e1_b * t[k] + K_e2_b * t[k]**2
        G_i[k] = K_i0_b + K_i1_b * t[k]

    phi_e_star[k] = arctan((r[k]*sin(nu[k]))/(2*a[k]*(e[k]+cos(nu[k]))))
    beta[k] = G_i[k] * cos(theta[k])
    # phi[k] = G_e[k] * phi_e_star[k]

    r_p = r[k]*cos(nu[k])
    v_p = np.sqrt(mu/p[k]) * (-sin(nu[k]))
    gamma[0] = arccos((r_p*v_p) / (norm(pos[k])*norm(vel[k])))
    
    alpha[k] = gamma[k] + phi[k]

    u_RTN[k] = [sin(alpha[k])*cos(beta[k]), cos(alpha[k])*cos(beta[k]), sin(beta[k])]

    # u[k] = rot_z(u_RTN[k], loan[k])
    # u[k] = rot_x(u[k], i[k])
    # u[k] = rot_z(u[k], omega[k])

    u[k] = rot_x(u_RTN[k], i[k])
    u[k] = rot_z(u_RTN[k], theta[k])

    m[k] = m[k-1] + m_dot * dt
    a_T[k] = T2W * m[k]
    # a_T[k] = T2W * m[k] * g

    acc[k] = -((mu*pos[k])/norm(pos[k])**3) + a_T[k]*u[k] * dt

    da[k] = ((2*a[k]**2*norm(vel[k]))/mu) * a_T[k]*cos(phi[k])
    de[k] = (a_T[k]/norm(vel[k])) * \
            (2*(e[k]+cos(nu[k]))*cos(phi[k])+r[k]/a[k]*sin(nu[k])*sin(phi[k]))
    di[k] = (a_T[k]*r[k]/norm(h[k]))*cos(theta[k])*sin(beta[k])

    if k % 100 == 0:
        print(k)

# for k in range(0, 10):
    # print("Pos: ", pos[k])
    # print("Vel: ", vel[k])
    # print("Acc: ", acc[k])
    # print("True Anomaly: ", nu[k])
    # print(r[k])
    # print(norm(e_vec[k]))
    # pass

# e_test = np.zeros(test_time)
# for k in range(0, test_time):
#     e_test[k] = norm(e_vec[k])

# plt.plot(e_test)
# plt.show()

# Plot 3D Trajectory
pos = pos / 1000
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(pos[:test_time, 0], pos[:test_time, 1], pos[:test_time, 2])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_xlim(-10000, 10000)
ax.set_ylim(-10000, 10000)
ax.set_zlim(-10000, 10000)
plt.show()

# Plot control vector
fig, ax = plt.subplots(1, 2)
ax[0].plot(u[:test_time, 0], label='u_x')
ax[0].plot(u[:test_time, 1], label='u_y')
ax[0].plot(u[:test_time, 2], label='u_z')
ax[0].set_xlabel('t (s)')
ax[0].set_ylabel('u')
ax[0].set_title('Unit control vector in ECI')
ax[0].set_box_aspect(1)
ax[0].legend()
ax[1].plot(u_RTN[:test_time, 0], label='u_RTN_x')
ax[1].plot(u_RTN[:test_time, 1], label='u_RTN_y')
ax[1].plot(u_RTN[:test_time, 2], label='u_RTN_z')
ax[1].set_xlabel('t (s)')
ax[1].set_ylabel('u')
ax[1].set_title('Unit control vector in RTN')
ax[1].set_box_aspect(1)
ax[1].legend()
fig.tight_layout()
plt.show()

# Plot e & i vs a
a = a / 1000
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(a[:test_time], e[:test_time], color='blue', label='Eccentricity')
ax2.plot(a[:test_time], np.rad2deg(i[:test_time]), color='red', label='Inclination')
ax1.set_xlabel('Semi-major axis (km)')
ax1.set_ylabel('Eccentricity', color='blue')
# ax1.set_ylim(0, 0.14)
ax2.set_ylabel('Inclination', color='red')
# ax2.set_ylim(0, 30)
fig.tight_layout()
plt.show()

# ax1.plot(a[:test_time], e[:test_time], color='blue', label='Eccentricity')
# ax2.plot(a[:test_time], np.rad2deg(i[:test_time]), color='red', label='Inclination')
# plt.show()

# fig, ax = plt.subplots(2, 3)
# ax[0, 0].plot(t[:test_time], a[:test_time])
# ax[0, 0].set_title('a')
# ax[0, 1].plot(t[:test_time], e[:test_time])
# ax[0, 1].set_title('e')
# ax[0, 2].plot(t[:test_time], np.rad2deg(i[:test_time]))
# ax[0, 2].set_title('i')
# ax[1, 0].plot(t[:test_time], loan[:test_time])
# ax[1, 0].set_title('loan')
# ax[1, 1].plot(t[:test_time], omega[:test_time])
# ax[1, 1].set_title('omega')
# ax[1, 2].plot(t[:test_time], nu[:test_time])
# ax[1, 2].set_title('nu')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(t[:test_time], pos[:test_time, 0])
# ax[0].plot(t[:test_time], pos[:test_time, 1])
# ax[0].plot(t[:test_time], pos[:test_time, 2])
# ax[0].set_title('pos')
# ax[1].plot(t[:test_time], vel[:test_time, 0])
# ax[1].plot(t[:test_time], vel[:test_time, 1])
# ax[1].plot(t[:test_time], vel[:test_time, 2])
# ax[1].set_title('vel')
# ax[2].plot(t[:test_time], acc[:test_time, 0])
# ax[2].plot(t[:test_time], acc[:test_time, 1])
# ax[2].plot(t[:test_time], acc[:test_time, 2])
# ax[2].set_title('acc')
# plt.show()

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(t[:test_time], da[:test_time])
# ax[0].set_title('da')
# ax[1].plot(t[:test_time], de[:test_time])
# ax[1].set_title('de')
# ax[2].plot(t[:test_time], di[:test_time])
# ax[2].set_title('di')
# plt.show()

# fig, ax = plt.subplots(2, 3)
# ax[0, 0].plot(t[:test_time], theta[:test_time])
# ax[0, 0].set_title('theta')
# ax[0, 1].plot(t[:test_time], phi_e_star[:test_time])
# ax[0, 1].set_title('phi_e_star')
# ax[0, 2].plot(t[:test_time], beta[:test_time])
# ax[0, 2].set_title('beta')
# ax[1, 0].plot(t[:test_time], phi[:test_time])
# ax[1, 0].set_title('phi')
# ax[1, 1].plot(t[:test_time], gamma[:test_time])
# ax[1, 1].set_title('gamma')
# ax[1, 2].plot(t[:test_time], alpha[:test_time])
# ax[1, 2].set_title('alpha')
# plt.show()

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(t[:test_time], G_e[:test_time])
# ax[0, 0].set_title('G_e')
# ax[0, 1].plot(t[:test_time], G_i[:test_time])
# ax[0, 1].set_title('G_i')
# ax[1, 0].plot(t[:test_time], m[:test_time])
# ax[1, 0].set_title('m')
# ax[1, 1].plot(t[:test_time], a_T[:test_time])
# ax[1, 1].set_title('a_T')
# plt.show()

# fig, ax = plt.subplots(1, 2)
# ax[0].plot(u[:test_time, 0], label='u_x')
# ax[0].plot(u[:test_time, 1], label='u_y')
# ax[0].plot(u[:test_time, 2], label='u_z')
# ax[0].legend()
# ax[1].plot(u_RTN[:test_time, 0], label='u_RTN_x')
# ax[1].plot(u_RTN[:test_time, 1], label='u_RTN_y')
# ax[1].plot(u_RTN[:test_time, 2], label='u_RTN_z')
# ax[1].legend()
# plt.show()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(a[:test_time], e[:test_time], color='blue', label='Eccentricity')
# ax2.plot(a[:test_time], np.rad2deg(i[:test_time]), color='red', label='Inclination')
# plt.show()