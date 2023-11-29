"""
Solution to MME 9656 Assigment 3 Question 1,
Balancing an Inverted Pendulum

Author: Vlad Pac
Due Date: November 29, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

# Set simulation time and time increment
sim_time = 3
dt = 0.001
num_points = (int) (sim_time / dt)

# Assign given constant values
M = 0.5
m = 0.2
J = 0.006
l = 0.3
c = 0.1
gamma = 0.006
g = 9.81

M_t = M + m
J_t = J + m*l**2
mu = M_t*J_t - (m*l)**2

A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, m**2*l**2*g/mu, -c*J_t/mu, -gamma*l*m/mu],
              [0, M_t*m*g*l/mu, -c*l*m/mu, -gamma*M_t/mu]])

B = np.array([0, 0, J_t/mu, l*m/mu])

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# print(A)
# print(B)

x = np.zeros((num_points, 4))
dx = np.zeros((num_points, 4))
y = np.zeros((num_points, 2))
u = np.zeros(num_points)
du = np.zeros(num_points)
e = np.zeros(num_points)                    # error

# Initial conditions
p_0 = 0
# theta_0 = -0.2
theta_0 = 0
dp_0 = 0
dtheta_0 = 0

x[0][0] = p_0
x[0][1] = theta_0
x[0][2] = dp_0
x[0][3] = dtheta_0

y[0] = np.matmul(C, x[0])

# Set control gains
K_p = 100
K_i = 1000
K_d = 10

# Calculate initial error and control response
e[0] = -y[0][1]
du[0] = K_p*(e[0] / dt) + K_i * e[0] + K_d * (e[0] / dt**2)

# Add impulse
r = 1
u[0] = u[0] + r

# If set to True, a step force is added
# isStep = True
isStep = False

for i in range(1, num_points):
    dx[i-1] = np.matmul(A, x[i-1]) + np.transpose(B) * u[i-1]
    x[i] = x[i-1] + dx[i-1] * dt
    y[i] = np.matmul(C, x[i])

    e[i] = -y[i][1]

    if i == 1:
        du[i] = K_p*((e[i]-e[i-1])/dt) + K_i*e[i] + K_d*((e[i]-2*e[i-1])/dt**2)
    else:
        du[i] = K_p*((e[i]-e[i-1])/dt) + K_i*e[i] + K_d*((e[i]-2*e[i-1]+e[i-2])/dt**2)

    if isStep:
        u[i] = u[i-1] + du[i-1] * dt + r * dt
    else:
        u[i] = u[i-1] + du[i-1] * dt


# Plot the output response
t = np.arange(0, sim_time, dt)

# plt.plot(t, y[:, 1])
# plt.show()

print(y[0])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t, y[:, 0], color='blue')
ax2.plot(t, y[:, 1], color='red')

ax1.set_title('Impulse Response Output')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Cart Position (m)', color='blue')
ax2.set_ylabel('Pendulum Angle (rad)', color='red')

plt.tight_layout()
plt.show()

# Plot the phase portrait
plt.plot(y[:, 1], x[:, 3])
plt.title('Impulse Response Phase Portrait')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot{\theta}$')
plt.tight_layout()
plt.show()

# Plot of the error
# plt.plot(t, e)
# plt.show()