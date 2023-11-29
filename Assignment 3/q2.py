"""
Solution to MME 9656 Assigment 3 Question 1,
Balancing an Inverted Pendulum

Author: Vlad Pac
Due Date: November 29, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

sim_time = 100
dt = 0.01
t = np.arange(0, sim_time, dt)

alpha = -1
beta = 1
delta = 0.3
omega = 1.2
# alpha = 1
# beta = 5
# delta = 0.02
# omega = 1

gamma = 0.5
# gamma = 3

# z = dx/dt
def z(y, t):
    x, dx = y
    return [dx, gamma*np.cos(omega*t) - delta*dx - alpha*x - beta*x**3]
    
# Initial conditions
x_0 = 0
dx_0 = 0
y_0 = [x_0, dx_0]

# Calculate position and velocity using odeint
x, dx = odeint(z, y_0, t).T

# Calculate acceleration
# d2x = gamma*np.cos(omega*t) - delta*dx - alpha*x - beta*x**3

# Plot time-series
plt.plot(t, x, color='blue', label='Position')
plt.plot(t, dx, color='red', label='Velocity')
# plt.plot(t, d2x)
plt.title('Time Series')
plt.xlabel('Time (s)')
plt.ylabel('Position (m) or Velocity (m/s)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot phase portrait
plt.plot(x, dx, color='blue')
plt.title('Phase Portrait')
plt.xlabel('x (m)')
plt.ylabel(r'$\dot{x}$ $(m/s)$')
plt.tight_layout()
plt.show()

# Plot Poincare section 
t = np.arange(0, 80000, 2*np.pi/omega)
x, dx = odeint(z, y_0, t).T

plt.scatter(x, dx, s=1, color='blue')
plt.title('Poincare Section')
plt.xlabel('x (m)')
plt.ylabel(r'$\dot{x}$ $(m/s)$')
plt.tight_layout()
plt.show()