import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import csv


sim_time = 100
dt = 0.1
t = np.arange(0, sim_time, dt)

alpha = -1
beta = 1
delta = 0.3
gamma = 0.35
omega = 1.2

# z = dx/dt
def z(y, t):
    x, dx = y
    return [dx, gamma*np.cos(omega*t) - delta*dx - alpha*x - beta*x**3]
    
# Initial conditions
x_0 = 0
dx_0 = 0
y_0 = [x_0, dx_0]

# Calculate position and velocity using odeint
x, v = odeint(z, y_0, t).T
a = gamma*np.cos(omega*t) - delta*v - alpha*x - beta*x**3

filename = 'Assignment 4\duffing_100s.csv'

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    for i in range(len(x)):
        writer.writerow([x[i], v[i], a[i]])


# plt.plot(t, x, color='blue')
# plt.plot(t, v, color='red')
# plt.plot(t, a, color='green')
# plt.show()