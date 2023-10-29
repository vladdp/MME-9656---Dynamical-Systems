# Vlad Pac

import numpy as np
import matplotlib.pyplot as plt

# Set the constants for the system
a = 0.2
b = 0.2
c = 6.3

# Set the initial conditions
x_0 = 1
y_0 = 1
z_0 = 1

# Set simulation time and time increment
dt = 0.01
time = 100

# Calculate the number of points
num_points = (int) (100 / dt)

# Initialize x, y, z arrays with a calculated size
x = np.zeros(num_points)
y = np.zeros(num_points)
z = np.zeros(num_points)

# Assign initial conditions as the first values of each array
x[0] = x_0
y[0] = y_0
z[0] = z_0

# Calculate each new point using the differential equations
for i in range(1, num_points):
    x[i] = x[i-1] - (y[i-1] + z[i-1]) * dt
    y[i] = y[i-1] + (x[i-1] + a * y[i-1]) * dt
    z[i] = z[i-1] + (b + x[i-1]*z[i-1] - c*z[i-1]) * dt
    
# Plot the Rossler attractor
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_title('Rossler Attractor c=%.1f' %c)
ax.set_title('Rossler Attractor (x0, y0, z0)=(%.2f, %.2f, %.2f)' %(x_0, y_0, z_0))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, linewidth=0.5)
plt.show()

# Plot the time series graph
# Create an array to store time values
t = np.arange(0, time, dt)

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_title('Time series (x0, y0, z0)=(%.2f, %.2f, %.2f)' %(x_0, y_0, z_0))
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

# Plot the Poincare Map

r = np.sqrt(x**2 + y**2)

fig, ax = plt.subplots()
ax.plot(r, z)
ax.set_title('Poincare Map (x0, y0, z0)=(%.2f, %.2f, %.2f)' %(x_0, y_0, z_0))
ax.set_xlabel('r')
ax.set_ylabel('z')
plt.show()