# Vlad Pac

import numpy as np
import matplotlib.pyplot as plt

# Set the constants for the equations
sigma = 10
b = 8/3
r = 30

# Set the initial conditions (found these numbers online?)
x_0 = 2
y_0 = 1
z_0 = 1

# Set simulation time and time increment
dt = 0.01
time = 100

# Calculate the number of points for the attractor given time and 
# time increments
num_points = (int) (100 / dt)

# Declare x, y, z arrays with set size
x = np.zeros(num_points)
y = np.zeros(num_points)
z = np.zeros(num_points)

# Set initial conditions as first values in the array
x[0] = x_0
y[0] = y_0
z[0] = z_0

# Calculate each new point given the equations multiplied by dt
for i in range(1, num_points):
    x[i] = x[i-1] + sigma*(y[i-1]-x[i-1]) * dt
    y[i] = y[i-1] + (r*x[i-1] - y[i-1] - x[i-1]*z[i-1]) * dt
    z[i] = z[i-1] + (x[i-1]*y[i-1] - b*z[i-1]) * dt

# Create a 3d axis for plotting and then plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title('Lorenz attractor r=%.2f' %r)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, linewidth=0.1)

plt.show()
