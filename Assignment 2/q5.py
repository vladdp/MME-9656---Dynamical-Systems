import numpy as np
import matplotlib.pyplot as plt

a = 0.2
b = 0.2
c = 6.3

x_0 = 0.91
y_0 = 1
z_0 = 1

dt = 0.001
time = 100

num_points = (int) (100 / dt)

x = np.zeros(num_points)
y = np.zeros(num_points)
z = np.zeros(num_points)

x[0] = x_0
y[0] = y_0
z[0] = z_0

for i in range(1, num_points):
    x[i] = x[i-1] - (y[i-1] + z[i-1]) * dt
    y[i] = y[i-1] + (x[i-1] + a * y[i-1]) * dt
    z[i] = z[i-1] + (b + x[i-1]*z[i-1] - c*z[i-1]) * dt
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z, linewidth=0.5)

plt.show()