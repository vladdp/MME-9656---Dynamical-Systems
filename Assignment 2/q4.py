import numpy as np
import matplotlib.pyplot as plt

sigma = 10
b = 8/3
r = 24.74

x_0 = 2
y_0 = 1
z_0 = 1

dt = 0.01
time = 100

num_points = (int) (100 / dt)

x = np.zeros(num_points)
y = np.zeros(num_points)
z = np.zeros(num_points)

x[0] = x_0
y[0] = y_0
z[0] = z_0

for i in range(1, num_points):
    x[i] = x[i-1] + sigma*(y[i-1]-x[i-1]) * dt
    y[i] = y[i-1] + (r*x[i-1] - y[i-1] - x[i-1]*z[i-1]) * dt
    z[i] = z[i-1] + (x[i-1]*y[i-1] - b*z[i-1]) * dt
    
# print(sigma, b, r)
# print(x[:10])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z, linewidth=0.1)

plt.show()
