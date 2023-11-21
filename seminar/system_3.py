# Phase portrait code inspired by
# https://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/

import numpy as np
import matplotlib.pyplot as plt

sim_time = 200
dt = 0.01
num_points = (int) (sim_time / dt)

x_0 = 0
y_0 = 0.1

def f(x, y):
    if x == 0 and y == 0:
        return 0   
    if y <= 0:
        return -2*x - 3*y  
    
    a = (x-1)**2 + y**2

    if y > 0 and a > 1:
        return -x + 1 
    if y > 0 and a <= 1:
        return -x + ( (x**2 + y**2) / (2*x) )
    
X = np.arange(-1.5, 3.5, dt)
Y = np.arange(-2.0, 2.5, dt)

X, Y = np.meshgrid(X, Y)

# x = np.zeros(num_points)
# y = np.zeros(num_points)

u = np.zeros(X.shape)
v = np.zeros(Y.shape)

rows, cols = Y.shape
print(Y.shape)

for i in range(rows):
    for j in range(cols):
        x = X[i, j]
        y = Y[i, j]
        dy = f(x, y)
        u[i, j] = y
        v[i, j] = dy

seed_points = np.array([[3, 2, 1, 0, -1],
                        [-2, -2, -2, -2, -2]])

# plt.quiver(X, Y, u, v)
# plt.streamplot(X, Y, u, v, density=[0.2, 0.2], broken_streamlines=False)
plt.streamplot(X, Y, u, v, start_points=seed_points.T, broken_streamlines=False)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

# x[0] = x_0
# y[0] = y_0

# for i in range(1, num_points):
#     x[i] = x[i-1] + y[i-1] * dt
#     y[i] = y[i-1] + f(x[i-1], y[i-1]) * dt

# plt.plot(x, y)
# plt.show()