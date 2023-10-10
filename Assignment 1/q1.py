import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5     # birth-rate of sparrows
delta = 0.65    # death-rate of hawks
c = 0.010       # interaction coefficient between sparrows and hawks
gamma = 0.013   # interaction coefficient between hawks and sparrows

# initial conditions
x_0 = 8         # population of sparrows
y_0 = 2         # population of hawks
t = 50          # time in years of the analysis

dt = 0.001                # discrete factor for equations
steps = int(t / dt)

x = np.zeros(steps + 1)     # array to hold sparrow population
y = np.zeros(steps + 1)     # array to hold hawk population

x[0] = x_0
y[0] = y_0

# Calculate x and y values over time using the provided nonlinear equations
# Multiply dx/dt and dy/dt by discrete factor
for i in range(1, steps + 1):
    x[i] = x[i-1] + (x[i-1] * (alpha - c*y[i-1])) * dt
    y[i] = y[i-1] + (y[i-1] * (gamma*x[i-1] - delta)) * dt

t = np.linspace(0, t, steps + 1)

# Plot the prey and predator population fluctuations
plt.plot(t, x, label='Sparrows')
plt.plot(t, y, label='Hawks')
plt.xlabel('Time (years)')
plt.ylabel('Scaled population')
plt.title('Sparrow and Hawk Population over Time')
plt.legend()
plt.show()

# Plot the phase plane
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Portrait')
plt.show()