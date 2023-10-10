import numpy as np
import matplotlib.pyplot as plt

alpha = 1.0     # birth-rate of sparrows
delta = 0.5     # death-rate of hawks
c = 0.1         # interaction coefficient between sparrows and hawks
gamma = 0.02   # interaction coefficient between hawks and sparrows

# initial conditions
x_0 = 100         # population of sparrows
y_0 = 20         # population of hawks
t = 100          # time in years of the analysis
timestep = 0.0001

x = [x_0]       # list to hold sparrow populations
y = [y_0]       # list to hold hawk populations

steps = int(t / timestep)

for i in range(steps):
    x_i = x[-1] + (x[-1] * (alpha - c*y[-1])) * timestep
    y_i = y[-1] + (y[-1] * (gamma*x[-1] - delta)) * timestep

    x.append(x_i)
    y.append(y_i)

t = np.linspace(0, 50, steps+1)

# print(x)
# print(y)

# plotting hawk population gave me negative values?

plt.plot(t, x)
plt.plot(t, y)
plt.show()