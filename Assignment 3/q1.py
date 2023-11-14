"""
Solution to MME 9656 Assigment 3 Question 1,
Balancing an Inverted Pendulum

Author: Vlad Pac
Due Date: November 29, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

# Set simulation time and time increment
sim_time = 10
dt = 0.001

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

B = np.array([[0],
              [0],
              [J_t/mu],
              [l*m/mu]])

# print(A)
# print(B)