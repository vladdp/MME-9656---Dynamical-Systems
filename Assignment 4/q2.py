import numpy as np

P1 = np.array([[1, 1, 1, -1, 1, 1, -1, 1, -1]])
P2 = np.array([[1, -1, -1, 1, -1, -1, 1, 1, 1]])
P3 = np.array([[1, 1, 1, 1, -1, 1, 1, 1, 1]])

M = 3
N = 9

I = np.identity(N)

sum = P1 * np.transpose(P1) + P2 * np.transpose(P2) \
        + P3 * np.transpose(P3)

W = (1/N) * sum - (M/N)*I

# print(W * 9)

P4 = np.array([[-1, 1, -1, -1, 1, -1, -1, 1, -1]])

