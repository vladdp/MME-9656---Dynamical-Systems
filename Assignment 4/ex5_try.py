import numpy as np

P1 = np.array([[1, 1, 1, 1, 1]])
P2 = np.array([[1, -1, -1, 1, -1]])
P3 = np.array([[-1, 1, -1, 1, 1]])

M = 3
N = 5

I = np.identity(N)

sum = P1 * np.transpose(P1) + P2 * np.transpose(P2) \
        + P3 * np.transpose(P3)

W = (1/N) * sum - (M/N)*I
# print(W * 5) Same as textbook

P4 = np.array([[1, -1, 1, 1, 1]])
P5 = np.array([[0, 1, -1, 1, 1]])
P6 = np.array([[-1, 1, 1, 1, -1]])

# Probe vector x_p
x_p = P1                
# x_p = P2                
# x_p = P3                
x_p = P4                
x_p = P5                
x_p = P6                
# print(x_p)

def v_i(i):
    v = np.sum(W[i]*x_p[0])
    print(v)
    return np.sum(W[i]*x_p[0])
        
def hsgn(v, x_i):
    if v > 0:
        return 1
    if v == 0:
        return x_i
    if v < 0:
        return -1
    
# Generate random order
random_i = np.arange(N)
np.random.shuffle(random_i)
# random_i = [1, 3, 2, 4, 0]    # 4
# random_i = [3, 4, 0, 1, 2]    # 5
random_i = [2, 1, 4, 3, 0]    # 6

# print(np.sum(W[1]*x_p[0]))

# Update the elements by iteration
for i in range(N):
    x_p[0, random_i[i]] = hsgn(v_i(random_i[i]), x_p[0, random_i[i]])

print(x_p)

if np.array_equal(x_p, P1):
    print("The net has converged to P1")

if np.array_equal(x_p, P2):
    print("The net has converged to P2")

if np.array_equal(x_p, P3):
    print("The net has converged to P3")