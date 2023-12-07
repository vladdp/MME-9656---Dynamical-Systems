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

# Probe vector x_p
# x_p = P1                
# x_p = P2                
# x_p = P3                
x_p = P4                
# print(x_p)

def v_i(i):
    return np.sum(W[i]*x_p[0])
        
def hsgn(v, x_i):
    if v > 0:
        return 1
    if v == 0:
        return x_i
    if v < 0:
        return -1
    
# Generate array to store random ordered index values
random_i = np.arange(N)

# Update the elements by iteration
for k in range(N):
    print(k)
    np.random.shuffle(random_i)
    x_prev = x_p.copy()

    for i in range(N):
        x_p[0, random_i[i]] = hsgn(v_i(random_i[i]), x_p[0, random_i[i]])

    print(x_prev)
    print(x_p)

    if np.array_equiv(x_p, P1):
        print("The net has converged to P1")
        break
    elif np.array_equiv(x_p, P2):
        print("The net has converged to P2")
        break
    elif np.array_equiv(x_p, P3):
        print("The net has converged to P3")
        break
    elif np.array_equiv(x_p, x_prev):
        print("The net has converged to a spurious state.")
        break
    else:
        print("The net has not converged. Iterating again.")
