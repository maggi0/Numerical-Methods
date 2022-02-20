import numpy as np

# matrix size
size = 50

# creating matrix A'
a1 = np.full(size, 9,dtype= np.double)
a2 = np.full(size - 1, 7, dtype = np.double)

# creating u and v vectors
temp = np.ones(size, dtype = np.double)
u = temp.T
v = temp.T

# vector b
temp = np.full(size, 5, dtype = np.double)
b = temp.T

# backsubstitution for A'
def back_substitution(a1, a2, b):

    z = np.zeros_like(b, dtype=np.double);

    z[-1] = b[-1] / a1[-1]

    for i in range(size-2, -1, -1):
        z[i] = (b[i] - a2[i]*z[i+1]) / a1[i]
        
    return z

# creating vectors z i z'
z1 = back_substitution(a1, a2, b)
z2 = back_substitution(a1, a2, u)

vTz2 = 0

for i in range(0, size):
    vTz2 += z2[i]

vz1 = 0

for i in range(0, size):
    vz1 += z1[i]

denominator = 1 + vTz2

nominator = z2 * vz1


# Sherman-Morrison
y = z1 - (nominator/denominator)
print("Solution of y:")
np.set_printoptions(precision = 10)
print(y)

# Checking solution
print("Checking solution: ")
a1 = np.diag(np.full(size, 9,dtype= np.double))
a1 += np.diag(np.full(size - 1, 7, dtype = np.double),1)
a = a1 + u*v.T
print(np.matmul(a,y))