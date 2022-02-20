import numpy as np

# Function that helps create a symmetrical matrix
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())
    
# Creating A1 matrix
a1 = np.zeros((5,5),dtype='float')
a1[0][0] = 2.40827208
a1[0][1] = -0.36066254
a1[0][2] = 0.80575445
a1[0][3] = 0.46309511
a1[0][4] = 1.20708553
a1[1][1] = 1.14839502
a1[1][2] = 0.02576113
a1[1][3] = 0.02672584
a1[1][4] = -1.03949556
a1[2][2] = 2.45964907
a1[2][3] = 0.13824088
a1[2][4] = 0.0472749
a1[3][3] = 2.05614464
a1[3][4] = -0.9434493
a1[4][4] = 1.92753926

a1_s = symmetrize(a1)
print("Matrix A1:")
print(a1_s)

# Creating A2 matrix
a2 = np.zeros((5,5),dtype='float')
a2[0][0] = 2.61370745
a2[0][1] = -0.6334453
a2[0][2] = 0.76061329
a2[0][3] = 0.24938964
a2[0][4] = 0.82783473
a2[1][1] = 1.51060349
a2[1][2] = 0.08570081
a2[1][3] = 0.31048984
a2[1][4] = -0.53591589
a2[2][2] = 2.46956812
a2[2][3] = 0.18519926
a2[2][4] = 0.13060923
a2[3][3] = 2.27845311
a2[3][4] = -0.54893124
a2[4][4] = 2.6276678

a2_s = symmetrize(a2)
print("Matrix A2:")
print(a2_s)

# Creating b and b' vectors
b = np.array([5.40780228, 3.67008677, 3.12306266, -1.11187948, 0.54437218])
b1 = b + np.array([1e-5,0,0,0,0])
print("Vector b: ", b.T)
print("Vector b': ", b1.T)

# Solving Ay=b
y_1 = np.linalg.solve(a1_s,b.T)
y_2 = np.linalg.solve(a2_s,b.T)
y_3 = np.linalg.solve(a1_s,b1.T)
y_4 = np.linalg.solve(a2_s,b1.T)

print("Result of A1y = b: ", y_1)
print("Result of A2y = b: ", y_2)
print("Result of A1y' = b': ", y_3)
print("Result of A2y' = b': ", y_4)

# Checking solutions
print("Checking solutions : ")
print(np.allclose(np.dot(a1_s, y_1), b.T))
print(np.allclose(np.dot(a2_s, y_2), b.T))
print(np.allclose(np.dot(a1_s, y_3), b1.T))
print(np.allclose(np.dot(a2_s, y_4), b1.T))

# Norm of y - y' for A1 and A2
print("Norm of y - y' for A1: ", np.linalg.norm(y_1 - y_3))
print("Norm of y - y' for A2: ", np.linalg.norm(y_2 - y_4))

# Calculating condition numbers
print("Condition number of A1: ", np.linalg.cond(a1_s))
print("Condition number of A2: ", np.linalg.cond(a2_s))

