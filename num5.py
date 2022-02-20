import numpy as np
import matplotlib.pyplot as plt

# matrix size
size = 100

# iteration limit
limit = 1000

# creating matrix A
a1 = np.full(size, 3, dtype= np.double)
a2 = np.full(size - 1, 1, dtype = np.double)
a3 = np.full(size - 2, 0.2, dtype = np.double)

# l1 = a2
# l2 = a3
# u1 = a2
# u2 = a3
# d = a1

b = np.arange(1, size+1)

a = np.diag(a1)
a += np.diag(a2,1)
a += np.diag(a3,2)
a += np.diag(a2,-1)
a += np.diag(a3,-2)

print("Exact solution using numpy: ")
print(np.linalg.solve(a,b))

d_inv = np.empty_like(a1, dtype = np.double)
for x in range(0, size):
    d_inv[x] = 1/a1[x]
    
d_invb = np.empty_like(b, dtype = np.double)
for x in range(0, size):
    d_invb[x] = d_inv[x]*b[x]
    
d_invl1 = np.empty_like(a2, dtype = np.double)
for x in range(0, size-1):
    d_invl1[x] = d_inv[x+1]*a2[x]
    
d_invl2 = np.empty_like(a3, dtype = np.double)
for x in range(0, size-2):
    d_invl2[x] = d_inv[x+2]*a3[x]
    
d_invu1 = np.empty_like(b, dtype = np.double)
for x in range(0, size-1):
    d_invu1[x] = d_inv[x]*a2[x]
    
d_invu2 = np.empty_like(b, dtype = np.double)
for x in range(0, size-2):
    d_invu2[x] = d_inv[x]*a3[x]
    
# Gauss-Seidel method    
def Gauss_Seidel(x): 
    n = 0
    gs = []
    while(n < limit):   
        temp = np.zeros_like(x, dtype = np.double)
        temp[0] = d_invb[0] - d_invu1[0]*x[1] - d_invu2[0]*x[2]
        temp[1] = d_invb[1] - d_invl1[0]*temp[0] - d_invu1[1]*x[2] - d_invu2[1]*x[3]
        for i in range(2, size - 2):
            temp[i] = d_invb[i] - d_invl1[i-1]*temp[i-1] - d_invl2[i-2]*temp[i-2] - d_invu1[i]*x[i+1] - d_invu2[i]*x[i+2]
        i = size - 2
        temp[i] = d_invb[i] - d_invl1[i-1]*temp[i-1] - d_invl2[i-2]*temp[i-2] - d_invu1[i]*x[i+1]
        i = size - 1
        temp[i] = d_invb[i] - d_invl1[i-1]*temp[i-1] - d_invl2[i-2]*temp[i-2]
        if(np.allclose(x, temp, rtol = 1e-8)):
            break
        gs.append(np.linalg.norm(x - temp))
        x = temp
        n += 1
        if(n == limit):
            print("Doesn't converge with Gauss-Seidel method")
            exit()
    return n, gs, x
    
x1 = np.zeros(size, dtype = np.double)
x2 = np.full(size, 1000, dtype = np.double)
x3 = np.arange(100, dtype = np.double)
x4 = np.linalg.solve(a,b)

n1, gs1, gsx1 = Gauss_Seidel(x1)
n2, gs2, gsx2 = Gauss_Seidel(x2)
n3, gs3, gsx3 = Gauss_Seidel(x3)
n4, gs4, gsx4 = Gauss_Seidel(x4)

# M = -D^(-1)*(L+U)

m_1 = np.empty_like(a2, dtype = np.double)
for x in range(0, size-1):
    m_1[x] = -(a2[x]*d_inv[x])
    
m_2 = np.empty_like(a3, dtype = np.double)
for x in range(0, size-2):
    m_2[x] = -(a3[x]*d_inv[x])
    
m_3 = np.empty_like(a2, dtype = np.double)
for x in range(0, size-1):
    m_3[x] = -(a2[x]*d_inv[x+1])
    
m_4 = np.empty_like(a3, dtype = np.double)
for x in range(0, size-2):
    m_4[x] = -(a3[x]*d_inv[x+2])

# Jacobi method
def Jacobi(x2):
    k = 0
    j = []
    while(k < limit):
        temp2 = np.zeros_like(x2, dtype = np.double)
        temp2[0] = d_invb[0] + m_1[0]*x2[1] + m_2[0]*x2[2]
        temp2[1] = d_invb[1] + m_3[0]*x2[0] + m_1[1]*x2[2] + m_2[1]*x2[3]
        for i in range(2, size - 2):
            temp2[i] = d_invb[i] + m_3[i-1]*x2[i-1] + m_4[i-2]*x2[i-2] + m_1[i]*x2[i+1] + m_2[i]*x2[i+2]
        i = size - 2
        temp2[i] = d_invb[i] + m_3[i-1]*x2[i-1] + m_4[i-2]*x2[i-2] + m_1[i]*x2[i+1]
        i = size - 1
        temp2[i] = d_invb[i] + m_3[i-1]*x2[i-1] + m_4[i-2]*x2[i-2]
        if(np.allclose(x2, temp2, rtol = 1e-8)):
            break
        j.append(np.linalg.norm(x2 - temp2))
        x2 = temp2
        k += 1
        if(k == limit):
            print("Doesn't converge for Jacobi method")
            exit()
    return k, j, x2

k1, j1, jx1 = Jacobi(x1)
k2, j2, jx2 = Jacobi(x2)
k3, j3, jx3 = Jacobi(x3)
k4, j4, jx4 = Jacobi(x4)

xAxis_gs1 = np.arange(1, n1+1)
xAxis_j1 = np.arange(1, k1+1)

plt.subplot(2,2,1)
plt.plot(xAxis_gs1, gs1, label='Gaussa-Seidel')
plt.plot(xAxis_j1, j1, label='Jacobi')
plt.legend(loc='best')
plt.title("Starting points x = [0,0,...,0]")
plt.grid()
plt.yscale('log')
plt.xlabel('Iteration count')
plt.ylabel('Norm x-x*')
plt.show()

xAxis_gs2 = np.arange(1, n2+1)
xAxis_j2 = np.arange(1, k2+1)

plt.subplot(2,2,2)
plt.plot(xAxis_gs2, gs2, label='Gaussa-Seidel')
plt.plot(xAxis_j2, j2, label='Jacobi')
plt.legend(loc='best')
plt.title("Starting points x = [1000,1000,...,1000]")
plt.grid()
plt.yscale('log')
plt.xlabel('Iteration count')
plt.ylabel('Norm x-x*')
plt.show()

xAxis_gs3 = np.arange(1, n3+1)
xAxis_j3 = np.arange(1, k3+1)

plt.subplot(2,2,3)
plt.plot(xAxis_gs3, gs3, label='Gaussa-Seidel')
plt.plot(xAxis_j3, j3, label='Jacobi')
plt.legend(loc='best')
plt.title("Starting points x = [0, 1, ..., 99]")
plt.grid()
plt.yscale('log')
plt.xlabel('Iteration count')
plt.ylabel('Norm x-x*')
plt.show()

xAxis_gs4 = np.arange(1, n4+1)
xAxis_j4 = np.arange(1, k4+1)

plt.subplot(2,2,4)
plt.plot(xAxis_gs4, gs4, label='Gaussa-Seidel')
plt.plot(xAxis_j4, j4, label='Jacobi')
plt.legend(loc='best')
plt.title("Starting points equal to solution")
plt.grid()
plt.yscale('log')
plt.xlabel('Iteration count')
plt.ylabel('Norm x-x*')
plt.show()

# figure = plt.gcf()
# figure.set_size_inches(12, 9)
# plt.savefig('num.png', bbox_inches='tight', dpi=300)