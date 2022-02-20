import numpy as np

# wielkosc macierzy
size = 100

# tworzenie macierzy A

arr_1 = np.empty(size - 1, dtype=np.double)

for x in range(1, size):
    arr_1[x-1] = 0.1/x    

arr_2 = np.empty(size - 2, dtype=np.double)

for x in range(1, size-1):
    arr_2[x-1] = 0.4/pow(x,2)

a = np.diag(np.full(size, 1.2))
a += np.diag(np.full(size - 1, 0.2), -1)
a += np.diag(arr_1, 1)
a += np.diag(arr_2, 2)

# diagonala
a1 = np.full(size, 1.2)
# 1. wstega ponizej diagonali
a2 = np.full(size - 1, 0.2)
# 1. wstega powyzej diagonali
a3 = arr_1
# 2. wstega powyzej diagonali
a4 = arr_2

# wektor x

x_1 = np.arange(1, size+1)

arr_x = x_1.T

# tworzenie macierzy u i l

# diagonala l
l1 = np.ones(size)
# 1. wstega ponizej diagonali l
l2 = np.empty(size - 1)
# diagonala u
u1 = np.empty(size)
# 1. wstega powyzej diagonali u
u2 = np.empty(size - 1)
# 2. wstega powyzej diagonali u
u3 = np.empty(size - 2)

u1[0] = a1[0]
u2[0] = a3[0]
u3[0] = a4[0]
l2[0] = a2[0]/a1[0]

for x in range(1, size - 2):
    u1[x] = a1[x] - l2[x-1]*u2[x-1]
    u2[x] = a3[x] - l2[x-1]*u3[x-1]
    u3[x] = a4[x]
    l2[x] = a2[x]/u1[x]

x = size - 2
u1[x] = a1[x] - l2[x-1]*u2[x-1]
u2[x] = a3[x] - l2[x-1]*u3[x-1]
l2[x] = a2[x]/u1[x]

x = size - 1
u1[x] = a1[x] - l2[x-1]*u2[x-1]

# Obliczanie wyznacznika macierzy A


det = 1

for x in range (0, size):
    det *= u1[x] 

print("Wyznacznik macierzy: ")
print(det)

print("Wyznacznik macierzy korzystajac z biblioteki numpy: ")
print(np.linalg.det(a))

# forward_substitution dla mojej macierzy L
def forward_substitution(l1, l2, x):
    z = np.zeros_like(x, dtype=np.double);
    
    z[0] = x[0] 
    
    for i in range(1, size):
        z[i] = (x[i] - (l2[i-1]*z[i-1]))
        
    return z

#back_substitution dla mojej macierzy U
def back_substitution(u1, u2, u3, z):

    y = np.zeros_like(z, dtype=np.double);

    y[-1] = z[-1] / u1[-1]
    y[-2] = (z[-2] - u2[-1]*y[-1]) / u1[-2]

    for i in range(size-3, -1, -1):
        y[i] = (z[i] - u2[i]*y[i+1] - u3[i]*y[i+2]) / u1[i]
        
    return y

z = forward_substitution(l1, l2, arr_x)

y = back_substitution(u1, u2, u3, z)

np.set_printoptions(precision = 7, suppress = 'true')
print("Rozwiazanie rownania dla y = A^(-1)*x: ")
print(y)

print("Sprawdzenie rozwiazania (mnozenie A z y): ")
print(np.dot(a, y))