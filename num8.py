import numpy as np
import scipy.integrate as integrate

def simpson(f, a, b, p):
    i = 0
    limit = 100
    integral = f(a, p) + f(b, p)
    calculated = []
    results = []
    while(i < limit):
        sum_calc = 0
        n = 2 ** i
        h = (b-a)/n
        for k in range(1, n, 2):
            calculated.insert(k-1, f(a + h*k, p))
        for j in range(0, len(calculated)):
            if(j % 2 == 0):
                sum_calc += 4*calculated[j]
            if(j % 2 == 1):
                sum_calc += 2*calculated[j]
        results.append((integral + sum_calc)*h/3)
        if(i > 1):
            if(np.allclose(results[i-1], results[i], atol = 1e-10, rtol=0)):
                return results[i]
        i = i + 1
    print("Simpson's method doesn't converge")

def romberg(f, a, b, p):
    I = np.zeros((10, 10))
    integral = f(a, p) + f(b, p)
    calculated = []
    results = []
    for i in range(0, 10):
        n = 2 ** i
        h = (b - a) / n
        for k in range(1, n, 2):
            calculated.insert(k-1, 2*f(a + h*k, p))
        
        result = (np.sum(calculated) + integral)*h*0.5 
        I[i, 0] = result

        for j in range(0, i):
            I[i, j+1] = (4**(j+1) * I[i, j] - I[i-1, j]) / (4**(j+1) - 1)
        
        results.append(I[i,i])
        if(i > 1):
            if(np.allclose(results[i], results[i-1], atol = 1e-10, rtol=0)):
                return results[i]
    print("Romberg's method doesn't converge")

def funkcja1(x, p):
    return np.sin(x)

def funkcja2(x, y):
    return np.log(x**2 + y**3 + 1)

def g(y, p):
    return simpson(funkcja2, 0, 1, y)

def g2(y, p):
    return romberg(funkcja2, 0, 1, y)

scipy_result1 = integrate.quad(np.sin, 0, 1)[0]
scipy_result2 = integrate.dblquad(lambda x,y: np.log(x**2 + y**3 + 1), 0, 1, lambda y: 0, lambda y:1)[0]
result1 = simpson(funkcja1, 0, 1, 0)
result2 = romberg(funkcja1, 0, 1, 0)
result3 = simpson(g, 0, 1, 0)
result4 = romberg(g2, 0, 1, 0)

print("Result of (1) using scipy : ")
print(scipy_result1)
print("Result of (1) using simpson method: ")
print(result1)
print("Result of (1) using romberg method: ")
print(result2)
print("Result of (2) using scipy: ")
print(scipy_result2)
print("Result of (2) using simpson method: ")
print(result3)
print("Result of (2) using romberg method: ")
print(result4)