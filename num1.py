import numpy as np
import matplotlib.pyplot as plt

# float
f32 = np.float32

# double 
f64 = np.float64 

# method A for double precision
def derivAD(f, x, h):
    return ((f64(f(x + h)) - f64(f(x))) / f64(h))

# method B for double precision
def derivBD(f, x, h):
    return ((f64(f(x + h)) - f64(f(x - h))) / f64(2*h))

# method A for float precision
def derivAF(f, x, h):
    return ((f32(f(x + h)) - f32(f(x))) / f32(h))

# method B for float precision
def derivBF(f, x, h):
    return ((f32(f(x + h)) - f32(f(x - h))) / f32(2*h))

# array containing h for double precision
array_A1 = np.geomspace(1e-16, 1, 100)

# array containing h for float precision 
array_A2 = np.geomspace(1e-7, 1, 100)

# exact derivative
exact_deriv = -(np.sin(0.3))

# calculating discrete derivative with given h
f_AD = np.vectorize(derivAD)
f_BD = np.vectorize(derivBD)
f_AF = np.vectorize(derivAF)
f_BF = np.vectorize(derivBF)

y_AF = f_AF(np.cos, 0.3, array_A2)
y_BF = f_BF(np.cos, 0.3, array_A2)
y_AD = f_AD(np.cos, 0.3, array_A1)
y_BD = f_BD(np.cos, 0.3, array_A1)

# calculating error
yAxis_AF = abs(y_AF - exact_deriv) 
yAxis_BF = abs(y_BF - exact_deriv)
yAxis_AD = abs(y_AD - exact_deriv) 
yAxis_BD = abs(y_BD - exact_deriv) 

# plot for double precision
plt.subplot(1,2,1)
plt.plot(array_A1, yAxis_AD, label='derivative')
plt.plot(array_A1, yAxis_BD, label='derivative2')
plt.legend(loc='best')
plt.title("Double")
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.show()

# plot for float precision
plt.subplot(1,2,2)
plt.plot(array_A2, yAxis_AF, label='derivative')
plt.plot(array_A2, yAxis_BF, label='derivative2')
plt.legend(loc='best')
plt.title("Float")
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.show()
