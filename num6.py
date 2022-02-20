import numpy as np
import matplotlib.pyplot as plt
import math

def interpolate(opt1, opt2, n, size):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    
    if(opt1 == "a"):
        for i in range(n+1):
            x[i] = -1 + 2*i/n
    elif(opt1 == "b"):
        for i in range(n+1):
            x[i] = np.cos((math.pi*((2*i)+1))/(2*(n+1)))
    else:
        print("Wrong option")
        exit()
    
    if(opt2 == "1"):
        for i in range(n+1):
            y[i] = 1/(1+(25*pow(x[i],2)))
    
    elif(opt2 == "2"):
        for i in range(n+1):
            y[i] = 1/(1+pow(x[i],2))
    
    else:
        print("Wrong option")
        exit()
    
    arr_x = np.zeros(size+1, dtype = 'float')
    
    for i in range(size+1):
        arr_x[i] = -1 + 2*i/size
    
    arr_y = np.zeros(size+1, dtype = 'float')
    
    for k in range(size+1):
        for i in range(n+1):
            s = 1
            for j in range(n+1):
                if i != j:
                    s = s * (arr_x[k] - x[j])/(x[i] - x[j])
            arr_y[k] = arr_y[k] + s * y[i]    
    
    return arr_x, arr_y, x, y

print("Choose the degree of the interpolation polynomial")
n = int(input())

size = 100

arr_x1, arr_y1, x1, y1 = interpolate("a", "1", n, size)
arr_x2, arr_y2, x2, y2 = interpolate("b", "1", n, size)
arr_x3, arr_y3, x3, y3 = interpolate("a", "2", n, size)
arr_x4, arr_y4, x4, y4 = interpolate("b", "2", n, size)

plt.subplot(2,2,1)
plt.plot(arr_x1, arr_y1, label="Interpolation polynomials")
plt.scatter(x1, y1, label="Set of known data points", color = 'red')
plt.title("Plot for method (a) and function (1)")   
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(arr_x2, arr_y2, label="Interpolation polynomials")
plt.scatter(x2, y2, label="Set of known data points", color = 'red')
plt.title("Plot for method (b) and function (1)")   
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(arr_x3, arr_y3, label="Interpolation polynomials")
plt.scatter(x3, y3, label="Set of known data points", color = 'red')
plt.title("Plot for method (a) and function (2)")   
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(arr_x4, arr_y4, label="Interpolation polynomials")
plt.scatter(x4, y4, label="Set of known data points", color = 'red') 
plt.title("Plot for method (b) and function (2)")   
plt.legend(loc='best')

# figure = plt.gcf()
# figure.set_size_inches(12, 7)
# plt.savefig('num63.svg', bbox_inches='tight', dpi=300)