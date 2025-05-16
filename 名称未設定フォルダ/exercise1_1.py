import numpy as np
from math import sin,pi
import matplotlib.pyplot as plt

def true_function(x):
    y = np.sin(pi * x * 0.8) * 10
    return y

x = np.linspace(-1,1,10000)
y = true_function(x)
plt.plot(x,y,label = 'sin')
plt.legend()
plt.savefig('ex1.1.png')
plt.show()