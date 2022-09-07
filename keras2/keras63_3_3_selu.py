import numpy as np
import matplotlib.pyplot as plt

alp = 2

def selu(x,alp):
    return (x>0)*x + (x<=0)*(alp*(np.exp(x) - 1))

selu2 = lambda x : (x>0)*x + (x<=0)*(alp*(np.exp(x) - 1))

x = np.arange(-10, 10, 0.1)
y = selu2(x)

plt.plot(x, y)
plt.grid()
plt.show()