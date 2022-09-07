import numpy as np
import matplotlib.pyplot as plt


def Leaky_ReLU(x):
    return np.maximum(0.01*x, x)

Leaky_ReLU2 = lambda x : np.maximum(0.01*x, x)

x = np.arange(-5, 5, 0.1)
y = Leaky_ReLU2(x)

plt.plot(x, y)
plt.grid()
plt.show()