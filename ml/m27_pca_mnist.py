from matplotlib.pyplot import axis
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist



#비지도학습이기 때문에 y값 안부름
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)





