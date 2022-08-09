from matplotlib.pyplot import axis
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist

#비지도학습이기 때문에 y값 안부름
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 784)
print(x.shape) # (70000, 784)

#######################################################################
#[실습]
#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95
#0.99
#0.999
#1.0
#힌트 np.argmax
#######################################################################

pca = PCA(n_components=784) 
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)
pca_EVR = pca.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
#print(sum(pca_EVR)) #0.999998352533973
#print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
#cumsum = np.argmax(cumsum,axis=1)
#print(cumsum)
print(np.argwhere(cumsum >= 0.95)[0]+1)
#[154]


