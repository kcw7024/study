# [실습]
# 아까 4가지로 모델을 맹그러봐
# 784개 DNN으로 만든거(최상의 성능인거 // # accuracy :  0.9524166584014893 / 0.996이상) 비교 
# time 체크 / fit에서 하고
# 1. 나의 최고의 DNN
# 2. 나의 최고의 CNN
# 3. PCA 0.95 했을때
# 3. PCA 0.99 했을때
# 3. PCA 0.999 했을때
# 3. PCA 1.0 했을때
# time = ?? # acc = ?? 

from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings(action='ignore')

'''
DNN 결과

#loss :  0.17032746970653534
#accuracy :  0.9514166712760925
============================
acc score : 1.0

CNN 결과

# loss :  0.06842680275440216
# accuracy :  0.9805999994277954

PCA

argmax 사용 0.95 :  154
argmax 사용 0.99 :  331
argmax 사용 0.999 :  486
argmax 사용 1.0 :  713

'''

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x = np.append(x_train, x_test, axis=0)


a=[]
pca = PCA(n_components=x_train.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
a.append(np.argmax(cumsum >= 0.95)+1)
a.append(np.argmax(cumsum >= 0.99)+1)
a.append(np.argmax(cumsum >= 0.999)+1)
a.append(np.argmax(cumsum >= 1.0)+1)
print(a)

model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
for i in range(len(a)):
    n = a[i]
    pca = PCA(n_components=n)
    x2_train = pca.fit_transform(x_train)
    x2_test = pca.transform(x_test)
    start = time.time()
    model.fit(x2_train, y_train, verbose=True)
    end = time.time()
    
    results = model.score(x2_test, y_test)
    print(n, '의 결과: ', results)
    print('시간: ', end-start)