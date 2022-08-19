from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xg
print(xg.__version__) # 1.6.1
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
#x = np.append(x_train, x_test, axis=0)
print(x_train.shape, x_test.shape)
#print(np.unique(y_train, return_counts=True))

parameters = [
{'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6]},
{'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.9,1]},
{'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.7,0.9]}
] 

# n_splits = 5
# kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=66)

lda = LinearDiscriminantAnalysis() # y라벨 개수보다 작아야만 한다
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# 2. 모델
model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                                         gpu_id=1), parameters, verbose=1, refit=True, n_jobs=-1)

# 3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
#Fitting 5 folds for each of 42 candidates, totalling 210 fits
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)


# pca = PCA(n_components=x_train.shape[1])
# x = pca.fit_transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)

# pca = PCA(n_components=np.argmax(cumsum >= 0.999)+1)
# x2_train = pca.fit_transform(x_train)
# x2_test = pca.transform(x_test)
# start = time.time()
# model.fit(x2_train, y_train, verbose=1)
# end = time.time()

# results = model.score(x2_test, y_test)
# print('결과: ', results)
# print('시간: ', end-start)

'''
결과:  0.9125
걸린 시간:  242.185932636261
'''