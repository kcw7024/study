#from termios import N_PPP
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩


import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
# print(datasets.DESCR)
# print(datasets.feature_names)
# print(x)
# print(y)
# print(x.shape,y.shape) # (150, 4) (150,)
# print("y의 라벨값 : ", np.unique(y))  # y의 라벨값 :  [0 1 2]


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=72)                                                

print(y_test)


n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)

#2. 모델
from sklearn.svm import LinearSVC, SVC

model = SVC() # 리니어 원핫 노필요


#3.4. 컴파일 훈련, 평가, 예측
# model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)

print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

# ACC :  [1.         0.91666667 1.         1.         0.91666667 1.
#  1.         1.         1.         0.83333333]
#  cross_val_score :  0.9667

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict)
#[0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 2 0 0 0 2 2 0 1 1 2 0 2 2 0]
acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)
# cross_val_predict ACC :  0.9666666666666667


