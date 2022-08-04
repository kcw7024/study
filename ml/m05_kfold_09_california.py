#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVC, LinearSVR
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)


'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

model = RandomForestRegressor()


#3. 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('r2스코어 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
# r2스코어 :  [0.81094811 0.78954889 0.81316511 0.78252629 0.79309711] 
#  cross_val_score :  0.7979
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict)
#[2.36927   1.25363   1.7920501 ... 2.71517   2.5296    2.43602  ]
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
#r2스코어 :  0.7717001771064038

