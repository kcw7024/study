#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time
import numpy as np


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )


n_splits = 10
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

model = LinearRegression()

#3. 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('r2스코어 : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))
# r2스코어 :  [0.33637575 0.38127576 0.3771777  0.24117433 0.30863795 0.44126398
#  0.33003566 0.25774243 0.34464434 0.48767096]
#  cross_val_score :  0.3506
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
#r2스코어 :  0.54327654473472
