from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
#print(x.shape, y.shape) # (178, 13), (178,)
#print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(np.min(x_train))  # 0.0
#print(np.max(x_train))  # 1.0

#print(np.min(x_test))  # 1.0
#print(np.max(x_test))  # 1.0


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

model = RandomForestClassifier()

#3. 컴파일 훈련
# model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)

print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

# ACC :  [1.         1.         0.96       1.         0.91666667] 
#  cross_val_score :  0.9753
# [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0 2
#  2 1 0 0 2 1 0 0 0 1 1 1 0 0 1 1 0]

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
print(y_predict)
#[0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 2 0 0 0 2 2 0 1 1 2 0 2 2 0]
acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)
# cross_val_predict ACC :  1.0


'''
1. LinearSVC 
accuracy :  0.9814814814814815

2. SVC
accuracy :  0.9814814814814815

3. Perceptron
accuracy :  0.9814814814814815

4. LogisticRegression
accuracy :  0.9814814814814815

5. KNeighborsClassifier
accuracy :  0.9444444444444444

6. DecisionTreeClassifier
accuracy :  0.9629629629629629

7. RandomForestClassifier
accuracy :  1.0

'''