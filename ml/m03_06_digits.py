import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(x,y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# print(y_test)
# print(y_train)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 


model = LinearSVC()


#3. 컴파일 훈련


use_models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

# model.fit(x_train, y_train)

# #4. 평가, 예측
# result = model.score(x_test, y_test)
# print('결과 acc : ', result)
# y_predict = model.predict(x_test)
# # y_predict = np.argmax(y_predict, axis= 1)
# # y_test = tf.argmax(y_test, axis= 1)
# acc= accuracy_score(y_test, y_predict)
# print('acc스코어 : ', acc) 

for model in use_models :
    model = model()
    name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(name,'의 ACC : ', result)


'''

LinearSVC 의 ACC :  0.9518518518518518
SVC 의 ACC :  0.9907407407407407
Perceptron 의 ACC :  0.9351851851851852
LogisticRegression 의 ACC :  0.9703703703703703
KNeighborsClassifier 의 ACC :  0.9944444444444445
DecisionTreeClassifier 의 ACC :  0.8425925925925926
RandomForestClassifier 의 ACC :  0.9851851851851852


'''