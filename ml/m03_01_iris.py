import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
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


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(y_test)
print(y_train)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알아둘것 논리적인회귀(이지만 분류모델!!!)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #결정트리방식의 분류모델
from sklearn.ensemble import RandomForestClassifier #DecisionTree가 앙상블로 되어있는 분류모델 


model = RandomForestClassifier() # 리니어 원핫 노필요


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

LinearSVC 의 ACC :  0.9666666666666667
SVC 의 ACC :  0.9666666666666667
Perceptron 의 ACC :  0.9333333333333333
LogisticRegression 의 ACC :  1.0
KNeighborsClassifier 의 ACC :  0.9666666666666667
DecisionTreeClassifier 의 ACC :  0.9666666666666667
RandomForestClassifier 의 ACC :  0.9


'''