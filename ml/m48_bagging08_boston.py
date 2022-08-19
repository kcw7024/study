from json import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
#from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알ac아둘것 논리적인회귀(이지만 분류모델!!!)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# model = BaggingRegressor(XGBRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=123
#                          )
#bagging 정리할것. 

use_models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]

for model in use_models :
    # model1 = use_models
    model1 = BaggingRegressor(model,
                              n_estimators=100,
                              n_jobs=-1,
                              random_state=123
                              )
    name = str(model).strip('()')    
    #name = model.__class__.__name__
    model1.fit(x_train, y_train)
    result = model1.score(x_test, y_test)
    print(name, '스코어 : ', result)




#3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# print(model.score(x_test, y_test)) 

'''
#XGBRegresor
0.9736842105263158


LinearRegression 스코어 :  0.7679486297115262
KNeighborsRegressor 스코어 :  0.7465061271591057
DecisionTreeRegressor 스코어 :  0.9185052743044945
RandomForestRegressor 스코어 :  0.9015831133076619


'''