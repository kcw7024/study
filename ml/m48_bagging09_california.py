from json import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# model = BaggingRegressor(XGBRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=123
#                           )
#bagging 정리할것. 

use_models = [XGBRegressor(), LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]

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
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', result)
    else:
        print(str(model).strip('()'), '의 스코어: ', result) 
    
    # print(name, '스코어 : ', result)


 

        
# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# print(model.score(x_test, y_test)) 

'''
XGB 의 스코어:  0.8530285541059979
LinearRegression 의 스코어:  0.5844826289140468
KNeighborsRegressor 의 스코어:  0.715095153377484+++++
DecisionTreeRegressor 의 스코어:  0.8123772603511592
RandomForestRegressor 의 스코어:  0.8091307966091522

'''