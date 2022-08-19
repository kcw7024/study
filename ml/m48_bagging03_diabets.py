from json import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True
)

# bagging 사용시 scale 필수
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



model = BaggingRegressor(XGBRegressor(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=123
                          )
#bagging 정리할것. 


use_models = [Perceptron(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]

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
model.fit(x_train, y_train)

#4. 평가, 예측
print(model.score(x_test, y_test)) 



'''
Perceptron 스코어 :  0.4179965646464249
KNeighborsRegressor 스코어 :  0.44958409224644336
DecisionTreeRegressor 스코어 :  0.5273824982517303
RandomForestRegressor 스코어 :  0.5531177065533573

'''

