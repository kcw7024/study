from operator import methodcaller
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer #scaling 
# :: QuantileTransformer, RobustScaler ->이상치에 자유로움
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=123   
)

model_list = [RandomForestClassifier(), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), KNeighborsClassifier()]


scalers = [StandardScaler(),MinMaxScaler(),
           MaxAbsScaler(),RobustScaler(),QuantileTransformer(),
           PowerTransformer(method = 'yeo-johnson'),
        #    PowerTransformer(method = 'box-cox')
           ]

for scaler in scalers : 
    name = str(scaler).strip('()')
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #2. 모델
    print("==",name,"="*40)
    for model in model_list :
        model_name = str(model).strip('()')
        #3. 훈련
        model.fit(x_train, y_train)
        #4. 평가, 예측
        y_predict = model.predict(x_test)
        results = accuracy_score(y_test, y_predict)
        print(model_name,":: 의 결과 : ", round(results, 4))

'''
== StandardScaler ========================================
RandomForestClassifier :: 의 결과 :  0.95
LogisticRegression :: 의 결과 :  0.9
KNeighborsClassifier :: 의 결과 :  0.875
DecisionTreeClassifier :: 의 결과 :  0.95
KNeighborsClassifier :: 의 결과 :  0.875
== MinMaxScaler ========================================
RandomForestClassifier :: 의 결과 :  0.95
LogisticRegression :: 의 결과 :  0.825
KNeighborsClassifier :: 의 결과 :  0.875
DecisionTreeClassifier :: 의 결과 :  0.95
KNeighborsClassifier :: 의 결과 :  0.875
== MaxAbsScaler ========================================
RandomForestClassifier :: 의 결과 :  0.95
LogisticRegression :: 의 결과 :  0.825
KNeighborsClassifier :: 의 결과 :  0.875
DecisionTreeClassifier :: 의 결과 :  0.8167
KNeighborsClassifier :: 의 결과 :  0.875
== RobustScaler ========================================
RandomForestClassifier :: 의 결과 :  0.95
LogisticRegression :: 의 결과 :  0.9
KNeighborsClassifier :: 의 결과 :  0.9333
DecisionTreeClassifier :: 의 결과 :  0.95
KNeighborsClassifier :: 의 결과 :  0.9333
== QuantileTransformer ========================================
RandomForestClassifier :: 의 결과 :  0.95
LogisticRegression :: 의 결과 :  0.85
KNeighborsClassifier :: 의 결과 :  0.925
DecisionTreeClassifier :: 의 결과 :  0.95
KNeighborsClassifier :: 의 결과 :  0.925
== PowerTransformer ========================================
RandomForestClassifier :: 의 결과 :  0.9417
LogisticRegression :: 의 결과 :  0.9333
KNeighborsClassifier :: 의 결과 :  0.9417
DecisionTreeClassifier :: 의 결과 :  0.95
KNeighborsClassifier :: 의 결과 :  0.9417

'''


