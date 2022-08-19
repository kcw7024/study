from operator import methodcaller
from sklearn.datasets import load_boston, fetch_california_housing
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=123   
)


model_list = [RandomForestRegressor(), LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]


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
        results = r2_score(y_test, y_predict)
        print(model_name,":: 의 결과 : ", round(results, 4))
    
    

'''
== StandardScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7607
LinearRegression :: 의 결과 :  0.6052
KNeighborsRegressor :: 의 결과 :  0.6374
DecisionTreeRegressor :: 의 결과 :  0.5345
KNeighborsRegressor :: 의 결과 :  0.6374
== MinMaxScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7614
LinearRegression :: 의 결과 :  0.6052
KNeighborsRegressor :: 의 결과 :  0.6555
DecisionTreeRegressor :: 의 결과 :  0.5291
KNeighborsRegressor :: 의 결과 :  0.6555
== MaxAbsScaler ========================================
RandomForestRegressor :: 의 결과 :  0.762
LinearRegression :: 의 결과 :  0.6052
KNeighborsRegressor :: 의 결과 :  0.6555
DecisionTreeRegressor :: 의 결과 :  0.5227
KNeighborsRegressor :: 의 결과 :  0.6555
== RobustScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7607
LinearRegression :: 의 결과 :  0.6052
KNeighborsRegressor :: 의 결과 :  0.6539
DecisionTreeRegressor :: 의 결과 :  0.5293
KNeighborsRegressor :: 의 결과 :  0.6539
== QuantileTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.761
LinearRegression :: 의 결과 :  0.5989
KNeighborsRegressor :: 의 결과 :  0.6585
DecisionTreeRegressor :: 의 결과 :  0.5161
KNeighborsRegressor :: 의 결과 :  0.6585
== PowerTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.7615
LinearRegression :: 의 결과 :  0.5899
KNeighborsRegressor :: 의 결과 :  0.6577
DecisionTreeRegressor :: 의 결과 :  0.5225
KNeighborsRegressor :: 의 결과 :  0.6577
'''


