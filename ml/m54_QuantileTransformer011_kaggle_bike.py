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

# 1. 데이터
path = 'C:/study/_data/bike_sharing/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

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
RandomForestRegressor :: 의 결과 :  0.9065
LinearRegression :: 의 결과 :  0.3868
KNeighborsRegressor :: 의 결과 :  0.4881
DecisionTreeRegressor :: 의 결과 :  0.7907
KNeighborsRegressor :: 의 결과 :  0.4881
== MinMaxScaler ========================================
RandomForestRegressor :: 의 결과 :  0.9059
LinearRegression :: 의 결과 :  0.3868
KNeighborsRegressor :: 의 결과 :  0.5324
DecisionTreeRegressor :: 의 결과 :  0.7882
KNeighborsRegressor :: 의 결과 :  0.5324
== MaxAbsScaler ========================================
RandomForestRegressor :: 의 결과 :  0.9064
LinearRegression :: 의 결과 :  0.3868
KNeighborsRegressor :: 의 결과 :  0.5324
DecisionTreeRegressor :: 의 결과 :  0.7957
KNeighborsRegressor :: 의 결과 :  0.5324
== RobustScaler ========================================
RandomForestRegressor :: 의 결과 :  0.9078
LinearRegression :: 의 결과 :  0.3868
KNeighborsRegressor :: 의 결과 :  0.4682
DecisionTreeRegressor :: 의 결과 :  0.791
KNeighborsRegressor :: 의 결과 :  0.4682
== QuantileTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.9066
LinearRegression :: 의 결과 :  0.3905
KNeighborsRegressor :: 의 결과 :  0.4722
DecisionTreeRegressor :: 의 결과 :  0.7889
KNeighborsRegressor :: 의 결과 :  0.4722
== PowerTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.9049
LinearRegression :: 의 결과 :  0.3949
KNeighborsRegressor :: 의 결과 :  0.4869
DecisionTreeRegressor :: 의 결과 :  0.7915
KNeighborsRegressor :: 의 결과 :  0.4869

'''


