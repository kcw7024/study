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
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

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
RandomForestRegressor :: 의 결과 :  0.739
LinearRegression :: 의 결과 :  0.5976
KNeighborsRegressor :: 의 결과 :  0.6219
DecisionTreeRegressor :: 의 결과 :  0.5108
KNeighborsRegressor :: 의 결과 :  0.6219
== MinMaxScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7423
LinearRegression :: 의 결과 :  0.5976
KNeighborsRegressor :: 의 결과 :  0.6389
DecisionTreeRegressor :: 의 결과 :  0.4765
KNeighborsRegressor :: 의 결과 :  0.6389
== MaxAbsScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7394
LinearRegression :: 의 결과 :  0.5976
KNeighborsRegressor :: 의 결과 :  0.6389
DecisionTreeRegressor :: 의 결과 :  0.4696
KNeighborsRegressor :: 의 결과 :  0.6389
== RobustScaler ========================================
RandomForestRegressor :: 의 결과 :  0.7368
LinearRegression :: 의 결과 :  0.5976
KNeighborsRegressor :: 의 결과 :  0.5898
DecisionTreeRegressor :: 의 결과 :  0.5057
KNeighborsRegressor :: 의 결과 :  0.5898
== QuantileTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.7358
LinearRegression :: 의 결과 :  0.5884
KNeighborsRegressor :: 의 결과 :  0.5983
DecisionTreeRegressor :: 의 결과 :  0.4722
KNeighborsRegressor :: 의 결과 :  0.5983
== PowerTransformer ========================================
RandomForestRegressor :: 의 결과 :  0.7402
LinearRegression :: 의 결과 :  0.5855
KNeighborsRegressor :: 의 결과 :  0.6006
DecisionTreeRegressor :: 의 결과 :  0.4589
KNeighborsRegressor :: 의 결과 :  0.6006
'''


