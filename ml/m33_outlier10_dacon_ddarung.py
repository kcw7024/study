# [실습]
# 이상치, 결측치 처리를 해서
# 기존 모델의 성능향상이 있는지를 비교

from typing import Counter
import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from collections import Counter

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

# 결측치 처리 시작
print(train_set.isnull().sum()) #결측치를 전부 더한다
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
imputer = KNNImputer()
imputer.fit(train_set)
data2 = imputer.transform(train_set)
print(data2)
print(train_set.isnull().sum()) # 없어졌는지 재확인
# 결측치 처리 끝남


# 이상치 제거 해보자~~~~~~~~~

print(train_set.columns)


def detect_outliers(df,n,features):
    
    outlier_indices = []  
    
    for col in features: 
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5*IQR  
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)           
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n)
        
    return multiple_outliers
 
outliers_drop = detect_outliers(train_set, 2, 
        ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'])

print(train_set.loc[outliers_drop])
train_set.loc[outliers_drop]
print(train_set.loc[outliers_drop])
print(train_set.shape)
train_set = train_set.drop(outliers_drop, axis = 0).reset_index(drop=True)
print(train_set.shape)
print(train_set)


x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

x = np.array(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# 2. 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

model = XGBRegressor()

# 3. 훈련
import time 
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)


# 결과:  0.6730835854043405
# 걸린 시간:  0.21849417686462402