from cgitb import reset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.feature_selection import SelectFromModel
import pandas as pd

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
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 

model = XGBRegressor(random_state=123, 
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                      gamma=1, )

#model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(
          x_train, y_train, early_stopping_rounds=200, 
          eval_set=[(x_train, y_train), (x_test, y_test)], # < [(훈련하고),(검증한다)]
          eval_metric = 'error',
          )

results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", r2)

print(model.feature_importances_)

# [0.03986917 0.04455113 0.25548902 0.07593288 0.04910125 0.04870857
#  0.06075545 0.05339111 0.30488744 0.06731401]

thresholds = model.feature_importances_
print("="*80)
for thresh in thresholds : 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # SelectFromModel :: 크거나 같은 컬럼을 다 뽑아준다
    select_x_train = selection.transform(x_train) 
    select_x_test = selection.transform(x_test) 

    selection_model = XGBRegressor(n_jobs=-1,
                                  random_state=123, 
                                  n_estimators=100,
                                  learning_rate=0.1,
                                  max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                                  gamma=1,  
                                  )
    
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    if (score*100) > 86 : 
        print(select_x_train.shape, select_x_test.shape)
        print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
    
    
    
'''
종 스코어 :  -0.8353413908241956
진짜 최종 test 점수 :  -0.8353413908241956
[0.17965588 0.00514241 0.09028392 0.03555241 0.13744648 0.13197927
 0.02914603 0.00340477 0.1737936  0.02648331 0.03412631 0.15298566]        
================================================================================
(8708, 9) (2178, 9)
Thresh=0.029, n=9, R2: 86.48%
(8708, 12) (2178, 12)
Thresh=0.003, n=12, R2: 86.48%
(8708, 10) (2178, 10)
Thresh=0.026, n=10, R2: 86.17%
(8708, 8) (2178, 8)
Thresh=0.034, n=8, R2: 86.10%

'''    

# import xgboost as xg
# print(xg.__version__)    
    
    
    
    