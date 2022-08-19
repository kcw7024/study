from cgitb import reset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.feature_selection import SelectFromModel
import pandas as pd

#1. 데이터
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
    if (score*100) > 75 :
        print(select_x_train.shape, select_x_test.shape)
        print("Thresh=%f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1],score*100))
    
    
    
'''
최종 스코어 :  -1.2682177842511395
진짜 최종 test 점수 :  -1.2682177842511395
[0.3992708  0.22331071 0.10973493 0.02792356 0.03687634 0.07422797
 0.051374   0.04294644 0.03433527]
================================================================================
(1167, 9) (292, 9)
Thresh=0.027924, n=9, R2: 75.01%
(1167, 7) (292, 7)
Thresh=0.036876, n=7, R2: 75.10%
(1167, 5) (292, 5)
Thresh=0.051374, n=5, R2: 75.13%
(1167, 6) (292, 6)
Thresh=0.042946, n=6, R2: 75.45%
(1167, 8) (292, 8)
Thresh=0.034335, n=8, R2: 75.80%

'''    

# import xgboost as xg
# print(xg.__version__)    
    
    
    
    