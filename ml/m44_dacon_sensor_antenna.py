from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

path = './_data/dacon_sensor_antenna/'

train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

# print(train_set.info) #X,Y가 합쳐져있음. 분리해줘야됨 으 
# print(test_set.info)

#1. 데이터

# x,y feature 분리해줌

train_x_set = train_set.filter(regex='X') 
train_y_set = train_set.filter(regex='Y')

test_x_set = test_set.filter(regex='X')
test_y_set = test_set.filter(regex='Y')

# 결측치 확인. 없음
# print(train_x_set.isnull().sum())
# print(train_y_set.isnull().sum())

#2. 모델
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#xgboost parameters
#'n_estimators' : [100, 200, 300, 400, 500, 1000] #디폴트100 / 1~inf / 정수
#'learning_rate' (eta) : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1
#'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf / 정수
#'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf 
#'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100], 디폴트 1 / 0~inf
#'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
#'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
#'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
#'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
#'reg_alpha' (alpha) : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0 / 0~inf / L1 절대값 가중치 규제
#'reg_lambda' (lambda) : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제



parameters = {'n_estimators' : [100,200,300,10,20,40,50],
              'learning_rate' : [0.1, 0.3, 0.2, 1, 0.2, 0.001],
              'max_depth' : [3,2,4,6,7,10], #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
              'gamma' : [1,0,2,4,10,7,8],
              'min_child_weight' : [1, 0.01, 0.001, 0.4, 0.5],
              #'subsample' : [1],
              #'colsample_bytree' : [0.5],
              #'colsample_bylevel' : [1],
              #'colsample_bynode' : [1],
              #'reg_alpha' : [0, 1, 0.2, 0.1, 0.01, 0.001, 2, 10],
              'reg_lambda' : [0, 0.1, 0.2, 0.01, 1, 10]
              } 


xgb = XGBRegressor(random_state=123)
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=8, verbose=2)

model.fit(train_x_set, train_y_set)

print("끝")

#3. 평가
test_x_set = pd.read_csv(path + 'test.csv', index_col=0)

pred = model.predict(test_x_set)
#print("최종스코어", pred)
print('최상의 점수 : ', model.best_score_)
# from sklearn.metrics import accuracy_score, r2_score
# score = r2_score(test_y_set, y_predict)
# print(score)

#최상의 점수 :  0.019634079352757954

submit = pd.read_csv(path+'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = pred[:,idx-1]
print('Done.')

submit.to_csv(path+'submission.csv', index=False)

#최상의 점수 :  0.061938888390985124