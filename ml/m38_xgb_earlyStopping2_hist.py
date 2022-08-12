from cgitb import reset
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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



parameters = {'n_estimators' : [100],
              'learning_rate' : [0.1],
              'max_depth' : [3], #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
              'gamma' : [1],
              'min_child_weight' : [1],
              'subsample' : [1],
              'colsample_bytree' : [0.5],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [0]
              } 

#2. 모델 

model = XGBClassifier(random_state=123, 
                      n_estimators=100,
                      learning_rate=0.1,
                      max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                      gamma=1, )

#model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(
          x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)], # < [(훈련하고),(검증한다)]
          #eval_set=[(x_test, y_test)] # eval_set = metrics
          eval_metric = 'error',
          #eval_set = 'error'
          # metric
          # 회귀 : rmse, mae, rmsle ..
          # 이진 : error, auc.., logloss .. 
          # 다중 : merror, mlogloss..
          )

#print('최상의 매개변수 : ', model.best_params_)
#print('최상의 점수 : ', model.best_score_)

results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

#최종 스코어 :  0.9736842105263158
from sklearn.metrics import accuracy_score, r2_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

print("="*70)
hist = model.evals_result()
print(hist)

# [실습]
# 그려봐~~

import matplotlib
import matplotlib.pyplot as plt #그려보자~
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

import matplotlib.pyplot as plt

# 한 그래프에 보임
# plt.figure(figsize=(8,8))
# plt.plot(hist['validation_0']['error'])
# plt.plot(hist['validation_1']['error'])
# plt.show()

# 한 화면에 두개의 그래프로 보임
plt.subplot(2, 1, 1)
plt.plot(hist['validation_0']['error'])
plt.subplot(2, 1, 2)
plt.plot(hist['validation_1']['error'])
plt.show()              