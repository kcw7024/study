# 실습
# 증폭 후 저장한 데이터 불러와서 
# 완성 및 성능 비교

from cgitb import reset
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

import pickle
path = 'd:/study_data/_save/_xg/'
[x_train, x_test, y_train, y_test] = pickle.load(open(path + 'm45_fetch_pickle1_save.dat', 'rb'))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 

model = XGBClassifier(random_state=123, 
                      n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3, #max_depth : 얕게잡을수록 좋다 너무 깊게잡을수록 과적합 우려 #None = 무한대
                      gamma=1)

#model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(
          x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)], # < [(훈련하고),(검증한다)]
          eval_metric = 'error',
          )

# 불러오기 2.모델, 3.훈련

#4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

from sklearn.metrics import accuracy_score, r2_score, f1_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)
print('f1_score(macro): ', f1_score(y_test, y_predict, average='macro')) 


# 최종 스코어 :  0.3504900906172818
# 진짜 최종 test 점수 :  0.3504900906172818
# f1_score(macro):  0.2018505706936526
