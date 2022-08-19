from cgitb import reset
import numpy as np
from sklearn.datasets import load_breast_cancer
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

#2. 모델 

model = XGBClassifier()

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

#3. 훈련
# model.fit(
#           x_train, y_train, early_stopping_rounds=10, 
#           eval_set=[(x_train, y_train), (x_test, y_test)], # < [(훈련하고),(검증한다)]
#           eval_metric = 'error',
#           )

# 불러오기 2.모델, 3.훈련
# import joblib
path = 'd:/study_data/_save/_xg/'
# model = joblib.load(path + 'm40_joblib_save.dat')
model.load_model(path + 'm41_xgb1_save_model.dat')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 스코어 : ", results)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

# import pickle
# path = 'd:/study_data/_save/_xg/'
# pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb'))
# import joblib
# joblib.dump(model, (path + 'm40_joblib_save.dat'))

