# 1 357개
# 0 212개

# 라벨 0을 112개 삭제해서 재구성
# smote 넣어서 만들기
# 적용전과 후 비교

import enum
from json import load
from re import X
from tabnanny import verbose
from tkinter.tix import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer


#1. 데이터


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

y1 = np.where(y==0)
del_y = y1[0][100:]
new_x = np.delete(x, del_y, axis=0)
new_y = np.delete(y, del_y)
print(new_y)

x = new_x
y = new_y

#print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))


# def outlier(data_out) : 
#     quartile_1, q2, quartile_3 = np.percentile(data_out,
#                                                [25, 50, 75]) # 25%와 75%의 사분위수를 구함, np.percentile()는 정렬된 데이터를 입력받아 사분위수를 구함
#     print('1사분위수 : ', quartile_1)
#     print('50%사분위수 : ', q2)
#     print('3사분위수 : ', quartile_3)
#     iqr = quartile_3 - quartile_1 # 사분위수를 구함
#     print('IQR : ', iqr)
#     lower_bound = quartile_1 - (iqr * 1.5) # 1.5배 사분위수를 구함
#     upper_bound = quartile_3 + (iqr * 1.5) # 1.5배 사분위수를 구함
#     print('최소값 : ', lower_bound)
#     print('최대값 : ', upper_bound)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound)) # 최소값과 최대값 이상의 값을 찾아서 반환함

# outliers_loc = outlier(y) # 최소값과 최대값 이상의 값을 찾아서 반환함
# print('최소값과 최대값 이상의 값을 찾아서 반환함 : ', outliers_loc)
# print(len(outliers_loc[0])) # 200

# x = np.delete(x, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함
# y = np.delete(y, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함



print(np.unique(y, return_counts=True))

# le = LabelEncoder()
# y = le.fit_transform(y)

# x = np.array(x)
# y = np.array(y)
# y = y.reshape(-1, 1)
# y = newlist

x_train, x_test, y_train, y_test = train_test_split(new_x,new_y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=new_y
                                                    )
#2. 모델

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {'n_estimators' : [100, 200, 300, 500, 400], 'max_depth': [40,30,20,50]}, #epochs
#     {'max_depth' : [6, 8, 10, 12, 20, 14, 40], 'n_jobs' : [-1, 3, 5]},
#     #{'min_samples_leaf' : [3, 5, 7, 10], 'n_estimators':[150, 300, 200], 'max_depth':[7, 8, 9, 10]},
#     #{'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}    
# ]

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알아둘것 논리적인회귀(이지만 이진분류모델!!!)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #결정트리방식의 분류모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #DecisionTree가 앙상블로 되어있는 분류모델 
from xgboost import XGBClassifier

model = RandomForestClassifier()

# model = SVC(C=1, kernel='linear', degree=3)
# model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=2,
#                      refit=True, n_jobs=-1) # 42 * 5(kfold) = 210
#n_jobs = CPU갯수 정의 (-1:제일마지막숫자라서 전부다 쓴다는 뜻.)
#refit = True면 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습

#3. 컴파일 훈련

model.fit(x_train, y_train)

# import time
# start = time.time()
# model.fit(x_train, y_train,
#           early_stopping_rounds=10, 
#           eval_set=[(x_train, y_train), (x_test, y_test)],
#           )
# #Fitting 5 folds for each of 135 candidates, totalling 675 fits
# end = time.time()
# print("최적의 매개변수 : ", model.best_estimator_)
# # 최적의 매개변수 :  SVC(C=1, kernel='linear')
# print("최적의 파라미터 : ", model.best_params_)
# # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# print("best_score_ : ", model.best_score_) # train에 대한 점수
# print("model.score : ", model.score(x_test, y_test)) #test score라서 위와 값이 다름

# #4. 평가, 예측

y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print("model.score : ", score) 
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score: ', f1_score(y_test, y_predict)) 

from imblearn.over_sampling import SMOTE

print("="*30, "SMOTE 적용 후", "="*30)
smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train) 

print(pd.Series(y_train).value_counts())

#2. 모델, #3. 훈련

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

#score = model.score(x_test, y_test)
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score: ', f1_score(y_test, y_predict, )) 

# acc :  0.9782608695652174
# f1_score:  0.9863013698630138
# ============================== SMOTE 적용 후 ==============================
# acc :  0.9565217391304348
# f1_score:  0.971830985915493