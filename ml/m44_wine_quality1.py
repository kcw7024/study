#csv로 만들기
#분류모델에서는y라벨 unique를 확인해야한다
from tabnanny import verbose
from tkinter.tix import Tree
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder


#1. 데이터

path = 'D:/study_data/_data/'
datasets = pd.read_csv(path+'winequality_white.csv', sep=';')

print(datasets.shape) # (4898, 12)
#print(datasets.describe())
print(datasets.info())

# pandas 를 numpy로 변환한다
# 1. to_numpy 사용
datasets2 = datasets.to_numpy()
print(type(datasets2))
print(datasets2.shape)

# 2. values 사용
# datasets = datasets.values

# x = datasets.iloc[: , :-1]
# y = datasets.iloc[: , -1]

x = datasets2[:, :11]
y = datasets2[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64)) #numpy로 확인
#print(datasets['quality'].value_counts()) # pandas로 y벨값 확인

# le = LabelEncoder()
# y = le.fit_transform(y)

# x = np.array(x)
# y = np.array(y)
# y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y
                                                    )


#2. 모델

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators' : [100, 200, 300, 500, 400], 'max_depth': [40,30,20,50]}, #epochs
    {'max_depth' : [6, 8, 10, 12, 20, 14, 40], 'n_jobs' : [-1, 3, 5]},
    #{'min_samples_leaf' : [3, 5, 7, 10], 'n_estimators':[150, 300, 200], 'max_depth':[7, 8, 9, 10]},
    #{'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}    
]

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
#scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#print(np.min(x_train))  # 0.0
#print(np.max(x_train))  # 1.0

#print(np.min(x_test))  # 1.0
#print(np.max(x_test))  # 1.0

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
# model.score :  0.7275510204081632
print('acc : ', accuracy_score(y_test, y_predict))
# model.score :  0.7275510204081632
# 다중분류에선 사용하지 않음, 이진분류용
# [None, 'micro', 'macro', 'weighted'] :: 다중분류로 사용하기 위한것 
# print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro')) 
# f1_score(macro) :  0.42630573296196533
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro')) 
# f1_score(micro) :  0.7132653061224491


# from sklearn.metrics import accuracy_score

# y_predict = model.predict(x_test)
# print("accuracy_score :" , accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
# print("걸린시간 : ", round(end-start, 2))


# accuracy_score : 0.6714285714285714
# 최적 튠 ACC :  0.6714285714285714
# 걸린시간 :  126.06

# accuracy_score : 0.6959183673469388
# 최적 튠 ACC :  0.6959183673469388
# 걸린시간 :  84.04

# accuracy_score : 0.7091836734693877
# 최적 튠 ACC :  0.7091836734693877
# 걸린시간 :  84.24

# accuracy_score : 0.710204081632653
# 최적 튠 ACC :  0.710204081632653
# 걸린시간 :  81.15