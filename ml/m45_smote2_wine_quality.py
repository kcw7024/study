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
dataset = pd.read_csv(path + 'winequality_white.csv', sep=';',
                      index_col=None, header=0) # ;로 구분되어있음


print(dataset.shape) # (4898, 12)
print(dataset.describe())
print(dataset.info())

x = dataset.to_numpy()[:, :-1] # numpy로 바꾸고, 마지막 열을 제외하고 x에 저장
y = dataset.to_numpy()[:, -1]
# x = dataset.values[:, :-1] # values는 numpy로 바꿔줌
# y = dataset.values[:, -1]
print(dataset.shape) # (4898, 12)
print(x) # (4898, 11)
print(y) # (4898,)
print(type(x)) # <class 'numpy.ndarray'>
print(type(y)) # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]

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


# newlist = []

# for i in y : 
#     if i <= 5 : 
#         newlist += [0]
#     elif i == 6 :
#         newlist += [1]
#     else :
#         newlist += [2]


print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))



# le = LabelEncoder()
# y = le.fit_transform(y)

# x = np.array(x)
# y = np.array(y)
# y = y.reshape(-1, 1)
# y = newlist

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y
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

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
#scaler = RobustScaler()
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

model.fit(x_train, y_train)


# #4. 평가, 예측

y_predict = model.predict(x_test)
#score = model.score(x_test, y_test)
#print("model.score : ", score) 
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro')) 

# model.score :  0.7510638297872341
# acc :  0.7510638297872341
# f1_score(micro) :  0.7510638297872341

from imblearn.over_sampling import SMOTE

print("="*30, "SMOTE 적용 후", "="*30)
smote = SMOTE(k_neighbors=1,random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train) 

print(pd.Series(y_train).value_counts())

#2. 모델, #3. 훈련

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

#score = model.score(x_test, y_test)
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro')) 


# acc :  0.7414893617021276
# f1_score(micro) :  0.7414893617021278
# ============================== SMOTE 적용 후 ==============================
# acc :  0.7180851063829787
# f1_score(macro) :  0.6715734216364313

# acc :  0.7436170212765958
# f1_score(micro) :  0.7436170212765957
# ============================== SMOTE 적용 후 ==============================
# acc :  0.7404255319148936
# f1_score(macro) :  0.6876887597977411

# acc :  0.7214285714285714
# f1_score(micro) :  0.7214285714285713
# ============================== SMOTE 적용 후 ==============================
# acc :  0.689795918367347
# f1_score(macro) :  0.42569972097790715