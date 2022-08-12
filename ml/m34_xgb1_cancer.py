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

xgb = XGBClassifier(random_state=123)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train, y_train)

print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

# 최상의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.9626373626373628

# 최상의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.964835164835165

# 최상의 매개변수 :  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수 :  0.9692307692307693

# 최상의 매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'subsample': 1}
# 최상의 점수 :  0.9736263736263737

# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 1}
# 최상의 점수 :  0.9736263736263737

