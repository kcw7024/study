import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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



parameters = {'n_estimators' : [100],
              'learning_rate' : [0.2],
              'max_depth' : [2], 
              'gamma' : [100],
              'min_child_weight' : [100],
              'subsample' : [1],
              'colsample_bytree' : [0.3],
              'colsample_bylevel' : [0],
              'colsample_bynode' : [0],
              'reg_alpha' : [10],
              'reg_lambda' : [1]
              } 

#2. 모델 

xgb = XGBRegressor(random_state=123)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, r2_score

y_predict = model.predict(x_test)
print("r2 스코어 :" , r2_score(y_test, y_predict))
# accuracy_score : 0.9666666666666667 model.score 와 동일하다.

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 R2 : ", r2_score(y_test, y_pred_best))
# 최적 튠 ACC :  0.9666666666666667
print("걸린시간 : ", round(end_time-start_time, 2))



# 최상의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.20077923073407505    

# 최상의 매개변수 :  {'learning_rate': 0.2, 'n_estimators': 100}
# 최상의 점수 :  0.25834227216709194

# 최상의 매개변수 :  {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 100}
# 최상의 점수 :  0.3240023930027659

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 100}
# 최상의 점수 :  0.3273187022140188

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100}
# 최상의 점수 :  0.39357783734065405

# 최상의 매개변수 :  {'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.39357783734065405

# 최상의 매개변수 :  {'colsample_bytree': 0.3, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.40071544810077847

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bytree': 0.3, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.405656068847982

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0.3, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.405656068847982

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0.3, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'reg_alpha': 10, 'subsample': 1}
# 최상의 점수 :  0.4072030597782466

# 최상의 매개변수 :  {'colsample_bylevel': 0, 'colsample_bynode': 0, 'colsample_bytree': 0.3, 'gamma': 100, 'learning_rate': 0.2, 'max_depth': 2, 'min_child_weight': 100, 'n_estimators': 100, 'reg_alpha': 10, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수 :  0.4072030597782466