import time
from inspect import Parameter
from unittest import result
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
bayesian_params = {
    'max_depth': (6, 16),
    # 'num_leaves' : (24, 64),
    # 'min_child_samples' : (10, 200),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'colsample_bylevel': (0.5, 1),
    'max_bin': (10, 500),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50)
}


def lgb_hamsu(max_depth, min_child_weight,
              subsample, colsample_bytree, colsample_bylevel, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators': 500, "learning_rate": 0.02,
        'max_depth': int(round(max_depth)),  # 무조건 정수형
        # 'num_leaves' : int(round(num_leaves)),
        # 'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight': int(round(min_child_weight)),
        'subsample': max(min(subsample, 1), 0),  # 0~1사이의 값
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'colsample_bylevel': max(min(colsample_bylevel, 1), 0),
        'max_bin': max(int(round(max_bin)), 100),  # 무조건 10 이상
        'reg_lambda': max(reg_lambda, 0),  # 무조건 양수만
        'reg_alpha': max(reg_alpha, 0),
    }

    #  * :: 여려개의 인자를 받겠다
    # ** :: 키워드 받겠다(딕셔너리형태)

    model = XGBClassifier(**params)

    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              # eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)

    return results


lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123
                              )

lgb_bo.maximize(init_points=5, n_iter=50)

print(lgb_bo.max)

'''
{'target': 1.0, 
    'params': {'colsample_bylevel': 0.6533744581056, 
    'colsample_bytree': 0.8935555548405743, 
    'max_bin': 179.35412914085055, 
    'max_depth': 10.454251510664996, 
    'min_child_weight': 8.00481013633981, 
    'reg_alpha': 3.495119838012323, 
    'reg_lambda': 2.860455343828491, 
    'subsample': 0.8202733832213396}}
'''

# 2. 모델


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.02,
    colsample_bylevel=max(min(0.5, 1), 0),
    colsample_bytree=max(min(0.5, 1), 0),
    max_bin=max(int(round(307.9966197609987)), 100),
    max_depth=int(round(6.0)),
    min_child_weight=int(round(1.0)),
    reg_alpha=max(0.01, 0),
    reg_lambda=max(0.001, 0),
    subsample=max(min(0.5, 1), 0)
)


# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))
print("걸린시간 : ", round(end-start, 2))
