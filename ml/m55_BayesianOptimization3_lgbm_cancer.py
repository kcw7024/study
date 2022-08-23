from unittest import result
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'max_depth' : (6, 16),
    'num_leaves' : (24, 64),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)    
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
            subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha) :
    params = {
        'n_estimators' : 500, "learning_rate" : 0.02,
        'max_depth' : int(round(max_depth)), # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)), 
        'subsample' : max(min(subsample,1),0), # 0~1사이의 값
        'colsample_bytree' : max(min(colsample_bytree,1),0), 
        'max_bin' : max(int(round(max_bin)),100), # 무조건 10 이상
        'reg_lambda' : max(reg_lambda,0), # 무조건 양수만
        'reg_alpha' : max(reg_alpha,0),
    }

    #  * :: 여려개의 인자를 받겠다
    # ** :: 키워드 받겠다(딕셔너리형태)    
     
    model = LGBMClassifier(**params)

    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123
                              )

lgb_bo.maximize(init_points=5, n_iter=50)

print(lgb_bo.max)

'''

{'target': 0.9619111259605747, 
    'params': {'colsample_bytree': 0.7483824940156792, 
    'max_bin': 52.851204428062914, 
    'max_depth': 7.938623941629713, 
    'min_child_samples': 14.211120954491978, 
    'min_child_weight': 2.1633531625025366, 
    'num_leaves': 30.597545879405807, 
    'reg_alpha': 17.73290548132164, 
    'reg_lambda': 5.528077604660862, 
    'subsample': 0.6331174927127484}
}

    
'''