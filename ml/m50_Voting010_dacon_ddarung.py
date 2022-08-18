# 실습

from logging import warning
from msilib.schema import Binary
from tabnanny import verbose
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier,  VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action='ignore')


# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

xg = XGBRegressor(n_estimators = 100,
                  learning_rate = 0.1,
                  max_depth = 3,
                  gamma = 1,
                  min_child_weight = 1,
                  subsample = 1,
                  colsample_bytree = 0.5,
                  colsample_bylevel = 1,
                  colsample_bynode = 1,
                  reg_alpha = 0,
                  reg_lambda = 0)

lg = LGBMRegressor(num_iterations = 100,
                    learning_rate= 0.1,
                    max_depth=1,
                    min_data_in_leaf = 20,
                    num_leaves=31,
                    #boosting = 'gbdt',
                    bagging_fraction = 1.0,
                    feature_fraction = 1.0,
                    lambda_l1 = 0.0                                        
)

cat = CatBoostRegressor(
                    max_depth=10,
                    learning_rate=0.01,
                    n_estimators=100,
                    #eval_metric='Accuracy',
                    #loss_function='MultiClass',
                    verbose=False
                    )

model = VotingRegressor(  
        estimators = [('XG', xg), ('LG', lg), ('CAT', cat)],
        #voting = 'soft'  # hard 도 있다.
        #회귀모델에서는 없는 파라미터
)

#3. 훈련    
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
# print("보팅 결과 : ", round(score, 4))

# 보팅 결과 :  0.9912

classifiers = [cat, xg, lg]

for model2 in classifiers :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2))

print("보팅 결과 : ", round(score, 4))

'''
CatBoostClassifier 정확도 : 1.0000
XGBClassifier 정확도 : 0.9825
LGBMClassifier 정확도 : 0.9825
보팅 결과 :  1.0


파라미터 튜닝 후

CatBoostRegressor 정확도 : 0.5688
XGBRegressor 정확도 : 0.7562
LGBMRegressor 정확도 : 0.6630
보팅 결과 :  0.7135

'''


