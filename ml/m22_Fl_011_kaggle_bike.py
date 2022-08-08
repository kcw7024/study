import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
path = 'C:/study/_data/bike_sharing/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x = np.array(x)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import matplotlib.pyplot as plt

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1] #features
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Impotances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)
    

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier


models = [DecisionTreeRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),XGBRegressor()]


# for i in range(len(models)) :
#     model = models[i]
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     fimp = model.feature_importances_
#     print("="*100)
#     print(name,'의 결과값 : ', result)
#     print('model.feature_importances : ', fimp)
#     print("="*100)  
# #     plt.subplot(2, 2, i+1)
# #     plot_feature_importances(models[i])
# #     if str(models[i]).startswith("XGB") : 
# #         plt.title('XGBRegressor')
# #     else :
# #         plt.title(models[i])

# # plt.show()


for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)



'''

1. 컬럼 삭제 하기 전 결과값

====================================================================================================
DecisionTreeRegressor 의 결과값 :  0.889564523302243
model.feature_importances :  [0.00260109 0.00357619 0.03341075 0.01412614 0.06356826 0.02345053
 0.03018411 0.0096009  0.63235863 0.02994791 0.0651619  0.09201359]
====================================================================================================
====================================================================================================
RandomForestRegressor 의 결과값 :  0.9512345221132069
model.feature_importances :  [0.00819332 0.00260956 0.04712703 0.01495359 0.05467586 0.04490658
 0.02963546 0.00880532 0.60715173 0.03811725 0.05440935 0.08941496]
====================================================================================================
====================================================================================================
GradientBoostingRegressor 의 결과값 :  0.843754404889862
model.feature_importances :  [8.49586566e-03 2.85949842e-04 5.32919589e-02 1.05623123e-02
 6.00198076e-02 4.98193065e-02 1.51760334e-02 6.78983421e-04
 6.48402737e-01 1.81558834e-02 3.97179320e-02 9.53932305e-02]
====================================================================================================
====================================================================================================
XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, gamma=None,
             gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, n_estimators=100, n_jobs=None,
             num_parallel_tree=None, predictor=None, random_state=None,
             reg_alpha=None, reg_lambda=None, ... 의 결과값 :  0.9576539539588307
model.feature_importances :  [0.03473732 0.01004884 0.23260237 0.03104127 0.0439783  0.06545382
 0.0128703  0.003248   0.25210503 0.04092309 0.06604593 0.20694576]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeRegressor 의 스코어:  0.8824945266899022
DecisionTreeRegressor 의 드랍후 스코어:  0.9098313147271108
RandomForestRegressor 의 스코어:  0.9451060957824069
RandomForestRegressor 의 드랍후 스코어:  0.9534463023891729
GradientBoostingRegressor 의 스코어:  0.8607428167906747
GradientBoostingRegressor 의 드랍후 스코어:  0.8677772623094095
XGB 의 스코어:  0.9472559569315105
XGB 의 드랍후 스코어:  0.9580691040233864


'''