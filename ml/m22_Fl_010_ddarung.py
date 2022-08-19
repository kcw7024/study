import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

allfeature = round(train_set.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

x = np.array(x)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
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
DecisionTreeRegressor 의 결과값 :  0.48787891222020774
model.feature_importances :  [0.62647864 0.17633604 0.00664555 0.02493977 0.05058246 0.02291534
 0.0309178  0.04083303 0.02035138]
====================================================================================================
====================================================================================================
RandomForestRegressor 의 결과값 :  0.770848841693597
model.feature_importances :  [0.60318272 0.17712822 0.00858845 0.03196331 0.04040279 0.0346551
 0.04452677 0.03704031 0.02251233]
====================================================================================================
====================================================================================================
GradientBoostingRegressor 의 결과값 :  0.7913591324958649
model.feature_importances :  [0.67079583 0.20806086 0.0116389  0.00974493 0.0163583  0.02574203
 0.02718792 0.02368428 0.00678695]
====================================================================================================
====================================================================================================
XGBRegressor 의 결과값 :  0.7847649057911799
model.feature_importances :  [0.44714096 0.11791874 0.22405049 0.0219841  0.04035296 0.02891764
 0.04964453 0.03955825 0.03043229]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeRegressor 의 스코어:  0.6207010003645456
DecisionTreeRegressor 의 드랍후 스코어:  0.5179041647622789
RandomForestRegressor 의 스코어:  0.7742154566566166
RandomForestRegressor 의 드랍후 스코어:  0.7898810864624304
GradientBoostingRegressor 의 스코어:  0.7669711171340123
GradientBoostingRegressor 의 드랍후 스코어:  0.7565011822092248
XGB 의 스코어:  0.7762904047633264
XGB 의 드랍후 스코어:  0.7770894918955813

'''