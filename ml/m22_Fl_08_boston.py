from unittest import result
import numpy as np
from sklearn.datasets import load_boston
from sympy import re


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#x = np.delete(x, 1, axis=1)


allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=1234 
) 

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1] #features
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Impotances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    

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
DecisionTreeRegressor 의 결과값 :  0.8997753695359156
model.feature_importances :  [0.05328248 0.00123682 0.00623127 0.00178833 0.01512651 0.2275169
 0.014239   0.08846795 0.00235108 0.00719659 0.00968205 0.01065295
 0.56222806]
====================================================================================================
====================================================================================================
RandomForestRegressor 의 결과값 :  0.9114167082587971
model.feature_importances :  [0.03402263 0.00087555 0.0068707  0.00082125 0.0183596  0.35000912
 0.0154975  0.06714549 0.00530912 0.0137047  0.01780107 0.01536122
 0.45422208]
====================================================================================================
====================================================================================================
GradientBoostingRegressor 의 결과값 :  0.9170793832727535
model.feature_importances :  [2.01792308e-02 8.29835039e-04 4.90822306e-03 4.60025010e-04
 2.80451497e-02 2.82104748e-01 1.08945327e-02 9.14453600e-02
 4.45858928e-03 1.07701343e-02 2.76503217e-02 1.39250134e-02
 5.04328837e-01]
====================================================================================================
====================================================================================================
XGBRegressor 의 결과값 :  0.9111783299858156
model.feature_importances :  [0.01364043 0.00209235 0.0154772  0.00537503 0.03178463 0.16471088
 0.01116188 0.04255228 0.00872496 0.04093317 0.04876338 0.01225793
 0.6025259 ]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeRegressor 의 스코어:  0.8433006510659851
DecisionTreeRegressor 의 드랍후 스코어:  0.7987780685548026
RandomForestRegressor 의 스코어:  0.894688128837771
RandomForestRegressor 의 드랍후 스코어:  0.9174364914218947
GradientBoostingRegressor 의 스코어:  0.9026085742461757
GradientBoostingRegressor 의 드랍후 스코어:  0.9190296729533707
XGB 의 스코어:  0.9009456585682043
XGB 의 드랍후 스코어:  0.9053041455210861

'''