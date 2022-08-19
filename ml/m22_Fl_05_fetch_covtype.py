from unittest import result
import numpy as np
from sklearn.datasets import fetch_covtype
from sympy import re
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

#x = np.delete(x, 1, axis=1)

allfeature = round(x.shape[1]*0.2, 1)
print('자를 갯수: ', int(allfeature))


le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
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


models = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]


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
#     plt.subplot(2, 2, i+1)
#     plot_feature_importances(models[i])
#     if str(models[i]).startswith("XGB") : 
#         plt.title('XGBRegressor')
#     else :
#         plt.title(models[i])

# plt.show()



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
DecisionTreeClassifier 의 결과값 :  0.8888888888888888
model.feature_importances :  [0.         0.         0.         0.         0.         0.
 0.42646862 0.         0.         0.40024104 0.         0.02087679
 0.15241354]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  0.9444444444444444
model.feature_importances :  [0.11718357 0.04083037 0.00940424 0.0227606  0.02254824 0.04534281
 0.13135616 0.0145129  0.01798192 0.20018081 0.08104221 0.14925054
 0.14760565]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  0.8611111111111112
model.feature_importances :  [2.31802270e-03 3.92115370e-02 7.82856707e-03 7.97727603e-03
 1.78573062e-03 1.75712570e-03 7.53830478e-02 5.63740273e-05
 9.49520733e-08 3.08797813e-01 3.85311560e-03 2.60824806e-01
 2.90206490e-01]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  0.8888888888888888
model.feature_importances :  [0.00715684 0.05944314 0.01588025 0.         0.03330867 0.00224883
 0.07457907 0.00876752 0.03545462 0.15266728 0.01350239 0.41068745
 0.18630388]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeClassifier 의 스코어:  0.9403371685756822
DecisionTreeClassifier 의 드랍후 스코어:  0.9403715910948943
RandomForestClassifier 의 스코어:  0.9563436400092941
RandomForestClassifier 의 드랍후 스코어:  0.9572214142492018


'''