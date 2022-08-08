from unittest import result
import numpy as np
from sklearn.datasets import load_breast_cancer
from sympy import re


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)


allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


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
#     # plt.subplot(2, 2, i+1)
#     # plot_feature_importances(models[i])
#     # if str(models[i]).startswith("XGB") : 
#     #     plt.title('XGBRegressor')
#     # else :
#     #     plt.title(models[i])

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
DecisionTreeClassifier 의 결과값 :  0.8947368421052632
model.feature_importances :  [0.         0.         0.         0.         0.01986652 0.
 0.         0.         0.         0.         0.         0.
 0.         0.01208949 0.         0.         0.         0.01734378
 0.00902843 0.         0.         0.009423   0.01467993 0.15328548
 0.02119796 0.         0.00939315 0.73369226 0.         0.        ]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  0.9298245614035088
model.feature_importances :  [0.05481409 0.01344726 0.06242425 0.04564885 0.00762746 0.00930008
 0.07758109 0.05299662 0.002839   0.00305626 0.01701808 0.00323635
 0.0093791  0.03745179 0.00321438 0.00468427 0.00479326 0.00548031
 0.00301191 0.00337073 0.08529012 0.01442662 0.16494846 0.10532026
 0.01844922 0.01645679 0.03320521 0.11958506 0.0116468  0.0092963 ]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  0.9122807017543859
model.feature_importances :  [5.52738857e-05 8.52038556e-03 6.12356506e-04 6.34699372e-03
 1.74874619e-03 2.83375905e-03 1.02035053e-03 3.52224307e-01
 1.08070418e-04 9.01816448e-07 4.93730806e-03 6.91555912e-05
 3.24419600e-05 2.26964195e-02 9.18311077e-05 6.14727530e-05
 3.92310827e-06 4.64447362e-06 1.41862015e-03 8.83252115e-05
 4.15570451e-02 3.34059145e-02 5.43450532e-02 5.22957814e-02
 1.07752508e-02 4.56176964e-06 7.43505483e-03 3.83846347e-01
 5.44471552e-03 8.01498942e-03]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  0.9385964912280702
model.feature_importances :  [0.01101296 0.01152203 0.00033031 0.02094301 0.00649445 0.
 0.01629187 0.30016553 0.         0.00072916 0.00043588 0.
 0.         0.01816854 0.00346737 0.00432523 0.00048088 0.
 0.00689773 0.         0.14143057 0.02041801 0.04719434 0.07517467
 0.01062466 0.00319993 0.01011148 0.2675406  0.00192347 0.0211173 ]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeClassifier 의 스코어:  0.8859649122807017
DecisionTreeClassifier 의 드랍후 스코어:  0.8771929824561403
RandomForestClassifier 의 스코어:  0.9210526315789473
RandomForestClassifier 의 드랍후 스코어:  0.9298245614035088
GradientBoostingClassifier 의 스코어:  0.9122807017543859
GradientBoostingClassifier 의 드랍후 스코어:  0.9210526315789473
XGB 의 스코어:  0.9385964912280702
XGB 의 드랍후 스코어:  0.9385964912280702

'''