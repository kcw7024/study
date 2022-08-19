
from unittest import result
import numpy as np
from sklearn.datasets import load_digits
from sympy import re


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

#x = np.delete(x, 1, axis=1)

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
DecisionTreeClassifier 의 결과값 :  0.8861111111111111
model.feature_importances :  [0.         0.0014979  0.00398825 0.00833354 0.00524508 0.05021593
 0.00152993 0.         0.         0.00259601 0.01491541 0.00220385
 0.00955727 0.01650103 0.00291629 0.         0.         0.00419293
 0.01313478 0.02370339 0.04750694 0.08608069 0.00228235 0.
 0.         0.00813125 0.02489139 0.05896133 0.04869065 0.04337368
 0.00221674 0.         0.         0.05486042 0.02375067 0.00889983
 0.07906535 0.02606167 0.         0.         0.         0.00630306
 0.07015961 0.05251199 0.0045108  0.00816672 0.01119603 0.
 0.         0.00115992 0.01592766 0.00578119 0.00425963 0.00875234
 0.02390878 0.         0.         0.00217819 0.01467767 0.00077328
 0.05806899 0.03361012 0.00274944 0.        ]
====================================================================================================
====================================================================================================
RandomForestClassifier 의 결과값 :  0.9722222222222222
model.feature_importances :  [0.00000000e+00 1.90074297e-03 1.64482660e-02 1.08185521e-02
 8.52680691e-03 2.02335957e-02 9.52421252e-03 6.72317924e-04
 1.48469457e-05 1.13813688e-02 2.37426537e-02 7.10255468e-03
 1.40850103e-02 3.07028839e-02 7.46282810e-03 7.73662408e-04
 2.32191814e-05 6.59601988e-03 1.90495176e-02 2.58154199e-02
 2.97894895e-02 4.34037293e-02 8.72358113e-03 1.96326748e-04
 1.85796542e-05 1.40625823e-02 3.99204552e-02 2.84062310e-02
 3.33666912e-02 2.62761533e-02 2.62033541e-02 6.69342650e-05
 0.00000000e+00 2.77491813e-02 2.45565398e-02 2.08835503e-02
 4.59123035e-02 2.03873722e-02 2.73827820e-02 0.00000000e+00
 0.00000000e+00 1.09727259e-02 4.11263099e-02 4.09349711e-02
 2.22609563e-02 1.71366739e-02 1.90515598e-02 9.04716241e-05
 3.76633963e-05 1.82096556e-03 1.90213472e-02 2.57127037e-02
 1.49635034e-02 2.42638890e-02 2.10449844e-02 2.83268966e-03
 5.63233076e-05 2.26487041e-03 2.02398116e-02 1.16863027e-02
 2.51079436e-02 3.20366841e-02 1.18063738e-02 3.34995912e-03]
====================================================================================================
====================================================================================================
GradientBoostingClassifier 의 결과값 :  0.975
model.feature_importances :  [0.00000000e+00 8.58052056e-04 1.56854060e-02 2.88968585e-03
 2.25642480e-03 5.34220733e-02 6.13776407e-03 8.36373868e-04
 0.00000000e+00 1.49004475e-03 1.52302712e-02 1.26735511e-03
 1.58009596e-02 1.35696913e-02 3.46570017e-03 7.21023221e-04
 2.27711306e-04 2.31310039e-03 1.05515952e-02 4.62029690e-02
 1.93604084e-02 8.79284346e-02 3.77649138e-03 2.94609070e-04
 2.78462260e-04 2.18003463e-03 4.34145610e-02 2.30068778e-02
 3.36844169e-02 2.72103803e-02 1.12775048e-02 6.96142911e-04
 0.00000000e+00 5.89962907e-02 4.32282962e-03 8.70311909e-03
 7.51850594e-02 1.73062302e-02 1.99223932e-02 0.00000000e+00
 1.43997055e-09 5.34823560e-03 8.02423407e-02 6.37422202e-02
 1.54041380e-02 1.39367222e-02 2.06094592e-02 2.18464315e-04
 2.68969257e-04 8.02830983e-04 6.06691026e-03 1.81103769e-02
 1.18867581e-02 1.05811106e-02 2.60803237e-02 1.81482478e-04
 2.07891708e-04 3.42331897e-04 9.20013970e-03 2.25966259e-03
 5.29431860e-02 4.96965648e-03 1.74556709e-02 8.67066951e-03]
====================================================================================================
====================================================================================================
XGBClassifier 의 결과값 :  0.975
model.feature_importances :  [0.         0.03648211 0.01304824 0.00828797 0.00430605 0.04471496
 0.00596952 0.00876894 0.         0.00536884 0.01515524 0.00465277
 0.00982321 0.00767834 0.00220352 0.01961521 0.         0.00496003
 0.00473713 0.04045011 0.0098936  0.04634573 0.00296796 0.02277583
 0.         0.00233769 0.03131091 0.00979861 0.03258161 0.02877315
 0.01884988 0.         0.         0.06312703 0.00717313 0.00944097
 0.05457681 0.01184776 0.02645437 0.         0.         0.01539725
 0.03461067 0.04481603 0.01343341 0.02158235 0.02245468 0.
 0.         0.00451847 0.00551131 0.01088032 0.01101712 0.0188521
 0.02234185 0.         0.         0.         0.01037737 0.00415869
 0.07070705 0.01810189 0.04138831 0.0153738 ]
====================================================================================================

2. 컬럼 삭제 후 결과값

DecisionTreeClassifier 의 스코어:  0.8583333333333333       
DecisionTreeClassifier 의 드랍후 스코어:  0.8527777777777777
RandomForestClassifier 의 스코어:  0.9805555555555555
RandomForestClassifier 의 드랍후 스코어:  0.9777777777777777
GradientBoostingClassifier 의 스코어:  0.975
GradientBoostingClassifier 의 드랍후 스코어:  0.9722222222222222
XGB 의 스코어:  0.975
XGB 의 드랍후 스코어:  0.975

'''