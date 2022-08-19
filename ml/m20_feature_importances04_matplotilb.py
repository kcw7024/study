from unittest import result
import numpy as np
from sklearn.datasets import load_diabetes
from sympy import re


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
) 

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
#model = GradientBoostingRegressor()
model = XGBRegressor()

# for model in use_models :
#     model = model()
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     print(name,'의 ACC : ', result)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)

print("="*80)
print(model,':',model.feature_importances_)


import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1] #features
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Impotances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)

plot_feature_importances(model)
plt.show()


'''

model.score :  0.046234925611890465
r2_score :  0.046234925611890465
================================================================================
DecisionTreeRegressor() : [0.07413783 0.01307606 0.34497272 0.08895758 0.02681448 0.10332579
 0.05579867 0.01331733 0.15841119 0.12118836]
 
model.score :  0.43909178187179143
r2_score :  0.43909178187179143
================================================================================
RandomForestRegressor() : [0.05664911 0.01281086 0.33767366 0.08644687 0.04574221 0.05556679
 0.06100216 0.03304795 0.22105261 0.09000776]

model.score :  0.4160185128926366
r2_score :  0.4160185128926366
================================================================================
GradientBoostingRegressor() : [0.04619565 0.01545641 0.33593563 0.09542872 0.03115416 0.06632719
 0.03859958 0.01412582 0.27768238 0.07909448]
 
model.score :  0.26078151031491137
r2_score :  0.26078151031491137
================================================================================
[0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191
 0.06551369 0.17944618 0.13779876 0.08540721]

 
'''


