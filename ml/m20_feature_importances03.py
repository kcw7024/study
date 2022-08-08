#[실습] 02번을 가져와서 피쳐 하나 삭제하고 성능비교

import numpy as np
from sklearn.datasets import load_diabetes
from sympy import re
import pandas as pd


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target
#print(x.shape)
x = np.delete(x, 1, axis=1)
#print(x.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
) 

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#model = DecisionTreeRegressor()
model = RandomForestRegressor()
#model = GradientBoostingRegressor()
#model = XGBRegressor()


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



'''

model.score :  0.046234925611890465
r2_score :  0.046234925611890465
================================================================================
DecisionTreeRegressor() : [0.07413783 0.01307606 0.34497272 0.08895758 0.02681448 0.10332579
 0.05579867 0.01331733 0.15841119 0.12118836]
 
컬럼 삭제후
model.score :  -0.10614346011186715
r2_score :  -0.10614346011186715
================================================================================
DecisionTreeRegressor() : [0.08592981 0.35781917 0.10488094 0.03838562 0.0872467  0.04078339
 0.01509407 0.15449244 0.11536786]
  
  
model.score :  0.43909178187179143
r2_score :  0.43909178187179143
================================================================================
RandomForestRegressor() : [0.05664911 0.01281086 0.33767366 0.08644687 0.04574221 0.05556679
 0.06100216 0.03304795 0.22105261 0.09000776]

컬럼 삭제후
model.score :  0.42081707364664533
r2_score :  0.42081707364664533
================================================================================
RandomForestRegressor() : [0.05618173 0.32716098 0.07480373 0.04513332 0.06032644 0.06159184
 0.03062853 0.24905423 0.0951192 ]


model.score :  0.4160185128926366
r2_score :  0.4160185128926366
================================================================================
GradientBoostingRegressor() : [0.04619565 0.01545641 0.33593563 0.09542872 0.03115416 0.06632719
 0.03859958 0.01412582 0.27768238 0.07909448]

컬럼 삭제후
model.score :  0.421115602081356
r2_score :  0.421115602081356
================================================================================
GradientBoostingRegressor() : [0.04606698 0.3428322  0.08631215 0.03793367 0.06974818 0.03793152
 0.01425496 0.28170112 0.08321921] 
 
 
model.score :  0.26078151031491137
r2_score :  0.26078151031491137
================================================================================
[0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191
 0.06551369 0.17944618 0.13779876 0.08540721]

컬럼 삭제후
model.score :  0.22142550751418588
r2_score :  0.22142550751418588
================================================================================
XGBRegressor() : [0.0260045  0.29435775 0.05291139 0.05066145 0.06820749 0.07702424
 0.20247848 0.14291398 0.08544067]






 
'''


