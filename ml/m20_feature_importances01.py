from unittest import result
import numpy as np
from sklearn.datasets import load_iris
from sympy import re


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
) 

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
#model = GradientBoostingClassifier()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print("="*80)
print(model,':',model.feature_importances_)


'''

model.score :  1.0
accuracy_score :  1.0

================================================================================
DecisionTreeClassifier() : [0.01669101 0.         0.58410048 0.39920851]
================================================================================
RandomForestClassifier() : [0.07894852 0.02322831 0.39175036 0.50607282]
================================================================================
GradientBoostingClassifier() : [0.00818142 0.01222001 0.26787491 0.71172367]
================================================================================
XGBClassifier : [0.00912187 0.0219429  0.678874   0.29006115]


'''


