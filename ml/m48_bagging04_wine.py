from json import load
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y
)

# bagging 사용시 scale 필수
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

model = BaggingClassifier(XGBClassifier(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=123
                          )
#bagging 정리할것. 


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print(model.score(x_test, y_test)) 



'''
#DecisionTreeClassifier
0.9444444444444444

#XGBClassifier
0.9444444444444444

#RandomForestClassifier
0.9722222222222222

'''

