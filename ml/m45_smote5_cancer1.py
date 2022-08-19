# smote 넣어서 증폭
# 넣었을때랑 안넣었을때 비교

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import pandas as pd
import numpy as np


datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(np.unique(y, return_counts=True))
#(array([0, 1]), array([212, 357], dtype=int64))
#print(pd.Series(y).value_counts())
print(x.shape, y.shape) # (569, 30) (569,)



# x = x[:-25]
# y = y[:-25]
# print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=123,
    stratify=y
)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2     6

#2. 모델

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

#3. 훈련 

model.fit(x_train, y_train)

#4. 평가,예측

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
# print("model.score : ", score) 
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score : ', f1_score(y_test, y_predict)) 
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro')) 

#4. SMOTE 적용
from imblearn.over_sampling import SMOTE

print("="*30, "SMOTE 적용 후", "="*30)
smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train) 
# 증폭, 평가데이터(test)는 해줄필요없음
# Smote :: 큰값에 맞춰서 증폭함
# 증폭시 큰데이터에 맞춰 되기때문에 데이터가 많아(커)질수록 느려진다
# 너무 큰 데이터는 단위를 나눠준 뒤에 증폭한다.


print(pd.Series(y_train).value_counts())
# 0    53
# 1    53
# 2    53

#2. 모델, #3. 훈련

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print('acc : ', accuracy_score(y_test, y_predict))
print('f1_score : ', f1_score(y_test, y_predict)) 


# acc :  0.965034965034965
# f1_score :  0.9720670391061451
# ============================== SMOTE 적용 후 ==============================
# acc :  0.972027972027972
# f1_score :  0.9775280898876404