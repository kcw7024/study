import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
#pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
# print(sk.__version__)


#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,)
print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts())
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25]
y = y[:-25]
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
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro')) 
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro')) 


'''
# 기본결과
acc :  0.9777777777777777
f1_score(macro) :  0.9797235023041475

# 데이터 축소 후(2라벨을 30개 줄인 후)
acc :  0.972972972972973
f1_score(macro) :  0.9797235023041475

# 데이터 축소 후(2라벨을 40개 줄인 후)
acc :  0.9428571428571428
f1_score(macro) :  0.8596176821983273

'''
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
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro')) 

# acc :  0.9487179487179487
# f1_score(macro) :  0.9404257630064081
# ============================== SMOTE 적용 후 ==============================
# acc :  0.9743589743589743
# f1_score(macro) :  0.9797235023041475

