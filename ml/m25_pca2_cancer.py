import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__) #0.24.2
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(569, 30) (569,)


#pca = PCA(n_components=30)
#PCA = 차원축소(열,컬럼,피쳐) / 압축

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


model = RandomForestClassifier()

for i in range(1, 31) :
#    x = datasets.data
    pca = PCA(n_components=i)
    x2 = pca.fit_transform(x)
    print(i, "번 압축했을때 Shape : " , x2.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=666, shuffle=True
    )
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print('결과 : ', result)
    print("="*40)
        
#x = pca.fit_transform(x)
#print(x.shape) # (506, 2)


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=666, shuffle=True
# )


       
    

#2. 모델
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from xgboost import XGBRegressor, XGBClassifier


# model = RandomForestClassifier()


#3. 훈련

#model.fit(x_train, y_train)#, eval_metric='error')

#4. 평가, 예측

#result = model.score(x_test, y_test)
#print('결과 : ', result)

'''
1 번 압축했을때 Shape :  (569, 1)
결과 :  0.9210526315789473
========================================
2 번 압축했을때 Shape :  (569, 2)
결과 :  0.9298245614035088
========================================
3 번 압축했을때 Shape :  (569, 3)
결과 :  0.9298245614035088
========================================
4 번 압축했을때 Shape :  (569, 4)
결과 :  0.9122807017543859
========================================
5 번 압축했을때 Shape :  (569, 5)
결과 :  0.9122807017543859
========================================
6 번 압축했을때 Shape :  (569, 6)
결과 :  0.9298245614035088
========================================
7 번 압축했을때 Shape :  (569, 7)
결과 :  0.9122807017543859
========================================
8 번 압축했을때 Shape :  (569, 8)
결과 :  0.9298245614035088
========================================
9 번 압축했을때 Shape :  (569, 9)
결과 :  0.9122807017543859
========================================
10 번 압축했을때 Shape :  (569, 10)
결과 :  0.9210526315789473
========================================
11 번 압축했을때 Shape :  (569, 11)
결과 :  0.9210526315789473
========================================
12 번 압축했을때 Shape :  (569, 12)
결과 :  0.9298245614035088
========================================
13 번 압축했을때 Shape :  (569, 13)
결과 :  0.9210526315789473
========================================
14 번 압축했을때 Shape :  (569, 14)
결과 :  0.9298245614035088
========================================
15 번 압축했을때 Shape :  (569, 15)
결과 :  0.9122807017543859
========================================
16 번 압축했을때 Shape :  (569, 16)
결과 :  0.9122807017543859
========================================
17 번 압축했을때 Shape :  (569, 17)
결과 :  0.9210526315789473
========================================
18 번 압축했을때 Shape :  (569, 18)
결과 :  0.9122807017543859
========================================
19 번 압축했을때 Shape :  (569, 19)
결과 :  0.9210526315789473
========================================
20 번 압축했을때 Shape :  (569, 20)
결과 :  0.9210526315789473
========================================
21 번 압축했을때 Shape :  (569, 21)
결과 :  0.9210526315789473
========================================
22 번 압축했을때 Shape :  (569, 22)
결과 :  0.9210526315789473
========================================
23 번 압축했을때 Shape :  (569, 23)
결과 :  0.9210526315789473
========================================
24 번 압축했을때 Shape :  (569, 24)
결과 :  0.9210526315789473
========================================
25 번 압축했을때 Shape :  (569, 25)
결과 :  0.9122807017543859
========================================
26 번 압축했을때 Shape :  (569, 26)
결과 :  0.9122807017543859
========================================
27 번 압축했을때 Shape :  (569, 27)
결과 :  0.9210526315789473
========================================
28 번 압축했을때 Shape :  (569, 28)
결과 :  0.9298245614035088
========================================
29 번 압축했을때 Shape :  (569, 29)
결과 :  0.9210526315789473
========================================
30 번 압축했을때 Shape :  (569, 30)
결과 :  0.9210526315789473
========================================

'''


















