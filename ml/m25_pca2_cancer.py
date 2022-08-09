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
    x = datasets.data
    pca = PCA(n_components=i)
    x = pca.fit_transform(x)
    print(x.shape)
    #print()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=666, shuffle=True
    )
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print('결과 : ', result)
    
#x = pca.fit_transform(x)
print(x.shape) # (506, 2)


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

# PCA 안썼을때
# 결과 :  0.9210526315789473

# PCA 2개일때
# 결과 :  0.1548699266090462

# PCA 11개일때
# 결과 :  0.7432679633113446




















