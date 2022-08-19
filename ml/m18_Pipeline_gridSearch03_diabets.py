#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=1234
                                                    )


n_splits = 5
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=99)

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth': [40,30,20,50], 'RF__min_samples_leaf':[15, 30, 50, 100]}, #epochs
    {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split':[2, 4, 5, 20], 'RF__n_jobs' : [-1, 3, 5]},
    {'RF__min_samples_leaf' : [3, 5, 7, 10], 'RF__n_estimators':[150, 300, 200], 'RF__max_depth':[7, 8, 9, 10]},
    {'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_jobs' : [-1, 2, 4]}    
]

'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.pipeline import make_pipeline, Pipeline

pipe = Pipeline([
    ('minmax', MinMaxScaler()), 
    ('RF', RandomForestRegressor()),
    ], #verbose=1
    ) 
 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV 

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


#3. 컴파일 훈련
model.fit(x_train, y_train)
#pipeline fit이 포함되어있다.

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

# model.score :  0.5662669136159426



