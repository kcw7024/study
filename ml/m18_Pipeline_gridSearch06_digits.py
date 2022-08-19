import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(x,y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100)


parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth': [40,30,20,50], 'RF__min_samples_leaf':[15, 30, 50, 100]}, #epochs
    {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split':[2, 4, 5, 20], 'RF__n_jobs' : [-1, 3, 5]},
    {'RF__min_samples_leaf' : [3, 5, 7, 10], 'RF__n_estimators':[150, 300, 200], 'RF__max_depth':[7, 8, 9, 10]},
    {'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_jobs' : [-1, 2, 4]}    
]

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

# print(y_test)
# print(y_train)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알아둘것 논리적인회귀(이지만 분류모델!!!)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #결정트리방식의 분류모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
 
pipe = Pipeline([
    ('minmax', MinMaxScaler()), 
    ('RF', RandomForestClassifier()),
    ], #verbose=1
    ) 
 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV 

model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)


#3. 훈련

model.fit(x_train, y_train)
#pipeline fit이 포함되어있다.

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

# model.score :  0.9703703703703703









