from matplotlib.pyplot import sca
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234
)

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth': [40,30,20,50], 'RF__min_samples_leaf':[15, 30, 50, 100]}, #epochs
    #{'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split':[2, 4, 5, 20], 'RF__n_jobs' : [-1, 3, 5]},
    {'RF__n_jobs' : [-1, 2, 4]}    
]

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

#model = SVC()
#model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 
#pipeline에서는 fit_transform이 적용된다.

pipe = Pipeline([
    ('minmax', MinMaxScaler()), 
    ('RF', RandomForestClassifier()),
    ], #verbose=1
    )

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

#model = GridSearchCV(pipe, parameters, cv=5, verbose=1) #error!!!
model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


#3. 훈련
model.fit(x_train, y_train)
#pipeline fit이 포함되어있다.

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)

# model.score :  1.0









