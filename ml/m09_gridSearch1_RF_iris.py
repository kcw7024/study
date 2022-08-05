#하이퍼 파라미터 튜닝
#최적의 파라미터를 찾아낸다
#해당 파라피터를 뽑아내고 추후에는 그 파라미터를 사용
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score, GridSearchCV #모든 경우의수를 다 넣겠다! cross validation
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩


import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
# print(datasets.DESCR)
# print(datasets.feature_names)
# print(x)
# print(y)
# print(x.shape,y.shape) # (150, 4) (150,)
# print("y의 라벨값 : ", np.unique(y))  # y의 라벨값 :  [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1234
    )

#print(y_test)
#print(y_train)

n_splits = 5
kfold = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth': [40,30,20,50], 'min_samples_leaf':[15, 30, 50, 100]}, #epochs
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split':[2, 4, 5, 20], 'n_jobs' : [-1, 3, 5]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'n_estimators':[150, 300, 200], 'max_depth':[7, 8, 9, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}    
]

#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알아둘것 논리적인회귀(이지만 분류모델!!!)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #결정트리방식의 분류모델
from sklearn.ensemble import RandomForestClassifier #DecisionTree가 앙상블로 되어있는 분류모델 

#model = SVC(C=1, kernel='linear', degree=3)
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1) # 42 * 5(kfold) = 210
#n_jobs = CPU갯수 정의 (-1:제일마지막숫자라서 전부다 쓴다는 뜻.)
#refit = True면 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습

#3. 컴파일 훈련
import time
start = time.time()
model.fit(x_train, y_train)
#Fitting 5 folds for each of 135 candidates, totalling 675 fits
end = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
print("best_score_ : ", model.best_score_) # train에 대한 점수
# best_score_ :  0.9916666666666668
print("model.score : ", model.score(x_test, y_test)) #test score라서 위와 값이 다름
# model.score :  0.9666666666666667

#4. 평가, 예측
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
print("accuracy_score :" , accuracy_score(y_test, y_predict))
# accuracy_score : 0.9666666666666667 model.score 와 동일하다.

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
# 최적 튠 ACC :  0.9666666666666667
print("걸린시간 : ", round(end-start, 2))


# 최적의 매개변수 :  RandomForestClassifier(max_depth=50, min_samples_leaf=15, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 50, 'min_samples_leaf': 15, 'n_estimators': 200}
# best_score_ :  0.95
# model.score :  1.0
# accuracy_score : 1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  22.47