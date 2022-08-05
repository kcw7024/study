#랜덤으로 모델을 뽑아서 실행 : 시간단축을 위해 시작
#하이퍼 파라미터 튜닝
#최적의 파라미터를 찾아낸다
#해당 파라피터를 뽑아내고 추후에는 그 파라미터를 사용
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV #모든 경우의수를 다 넣겠다! cross validation
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
    {"C" : [1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},                              # 12
    {"C" : [1, 10, 100], "kernel":["rbf"], "gamma":[0.001,0.0001]},                                 # 6
    {"C" : [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.01,0.001,0.0001], "degree":[3,4]}   # 24
]                                                                                                   #총 42개중 10개만 빼서 작업한다.

#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression #꼭 알아둘것 논리적인회귀(이지만 분류모델!!!)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #결정트리방식의 분류모델
from sklearn.ensemble import RandomForestClassifier #DecisionTree가 앙상블로 되어있는 분류모델 

#model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42 * 5(kfold) = 210
#n_jobs = CPU갯수 정의 (-1:제일마지막숫자라서 전부다 쓴다는 뜻.)
#refit = True면 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42 * 5(kfold) = 210

#3. 컴파일 훈련
import time
start = time.time()
model.fit(x_train, y_train)
#Fitting 5 folds for each of 10 candidates, totalling 50 fits
end = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_ : ", model.best_score_) # train에 대한 점수
print("model.score : ", model.score(x_test, y_test)) #test score라서 위와 값이 다름

#4. 평가, 예측
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score :" , accuracy_score(y_test, y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
print("걸린시간 : ", round(end-start, 4))


# best_estimator 와 predict의 값은 같다.

# 최적의 매개변수 :  SVC(C=1, degree=5, kernel='linear')
# 최적의 파라미터 :  {'kernel': 'linear', 'degree': 5, 'C': 1}
# best_score_ :  0.975
# model.score :  1.0
# accuracy_score : 1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  2.2475

# print(pd.DataFrame())

