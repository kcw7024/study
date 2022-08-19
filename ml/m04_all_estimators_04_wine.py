from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
#print(x.shape, y.shape) # (178, 13), (178,)
#print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(np.min(x_train))  # 0.0
#print(np.max(x_train))  # 1.0

#print(np.min(x_test))  # 1.0
#print(np.max(x_test))  # 1.0


#2. 모델
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41

for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')

'''
AdaBoostClassifier 의 정답률 :  0.5370370370370371
BaggingClassifier 의 정답률 :  0.9814814814814815
BernoulliNB 의 정답률 :  0.9259259259259259
CalibratedClassifierCV 의 정답률 :  0.9814814814814815
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
ComplementNB 은 실행되지 않는다.
DecisionTreeClassifier 의 정답률 :  0.9814814814814815
DummyClassifier 의 정답률 :  0.4074074074074074
ExtraTreeClassifier 의 정답률 :  0.9259259259259259
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  0.9814814814814815
GaussianProcessClassifier 의 정답률 :  0.9814814814814815
GradientBoostingClassifier 의 정답률 :  0.9629629629629629
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  0.9444444444444444
LabelPropagation 의 정답률 :  0.9629629629629629
LabelSpreading 의 정답률 :  0.9629629629629629
LinearDiscriminantAnalysis 의 정답률 :  0.9814814814814815
LinearSVC 의 정답률 :  0.9814814814814815
LogisticRegression 의 정답률 :  0.9814814814814815
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.9629629629629629
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 은 실행되지 않는다.
NearestCentroid 의 정답률 :  0.9814814814814815
NuSVC 의 정답률 :  0.9814814814814815
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.9814814814814815
Perceptron 의 정답률 :  0.9814814814814815
QuadraticDiscriminantAnalysis 의 정답률 :  0.9629629629629629
RadiusNeighborsClassifier 은 실행되지 않는다.
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9629629629629629
RidgeClassifierCV 의 정답률 :  0.9629629629629629
SGDClassifier 의 정답률 :  0.9444444444444444
SVC 의 정답률 :  0.9814814814814815
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.

'''