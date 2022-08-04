import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩

from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델
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
AdaBoostClassifier 의 정답률 :  0.7
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.36666666666666664
CalibratedClassifierCV 의 정답률 :  0.9333333333333333
CategoricalNB 의 정답률 :  0.36666666666666664
ClassifierChain 은 실행되지 않는다.
ComplementNB 의 정답률 :  0.6333333333333333
DecisionTreeClassifier 의 정답률 :  0.9
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9333333333333333
ExtraTreesClassifier 의 정답률 :  0.9666666666666667
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9
HistGradientBoostingClassifier 의 정답률 :  0.9333333333333333
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  1.0
LabelSpreading 의 정답률 :  1.0
LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667
LinearSVC 의 정답률 :  0.9333333333333333
LogisticRegression 의 정답률 :  0.9666666666666667
LogisticRegressionCV 의 정답률 :  0.9666666666666667
MLPClassifier 의 정답률 :  0.9
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 의 정답률 :  0.6
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.9
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 :  0.6333333333333333
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.9
RidgeClassifierCV 의 정답률 :  0.8666666666666667
SGDClassifier 의 정답률 :  0.9333333333333333
SVC 의 정답률 :  0.9666666666666667
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.
'''