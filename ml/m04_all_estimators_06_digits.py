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
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


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

# print(y_test)
# print(y_train)


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

AdaBoostClassifier 의 정답률 :  0.2388888888888889
BaggingClassifier 의 정답률 :  0.9222222222222223
BernoulliNB 의 정답률 :  0.8444444444444444
CalibratedClassifierCV 의 정답률 :  0.9629629629629629
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
ComplementNB 의 정답률 :  0.7759259259259259
DecisionTreeClassifier 의 정답률 :  0.8425925925925926
DummyClassifier 의 정답률 :  0.08703703703703704
ExtraTreeClassifier 의 정답률 :  0.7648148148148148
ExtraTreesClassifier 의 정답률 :  0.987037037037037
GaussianNB 의 정답률 :  0.85
GaussianProcessClassifier 의 정답률 :  0.1259259259259259
GradientBoostingClassifier 의 정답률 :  0.9611111111111111
HistGradientBoostingClassifier 의 정답률 :  0.9685185185185186
KNeighborsClassifier 의 정답률 :  0.9944444444444445
LabelPropagation 의 정답률 :  0.10740740740740741
LabelSpreading 의 정답률 :  0.10740740740740741
LinearDiscriminantAnalysis 의 정답률 :  0.9592592592592593
LinearSVC 의 정답률 :  0.95
LogisticRegression 의 정답률 :  0.9703703703703703
LogisticRegressionCV 의 정답률 :  0.975925925925926
MLPClassifier 의 정답률 :  0.9777777777777777
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 의 정답률 :  0.8888888888888888
NearestCentroid 의 정답률 :  0.9055555555555556
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.9407407407407408
Perceptron 의 정답률 :  0.9351851851851852
QuadraticDiscriminantAnalysis 의 정답률 :  0.7518518518518519
RadiusNeighborsClassifier 은 실행되지 않는다.
RandomForestClassifier 의 정답률 :  0.9833333333333333
RidgeClassifier 의 정답률 :  0.9388888888888889
RidgeClassifierCV 의 정답률 :  0.9388888888888889
SGDClassifier 의 정답률 :  0.9592592592592593
SVC 의 정답률 :  0.9907407407407407
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.

'''