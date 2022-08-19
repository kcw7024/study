from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
        # model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('Model Name : ', name)
        print('ACC : ', scores) 
        print('cross_val_score : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        #print(y_predict)
        
        #y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')
    
    

'''
Model Name :  AdaBoostClassifier
ACC :  [0.61538462 0.92307692 0.69230769 0.53846154 1.         0.75
 0.66666667 0.58333333 0.75       0.5       ]
cross_val_score :  0.7019
Model Name :  BaggingClassifier
ACC :  [0.92307692 1.         0.84615385 0.84615385 0.91666667 0.91666667
 1.         1.         0.91666667 1.        ]
cross_val_score :  0.9365
Model Name :  BernoulliNB
ACC :  [1.         0.92307692 0.92307692 0.92307692 1.         0.83333333
 1.         0.91666667 0.83333333 0.83333333]
cross_val_score :  0.9186
Model Name :  CalibratedClassifierCV
ACC :  [1.         1.         1.         1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9917
Model Name :  CategoricalNB
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
Model Name :  ComplementNB
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
ComplementNB 은 실행되지 않는다.
Model Name :  DecisionTreeClassifier
ACC :  [0.84615385 1.         0.84615385 0.92307692 0.83333333 0.83333333
 0.91666667 0.83333333 0.83333333 0.83333333]
cross_val_score :  0.8699
Model Name :  DummyClassifier
ACC :  [0.38461538 0.53846154 0.38461538 0.30769231 0.08333333 0.5
 0.5        0.33333333 0.5        0.41666667]
cross_val_score :  0.3949
Model Name :  ExtraTreeClassifier
ACC :  [0.84615385 0.84615385 0.92307692 0.76923077 0.83333333 0.83333333
 0.83333333 1.         0.66666667 1.        ]
cross_val_score :  0.8551
Model Name :  ExtraTreesClassifier
ACC :  [0.92307692 1.         1.         0.92307692 1.         1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9846
Model Name :  GaussianNB
ACC :  [0.92307692 0.92307692 1.         1.         1.         1.
 1.         1.         0.91666667 1.        ]
cross_val_score :  0.9763
Model Name :  GaussianProcessClassifier
ACC :  [0.92307692 1.         1.         0.92307692 0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9763
Model Name :  GradientBoostingClassifier
ACC :  [0.92307692 1.         0.92307692 0.84615385 0.91666667 0.83333333
 1.         1.         0.75       1.        ]
cross_val_score :  0.9192
Model Name :  HistGradientBoostingClassifier
ACC :  [0.92307692 1.         0.92307692 0.92307692 0.91666667 0.91666667
 1.         1.         1.         1.        ]
cross_val_score :  0.9603
Model Name :  KNeighborsClassifier
ACC :  [0.92307692 1.         0.84615385 0.92307692 0.91666667 0.91666667
 1.         1.         1.         1.        ]
cross_val_score :  0.9526
Model Name :  LabelPropagation
ACC :  [0.92307692 0.92307692 1.         0.92307692 0.91666667 1.
 1.         1.         0.91666667 1.        ]
cross_val_score :  0.9603
Model Name :  LabelSpreading
ACC :  [0.92307692 0.92307692 1.         0.92307692 0.91666667 1.
 1.         1.         0.91666667 1.        ]
cross_val_score :  0.9603
Model Name :  LinearDiscriminantAnalysis
ACC :  [1.         1.         1.         1.         1.         1.
 1.         1.         1.         0.91666667]
cross_val_score :  0.9917
Model Name :  LinearSVC
ACC :  [1.         0.92307692 1.         1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.984
Model Name :  LogisticRegression
ACC :  [0.92307692 1.         1.         1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.984
Model Name :  LogisticRegressionCV
ACC :  [0.92307692 1.         1.         1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.984
Model Name :  MLPClassifier
ACC :  [0.92307692 1.         1.         1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.984
MultiOutputClassifier 은 실행되지 않는다.
Model Name :  MultinomialNB
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultinomialNB 은 실행되지 않는다.
Model Name :  NearestCentroid
ACC :  [0.92307692 1.         1.         0.92307692 0.91666667 0.91666667
 1.         1.         0.91666667 0.91666667]
cross_val_score :  0.9513
Model Name :  NuSVC
ACC :  [0.92307692 1.         1.         1.         0.91666667 1.
 1.         1.         1.         0.91666667]
cross_val_score :  0.9756
NuSVC 은 실행되지 않는다.
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
Model Name :  PassiveAggressiveClassifier
ACC :  [0.92307692 1.         0.92307692 1.         0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9763
Model Name :  Perceptron
ACC :  [1.         0.92307692 1.         1.         1.         1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9923
Model Name :  QuadraticDiscriminantAnalysis
ACC :  [1.         1.         1.         1.         1.         1.
 1.         1.         0.91666667 1.        ]
cross_val_score :  0.9917
Model Name :  RadiusNeighborsClassifier
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
RadiusNeighborsClassifier 은 실행되지 않는다.
Model Name :  RandomForestClassifier
ACC :  [0.92307692 1.         0.92307692 0.92307692 0.91666667 1.
 1.         1.         1.         1.        ]
cross_val_score :  0.9686
Model Name :  RidgeClassifier
ACC :  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
cross_val_score :  1.0
Model Name :  RidgeClassifierCV
ACC :  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
cross_val_score :  1.0
Model Name :  SGDClassifier
ACC :  [1.         1.         1.         1.         0.91666667 1.
 1.         1.         1.         0.91666667]
cross_val_score :  0.9833
Model Name :  SVC
ACC :  [0.92307692 1.         1.         1.         0.91666667 1.
 1.         1.         1.         0.91666667]
cross_val_score :  0.9756
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.

'''