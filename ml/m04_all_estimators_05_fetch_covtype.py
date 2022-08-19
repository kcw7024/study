from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
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
from sklearn.datasets import fetch_covtype
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]


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
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


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
        
#예외처리 더 알아놓을것.


'''

AdaBoostClassifier 의 정답률 :  0.534216082247108591
BaggingClassifier 의 정답률 :  0.95938704791628428 
BernoulliNB 의 정답률 :  0.659353772718927818
CalibratedClassifierCV 의 정답률 :  0.7121810170736185
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.       
ComplementNB 은 실행되지 않는다.
DecisionTreeClassifier 의 정답률 :  0.934808035615935377
DummyClassifier 의 정답률 :  0.4864088030129083167
ExtraTreeClassifier 의 정답률 :  0.853434220298512942
ComplementNB 은 실행되지 않는다.
DecisionTreeClassifier 의 정답률 :  0.9348035615935377
DummyClassifier 의 정답률 :  0.48640880301083167
ExtraTreeClassifier 의 정답률 :  0.8534342298512942
ExtraTreesClassifier 의 정답률 :  0.9513092069028823
GaussianNB 의 정답률 :  0.09025036717459152
GaussianProcessClassifier 은 실행되지 않는다.
GradientBoostingClassifier 의 정답률 :  0.774262208555168
HistGradientBoostingClassifier 의 정답률 :  0.7800796309895355
KNeighborsClassifier 의 정답률 :  0.924895584725537
LabelPropagation 은 실행되지 않는다.
LabelSpreading 은 실행되지 않는다.
LinearDiscriminantAnalysis 의 정답률 :  0.6787451808334863
LinearSVC 의 정답률 :  0.7123990269873325
LogisticRegression 의 정답률 :  0.723821599045346
LogisticRegressionCV 의 정답률 :  0.7241715623278869
MLPClassifier 의 정답률 :  0.8743172847438957
MultiOutputClassifier 은 실행되지 않는다.
MultinomialNB 은 실행되지 않는다.
NearestCentroid 의 정답률 :  0.44703506517349
NuSVC 은 실행되지 않는다.
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
PassiveAggressiveClassifier 의 정답률 :  0.6047824490545254
Perceptron 의 정답률 :  0.6318558380760051
QuadraticDiscriminantAnalysis 의 정답률 :  0.08735313016339269
RadiusNeighborsClassifier 은 실행되지 않는다.
RandomForestClassifier 의 정답률 :  0.9525139985313016
RidgeClassifier 의 정답률 :  0.7003166880851845
RidgeClassifierCV 의 정답률 :  0.7002994767762071
SGDClassifier 의 정답률 :  0.7143611162107583


'''