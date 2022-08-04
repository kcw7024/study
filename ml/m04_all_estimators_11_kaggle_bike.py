# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
path = 'C:/study/_data/bike_sharing/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='regressor')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41

for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)
    except:
        # continue
        print(name, '은 실행되지 않는다.')

'''

ARDRegression 의 정답률 :  0.39643832603995854
AdaBoostRegressor 의 정답률 :  0.5978681799441947
BaggingRegressor 의 정답률 :  0.9377887914710734
BayesianRidge 의 정답률 :  0.3966543912131405
CCA 의 정답률 :  0.07132237555400922
DecisionTreeRegressor 의 정답률 :  0.8971311645372719
DummyRegressor 의 정답률 :  -0.00028666396450471865
ElasticNet 의 정답률 :  0.14331009564424035
ElasticNetCV 의 정답률 :  0.38266403224221146
ExtraTreeRegressor 의 정답률 :  0.8508671692585046
ExtraTreesRegressor 의 정답률 :  0.9533034166810336
GammaRegressor 의 정답률 :  0.08822807411550837
GaussianProcessRegressor 의 정답률 :  -77.10786652662779
GradientBoostingRegressor 의 정답률 :  0.8624240540590182
HistGradientBoostingRegressor 의 정답률 :  0.955346818580499
HuberRegressor 의 정답률 :  0.3699762716185667
IsotonicRegression 은 실행되지 않는다.
KNeighborsRegressor 의 정답률 :  0.6952933302808703
KernelRidge 의 정답률 :  0.3957443663042901
Lars 의 정답률 :  0.3965960655680766
LarsCV 의 정답률 :  0.39609375678354664
Lasso 의 정답률 :  0.3923060009779967
LassoCV 의 정답률 :  0.3962640935108953
LassoLars 의 정답률 :  -0.00028666396450471865
LassoLarsCV 의 정답률 :  0.39609375678354664
LassoLarsIC 의 정답률 :  0.39610529461222643
LinearRegression 의 정답률 :  0.39677950279105434
LinearSVR 의 정답률 :  0.33687739137611594
MLPRegressor 의 정답률 :  0.516773578426929
MultiOutputRegressor 은 실행되지 않는다.
MultiTaskElasticNet 은 실행되지 않는다.
MultiTaskElasticNetCV 은 실행되지 않는다.
MultiTaskLasso 은 실행되지 않는다.
MultiTaskLassoCV 은 실행되지 않는다.
NuSVR 의 정답률 :  0.3694651315284101
OrthogonalMatchingPursuit 의 정답률 :  0.15446184076275415
OrthogonalMatchingPursuitCV 의 정답률 :  0.3947138305667761
PLSCanonical 의 정답률 :  -0.341321985238181
PLSRegression 의 정답률 :  0.38772151010529143
PassiveAggressiveRegressor 의 정답률 :  0.34001691955106694
PoissonRegressor 의 정답률 :  0.4034500295548591
RANSACRegressor 의 정답률 :  0.2938076106376821
RadiusNeighborsRegressor 의 정답률 :  0.23875827199114585
RandomForestRegressor 의 정답률 :  0.947522096660473
RegressorChain 은 실행되지 않는다.
Ridge 의 정답률 :  0.3967095790136558
RidgeCV 의 정답률 :  0.3967095790136216
SGDRegressor 의 정답률 :  0.3954947148402399
SVR 의 정답률 :  0.36362001607119265
StackingRegressor 은 실행되지 않는다.
TheilSenRegressor 의 정답률 :  0.39493319063634347
TransformedTargetRegressor 의 정답률 :  0.39677950279105434
TweedieRegressor 의 정답률 :  0.08857451723216136
VotingRegressor 은 실행되지 않는다.

'''