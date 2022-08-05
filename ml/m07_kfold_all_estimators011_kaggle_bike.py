# Kaggle Bike_sharing
import numpy as np
import pandas as pd 
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
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

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


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

Model Name :  ARDRegression
ACC :  [0.38238443 0.35955558 0.39403707 0.4026843  0.40208391 0.39000025
 0.36009866 0.4021928  0.37853522 0.41484286]
cross_val_score :  0.3886
Model Name :  AdaBoostRegressor
ACC :  [0.65542743 0.6808942  0.68798261 0.67988927 0.6803083  0.63357637
 0.66000462 0.67572982 0.6519418  0.67773888]
cross_val_score :  0.6683
Model Name :  BaggingRegressor
ACC :  [0.95128529 0.92667722 0.94366156 0.94073115 0.93872853 0.934468
 0.94808449 0.94211312 0.94740254 0.93555898]
cross_val_score :  0.9409
Model Name :  BayesianRidge
ACC :  [0.38222792 0.36005773 0.39402934 0.40483189 0.40226522 0.38947074
 0.35968523 0.40279076 0.37939525 0.41429224]
cross_val_score :  0.3889
Model Name :  CCA
ACC :  [ 0.07115824 -0.06223277  0.10104684  0.06735442  0.03464453  0.0147394
  0.05149801  0.11090747  0.13535114  0.05393753]
cross_val_score :  0.0578
Model Name :  DecisionTreeRegressor
ACC :  [0.91722913 0.88633612 0.88949185 0.89476643 0.88756613 0.88566205
 0.90171848 0.90692244 0.9127177  0.86866905]
cross_val_score :  0.8951
Model Name :  DummyRegressor
ACC :  [-2.41361117e-05 -1.00087202e-03 -1.72231557e-03 -5.98684761e-04
 -3.14282872e-04 -2.11007203e-04 -1.44676096e-04 -7.00774907e-05
 -3.67726592e-03 -2.49426957e-03]
cross_val_score :  -0.001
Model Name :  ElasticNet
ACC :  [0.13556112 0.14198729 0.1382273  0.14469773 0.1414084  0.14649254
 0.13459921 0.14044334 0.13517024 0.143526  ]
cross_val_score :  0.1402
Model Name :  ElasticNetCV
ACC :  [0.36658472 0.35807174 0.37760208 0.39133793 0.38623552 0.38044955
 0.35414654 0.38422344 0.36632173 0.39643803]
cross_val_score :  0.3761
Model Name :  ExtraTreeRegressor
ACC :  [0.88805167 0.84283388 0.905184   0.89009812 0.86629379 0.84826543
 0.89308861 0.8516395  0.80078847 0.77731522]
cross_val_score :  0.8564
Model Name :  ExtraTreesRegressor
ACC :  [0.9595173  0.94810478 0.94400904 0.947062   0.9435137  0.94976549
 0.95337037 0.95161378 0.95227254 0.94773184]
cross_val_score :  0.9497
Model Name :  GammaRegressor
ACC :  [0.05713606 0.05317428 0.05648681 0.05903069 0.05805813 0.05974116
 0.05885147 0.05998916 0.0604646  0.0561845 ]
cross_val_score :  0.0579
Model Name :  GaussianProcessRegressor
ACC :  [-38.97654952 -26.4555004  -17.03587793 -23.64463406 -54.52625846
 -90.76334643 -59.05233486 -16.12486146 -47.20963347 -23.99478209]
cross_val_score :  -39.7784
Model Name :  GradientBoostingRegressor
ACC :  [0.87303979 0.86496149 0.85484689 0.86302976 0.88520204 0.87448161
 0.86956972 0.86387025 0.86420611 0.87829853]
cross_val_score :  0.8692
Model Name :  HistGradientBoostingRegressor
ACC :  [0.95608084 0.942977   0.94714729 0.95596082 0.94616513 0.94357661
 0.95924879 0.95435848 0.95236368 0.94504706]
cross_val_score :  0.9503
Model Name :  HuberRegressor
ACC :  [0.34990914 0.33209639 0.3534502  0.37386571 0.3821292  0.36058065
 0.33784872 0.36298962 0.33028487 0.38936666]
cross_val_score :  0.3573
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [0.66493743 0.6541183  0.64644834 0.68390446 0.69233518 0.65104312
 0.64696285 0.6615423  0.67262845 0.70213463]
cross_val_score :  0.6676
Model Name :  KernelRidge
ACC :  [0.38188387 0.35925586 0.39332145 0.40345482 0.40086181 0.39098966
 0.35764057 0.40070299 0.38037897 0.4144639 ]
cross_val_score :  0.3883
Model Name :  Lars
ACC :  [0.38228807 0.35991395 0.39426652 0.40434639 0.40254926 0.3891609
 0.35948842 0.40301927 0.37893314 0.4145088 ]
cross_val_score :  0.3888
Model Name :  LarsCV
ACC :  [0.38233131 0.36038537 0.39357265 0.40344479 0.40222278 0.39056578
 0.36092836 0.40164437 0.37728033 0.4147073 ]
cross_val_score :  0.3887
Model Name :  Lasso
ACC :  [0.37992579 0.36028829 0.38725303 0.40303728 0.39745205 0.39006722
 0.3597899  0.39660817 0.37554485 0.41297375]
cross_val_score :  0.3863
Model Name :  LassoCV
ACC :  [0.3821272  0.3600387  0.39348627 0.40393925 0.40179958 0.38945355
 0.36093598 0.40207745 0.37899411 0.41441332]
cross_val_score :  0.3887
Model Name :  LassoLars
ACC :  [-2.41361117e-05 -1.00087202e-03 -1.72231557e-03 -5.98684761e-04
 -3.14282872e-04 -2.11007203e-04 -1.44676096e-04 -7.00774907e-05
 -3.67726592e-03 -2.49426957e-03]
cross_val_score :  -0.001
Model Name :  LassoLarsCV
ACC :  [0.38233131 0.36038537 0.39357265 0.40344479 0.40222278 0.39056578
 0.36092836 0.40164437 0.37799033 0.4147073 ]
cross_val_score :  0.3888
Model Name :  LassoLarsIC
ACC :  [0.38233004 0.36077011 0.39393444 0.40357693 0.4022465  0.39063495
 0.36080977 0.40167289 0.37717808 0.41474136]
cross_val_score :  0.3888
Model Name :  LinearRegression
ACC :  [0.38228807 0.35991395 0.39426652 0.40434639 0.40254926 0.3891609
 0.35948842 0.40301927 0.37893314 0.4145088 ]
cross_val_score :  0.3888
Model Name :  LinearSVR
ACC :  [0.31136461 0.30322862 0.30820919 0.33670306 0.34493442 0.32246019
 0.30873573 0.31897211 0.29338961 0.34870379]
cross_val_score :  0.3197
Model Name :  MLPRegressor
ACC :  [0.47827921 0.43919929 0.4795339  0.48820258 0.49989652 0.47794202
 0.47857538 0.45040084 0.46744962 0.49523603]
cross_val_score :  0.4755
MultiOutputRegressor 은 실행되지 않는다.
Model Name :  MultiTaskElasticNet
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskElasticNet 은 실행되지 않는다.
Model Name :  MultiTaskElasticNetCV
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskElasticNetCV 은 실행되지 않는다.
Model Name :  MultiTaskLasso
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskLasso 은 실행되지 않는다.
Model Name :  MultiTaskLassoCV
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskLassoCV 은 실행되지 않는다.
Model Name :  NuSVR
ACC :  [0.33686593 0.33609294 0.33618722 0.36783809 0.3664704  0.34743876
 0.33714443 0.35202029 0.31926107 0.37028105]
cross_val_score :  0.347
Model Name :  OrthogonalMatchingPursuit
ACC :  [0.13931541 0.13888677 0.16431076 0.17433452 0.17373996 0.13738426
 0.17893416 0.16386933 0.13590291 0.20265469]
cross_val_score :  0.1609
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.38149296 0.35807629 0.39130238 0.40517645 0.39870938 0.3911391
 0.35953792 0.40063775 0.37943345 0.41442642]
cross_val_score :  0.388
Model Name :  PLSCanonical
ACC :  [-0.32705583 -0.51540605 -0.25955295 -0.30686001 -0.38270018 -0.36965464
 -0.3431274  -0.24315303 -0.23369563 -0.34455821]
cross_val_score :  -0.3326
Model Name :  PLSRegression
ACC :  [0.37487161 0.35651885 0.39024788 0.4016301  0.39985715 0.38820526
 0.35431114 0.39392813 0.37020238 0.40387735]
cross_val_score :  0.3834
Model Name :  PassiveAggressiveRegressor
ACC :  [0.3332927  0.30678107 0.34482353 0.3891701  0.35273868 0.31938639
 0.3516454  0.30635912 0.35241804 0.35776137]
cross_val_score :  0.3414
Model Name :  PoissonRegressor
ACC :  [0.40478885 0.37137758 0.42578981 0.43383565 0.43270261 0.41214006
 0.40507287 0.43438455 0.42115653 0.44116033]
cross_val_score :  0.4182
Model Name :  RANSACRegressor
ACC :  [ 0.12272926 -0.03474675  0.14355967  0.11491756  0.07088976  0.29932191
  0.15433313  0.21463917  0.24629078  0.1436917 ]
cross_val_score :  0.1476
Model Name :  RadiusNeighborsRegressor
ACC :  [0.22606102 0.22279616 0.23560804 0.23460241 0.22676086 0.24416645
 0.21542502 0.24240584 0.24563359 0.23509074]
cross_val_score :  0.2329
Model Name :  RandomForestRegressor
ACC :  [0.95735934 0.93293119 0.9471826  0.94695867 0.94632399 0.93994556
 0.95444438 0.94919525 0.95486548 0.94023036]
cross_val_score :  0.9469
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.38225934 0.35999255 0.39412391 0.40467552 0.40237963 0.38935628
 0.35959581 0.40289444 0.3792602  0.41438897]
cross_val_score :  0.3889
Model Name :  RidgeCV
ACC :  [0.38186209 0.3606568  0.39350739 0.40467552 0.40170076 0.38935628
 0.3605011  0.40201408 0.3792602  0.41359313]
cross_val_score :  0.3887
Model Name :  SGDRegressor
ACC :  [0.38104272 0.35773902 0.3928771  0.40457436 0.3999125  0.39030984
 0.35935787 0.40073779 0.37978231 0.41213186]
cross_val_score :  0.3878
Model Name :  SVR
ACC :  [0.32507346 0.31570375 0.31611253 0.35586436 0.35884678 0.33201075
 0.33046233 0.33885283 0.30713843 0.36725176]
cross_val_score :  0.3347
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [0.37657297 0.35629013 0.39063806 0.39845692 0.40336585 0.38038773
 0.36135251 0.39783802 0.36853881 0.41550762]
cross_val_score :  0.3849
Model Name :  TransformedTargetRegressor
ACC :  [0.38228807 0.35991395 0.39426652 0.40434639 0.40254926 0.3891609
 0.35948842 0.40301927 0.37893314 0.4145088 ]
cross_val_score :  0.3888
Model Name :  TweedieRegressor
ACC :  [0.08375193 0.08792287 0.08480639 0.08927866 0.08719554 0.09080281
 0.08338322 0.08677436 0.08243495 0.08754822]
cross_val_score :  0.0864
VotingRegressor 은 실행되지 않는다.
'''