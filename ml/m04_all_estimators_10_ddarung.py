# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
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
ARDRegression 의 정답률 :  0.6355128639066344
AdaBoostRegressor 의 정답률 :  0.5845961985704509
BaggingRegressor 의 정답률 :  0.7959100285701535
BayesianRidge 의 정답률 :  0.6343813039445048
CCA 의 정답률 :  0.37536608531068993
DecisionTreeRegressor 의 정답률 :  0.6410398980967464
DummyRegressor 의 정답률 :  -0.0014266030081080405
ElasticNet 의 정답률 :  0.5335508186174258
ElasticNetCV 의 정답률 :  0.6210926069487763
ExtraTreeRegressor 의 정답률 :  0.6048015138833245
ExtraTreesRegressor 의 정답률 :  0.854800552958622
GammaRegressor 의 정답률 :  0.4611059935633042
GaussianProcessRegressor 의 정답률 :  0.41903564452701414
GradientBoostingRegressor 의 정답률 :  0.8157156154253873
HistGradientBoostingRegressor 의 정답률 :  0.8405863880397957
HuberRegressor 의 정답률 :  0.6176744381999368
IsotonicRegression 은 실행되지 않는다.
KNeighborsRegressor 의 정답률 :  0.6607594744356601
KernelRidge 의 정답률 :  -0.1716120571269808
Lars 의 정답률 :  0.63536620139742
LarsCV 의 정답률 :  0.6348388799697122
Lasso 의 정답률 :  0.6191469149117221
LassoCV 의 정답률 :  0.6349504079852805
LassoLars 의 정답률 :  0.29304184108007225
LassoLarsCV 의 정답률 :  0.6348388799697122
LassoLarsIC 의 정답률 :  0.6316063282320595
LinearRegression 의 정답률 :  0.63536620139742
LinearSVR 의 정답률 :  0.5350343964147886
MLPRegressor 의 정답률 :  0.5713411905131447
MultiOutputRegressor 은 실행되지 않는다.
MultiTaskElasticNet 은 실행되지 않는다.
MultiTaskElasticNetCV 은 실행되지 않는다.
MultiTaskLasso 은 실행되지 않는다.
MultiTaskLassoCV 은 실행되지 않는다.
NuSVR 의 정답률 :  0.39851170942274783
OrthogonalMatchingPursuit 의 정답률 :  0.408429451172
OrthogonalMatchingPursuitCV 의 정답률 :  0.6171485754974859
PLSCanonical 의 정답률 :  -0.11338536505482644
PLSRegression 의 정답률 :  0.6287903355798831
PassiveAggressiveRegressor 의 정답률 :  0.5910182970975658
PoissonRegressor 의 정답률 :  0.6747410977165216
RANSACRegressor 의 정답률 :  0.474689436550889
RadiusNeighborsRegressor 은 실행되지 않는다.
RandomForestRegressor 의 정답률 :  0.8259250015492314
RegressorChain 은 실행되지 않는다.
Ridge 의 정답률 :  0.6351363294437448
RidgeCV 의 정답률 :  0.6351363294436061
SGDRegressor 의 정답률 :  0.6315019427519628
SVR 의 정답률 :  0.41232855256467515
StackingRegressor 은 실행되지 않는다.
TheilSenRegressor 의 정답률 :  0.6137199627310708
TransformedTargetRegressor 의 정답률 :  0.63536620139742
TweedieRegressor 의 정답률 :  0.46891583401457126
VotingRegressor 은 실행되지 않는다.

'''