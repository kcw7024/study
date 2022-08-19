#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import r2_score


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

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
ARDRegression 의 정답률 :  0.6150897245057487
AdaBoostRegressor 의 정답률 :  0.4157593792076174
BaggingRegressor 의 정답률 :  0.8051235966689778
BayesianRidge 의 정답률 :  0.6161106155953302
CCA 의 정답률 :  0.5787712997952789
DecisionTreeRegressor 의 정답률 :  0.6188232254838641
DummyRegressor 의 정답률 :  -0.00032418638108300257
ElasticNet 의 정답률 :  0.4367013171669184
ElasticNetCV 의 정답률 :  0.5961473821049225
ExtraTreeRegressor 의 정답률 :  0.5775022390681899
ExtraTreesRegressor 의 정답률 :  0.8258712931890331
GammaRegressor 의 정답률 :  -0.00032418638108300257
GaussianProcessRegressor 의 정답률 :  -2.726194481645148
GradientBoostingRegressor 의 정답률 :  0.8005238581563235
HistGradientBoostingRegressor 의 정답률 :  0.8482846765220788
HuberRegressor 의 정답률 :  -11.151998161891376
IsotonicRegression 은 실행되지 않는다.
KNeighborsRegressor 의 정답률 :  0.1585059607598721
KernelRidge 의 정답률 :  0.5460528791238092
Lars 의 정답률 :  0.6161406602616106
LarsCV 의 정답률 :  0.6165012527241961
Lasso 의 정답률 :  0.28647943360036954
LassoCV 의 정답률 :  0.6007326417887116
LassoLars 의 정답률 :  -0.00032418638108300257
LassoLarsCV 의 정답률 :  0.6165012527241961
LassoLarsIC 의 정답률 :  0.6161406602616106
LinearRegression 의 정답률 :  0.6161406602616104
LinearSVR 의 정답률 :  -5.990675828543982
MLPRegressor 의 정답률 :  0.6281164653503732
MultiOutputRegressor 은 실행되지 않는다.
MultiTaskElasticNet 은 실행되지 않는다.
MultiTaskElasticNetCV 은 실행되지 않는다.
MultiTaskLasso 은 실행되지 않는다.
MultiTaskLassoCV 은 실행되지 않는다.
NuSVR 의 정답률 :  0.00495415663004517
OrthogonalMatchingPursuit 의 정답률 :  0.4949710792348151
OrthogonalMatchingPursuitCV 의 정답률 :  0.6135096270901488
PLSCanonical 의 정답률 :  0.3584924231283567
PLSRegression 의 정답률 :  0.5250313399190631
PassiveAggressiveRegressor 의 정답률 :  -4.719511143195764
PoissonRegressor 의 정답률 :  -0.00032418638108300257
RANSACRegressor 의 정답률 :  -10.126877024914872
RadiusNeighborsRegressor 은 실행되지 않는다.
RandomForestRegressor 의 정답률 :  0.8239841393374512
RegressorChain 은 실행되지 않는다.
Ridge 의 정답률 :  0.6161338693594268
RidgeCV 의 정답률 :  0.6160703168791888
SGDRegressor 의 정답률 :  -1.0278308981187916e+29
SVR 의 정답률 :  -0.026503353134800234
StackingRegressor 은 실행되지 않는다.
TheilSenRegressor 의 정답률 :  -43.26759309866334
TransformedTargetRegressor 의 정답률 :  0.6161406602616104
TweedieRegressor 의 정답률 :  0.4893883239259552
VotingRegressor 은 실행되지 않는다.
'''