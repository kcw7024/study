#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
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
        print(name, '의 정답률 : ', r2 )
    except:
        # continue
        print(name, '은 실행되지 않는다.')



'''
ARDRegression 의 정답률 :  0.6530588624615992
AdaBoostRegressor 의 정답률 :  0.531249110130183
BaggingRegressor 의 정답률 :  0.5414863823088113
BayesianRidge 의 정답률 :  0.6530747203807882   
CCA 의 정답률 :  0.66538953552954
DecisionTreeRegressor 의 정답률 :  0.11066862184822479
DummyRegressor 의 정답률 :  -0.014400454257647022
ElasticNet 의 정답률 :  -0.005328255954596406
ElasticNetCV 의 정답률 :  0.5416201538470302
ExtraTreeRegressor 의 정답률 :  0.15641240244023735
ExtraTreesRegressor 의 정답률 :  0.5779803486415425
GammaRegressor 의 정답률 :  -0.007823542301776287
GaussianProcessRegressor 의 정답률 :  -19.915150998908704
GradientBoostingRegressor 의 정답률 :  0.5239394090395422
HistGradientBoostingRegressor 의 정답률 :  0.5348545542371936
HuberRegressor 의 정답률 :  0.656984938799722
IsotonicRegression 은 실행되지 않는다.
KNeighborsRegressor 의 정답률 :  0.5403351561734346
KernelRidge 의 정답률 :  -3.408156802444129
Lars 의 정답률 :  0.6579209558684549
LarsCV 의 정답률 :  0.6564491674569455
Lasso 의 정답률 :  0.37852179252759377
LassoCV 의 정답률 :  0.658623475226203
LassoLars 의 정답률 :  0.41547460227639654
LassoLarsCV 의 정답률 :  0.657920955868455
LassoLarsIC 의 정답률 :  0.6459612374718955
LinearRegression 의 정답률 :  0.6579209558684551
LinearSVR 의 정답률 :  -0.44900647429127627
MLPRegressor 의 정답률 :  -2.99842253913036
MultiOutputRegressor 은 실행되지 않는다.
MultiTaskElasticNet 은 실행되지 않는다.
MultiTaskElasticNetCV 은 실행되지 않는다.
MultiTaskLasso 은 실행되지 않는다.
MultiTaskLassoCV 은 실행되지 않는다.
NuSVR 의 정답률 :  0.15030505970863384
OrthogonalMatchingPursuit 의 정답률 :  0.4157059575736308
OrthogonalMatchingPursuitCV 의 정답률 :  0.6538215435424612
PLSCanonical 의 정답률 :  -0.8475175977502751
PLSRegression 의 정답률 :  0.654541804077198
PassiveAggressiveRegressor 의 정답률 :  0.5232269388753091
PoissonRegressor 의 정답률 :  0.39364694991272375
RANSACRegressor 의 정답률 :  0.21920839706553852
RadiusNeighborsRegressor 의 정답률 :  -0.014400454257647022
RandomForestRegressor 의 정답률 :  0.5509788924821066
RegressorChain 은 실행되지 않는다.
Ridge 의 정답률 :  0.5059109261010939
RidgeCV 의 정답률 :  0.6432620271172018
SGDRegressor 의 정답률 :  0.48428939485000966
SVR 의 정답률 :  0.14455787342934745
StackingRegressor 은 실행되지 않는다.
TheilSenRegressor 의 정답률 :  0.65139596014595
TransformedTargetRegressor 의 정답률 :  0.6579209558684551
TweedieRegressor 의 정답률 :  -0.00749753759666727
VotingRegressor 은 실행되지 않는다.

'''