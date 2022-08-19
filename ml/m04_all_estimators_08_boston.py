#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from sklearn.datasets import load_boston
import time

#1. 데이터
datasets = load_boston()
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

ARDRegression 의 정답률 :  0.8012569266997994
AdaBoostRegressor 의 정답률 :  0.9076248825639956
BaggingRegressor 의 정답률 :  0.900197943672475
BayesianRidge 의 정답률 :  0.7937918622384765
CCA 의 정답률 :  0.7913477184424629
DecisionTreeRegressor 의 정답률 :  0.7959265447226536
DummyRegressor 의 정답률 :  -0.0005370164400797517
ElasticNet 의 정답률 :  0.7338335519267194
ElasticNetCV 의 정답률 :  0.7167760356856182
ExtraTreeRegressor 의 정답률 :  0.8202642355386556
ExtraTreesRegressor 의 정답률 :  0.9336001368795159
GammaRegressor 의 정답률 :  -0.0005370164400797517
GaussianProcessRegressor 의 정답률 :  -6.073105259620457
GradientBoostingRegressor 의 정답률 :  0.9455574542650164
HistGradientBoostingRegressor 의 정답률 :  0.9323597806119726
HuberRegressor 의 정답률 :  0.7660741422086157
IsotonicRegression 은 실행되지 않는다.
KNeighborsRegressor 의 정답률 :  0.5900872726222293
KernelRidge 의 정답률 :  0.8333325493789002
Lars 의 정답률 :  0.7746736096721596
LarsCV 의 정답률 :  0.7981576314184007
Lasso 의 정답률 :  0.7240751024070102
LassoCV 의 정답률 :  0.7517507753137198
LassoLars 의 정답률 :  -0.0005370164400797517
LassoLarsCV 의 정답률 :  0.8127604328474288
LassoLarsIC 의 정답률 :  0.8131423868817642
LinearRegression 의 정답률 :  0.8111288663608656
LinearSVR 의 정답률 :  0.7809490626658678
MLPRegressor 의 정답률 :  0.53969428883143
MultiOutputRegressor 은 실행되지 않는다.
MultiTaskElasticNet 은 실행되지 않는다.
MultiTaskElasticNetCV 은 실행되지 않는다.
MultiTaskLasso 은 실행되지 않는다.
MultiTaskLassoCV 은 실행되지 않는다.
NuSVR 의 정답률 :  0.2594558622083819
OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
PLSCanonical 의 정답률 :  -2.2317079741425734
PLSRegression 의 정답률 :  0.8027313142007887
PassiveAggressiveRegressor 의 정답률 :  -0.6467612574296651
PoissonRegressor 의 정답률 :  0.8575566856857632
RANSACRegressor 의 정답률 :  0.7007898028187837
RadiusNeighborsRegressor 은 실행되지 않는다.
RandomForestRegressor 의 정답률 :  0.9242219765599895
RegressorChain 은 실행되지 않는다.
Ridge 의 정답률 :  0.8098487632912241
RidgeCV 의 정답률 :  0.8112529185024568
SGDRegressor 의 정답률 :  -1.9924011575778638e+26
SVR 의 정답률 :  0.23474677555722312
StackingRegressor 은 실행되지 않는다.
TheilSenRegressor 의 정답률 :  0.7877975498570333
TransformedTargetRegressor 의 정답률 :  0.8111288663608656
TweedieRegressor 의 정답률 :  0.7431122091082558
VotingRegressor 은 실행되지 않는다.
'''