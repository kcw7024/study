#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


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
ACC :  [0.56171883 0.6016205  0.57553389 0.60424184 0.60112895 0.61802017
 0.61837816 0.27093512 0.59697576 0.58841803]
cross_val_score :  0.5637
Model Name :  AdaBoostRegressor
ACC :  [0.44151396 0.2959587  0.42952202 0.48874915 0.38329531 0.45746468
 0.34892245 0.36433581 0.29170701 0.45410387]
cross_val_score :  0.3956
Model Name :  BaggingRegressor
ACC :  [0.75771495 0.77136957 0.767278   0.79544822 0.77465661 0.80573516
 0.7844547  0.73659353 0.78593174 0.77927893]
cross_val_score :  0.7758
Model Name :  BayesianRidge
ACC :  [0.57303024 0.60370203 0.58923479 0.61838075 0.61375553 0.63302715
 0.6255951  0.27382291 0.61041019 0.60170628]
cross_val_score :  0.5743
Model Name :  CCA
ACC :  [0.54438132 0.55677217 0.54802667 0.57748133 0.56635195 0.58846627
 0.57795304 0.37515206 0.57395002 0.55229379]
cross_val_score :  0.5461
Model Name :  DecisionTreeRegressor
ACC :  [0.52417126 0.54837022 0.57360447 0.60360704 0.58758599 0.65155756
 0.63208361 0.49213304 0.60721708 0.64810245]
cross_val_score :  0.5868
Model Name :  DummyRegressor
ACC :  [-1.62553890e-05 -7.07329910e-04 -2.56646421e-04 -2.23137080e-03
 -1.58271067e-03 -1.67352910e-04 -1.52441894e-03 -5.97139391e-03
 -9.30950032e-05 -3.31257599e-04]
cross_val_score :  -0.0013
Model Name :  ElasticNet
ACC :  [0.40738631 0.39572233 0.39962394 0.43687019 0.41846896 0.44044992
 0.42721956 0.38128491 0.43346795 0.39906867]
cross_val_score :  0.414
Model Name :  ElasticNetCV
ACC :  [0.47701134 0.57451946 0.56571728 0.60378695 0.59466231 0.61025255
 0.60099652 0.33945901 0.59439224 0.57253715]
cross_val_score :  0.5533
Model Name :  ExtraTreeRegressor
ACC :  [0.52051579 0.54830981 0.50399995 0.51800174 0.55007832 0.60371127
 0.61924586 0.52260983 0.46326039 0.56871167]
cross_val_score :  0.5418
Model Name :  ExtraTreesRegressor
ACC :  [0.78334352 0.79332567 0.78524486 0.81502969 0.80542746 0.83404542
 0.82234963 0.76303771 0.81847189 0.81872983]
cross_val_score :  0.8039
Model Name :  GammaRegressor
ACC :  [-1.63336753e-05 -7.16729673e-04 -2.59026361e-04 -2.31847041e-03
 -1.52544583e-03 -1.71762359e-04 -1.57257599e-03 -5.70876656e-03
 -9.24229109e-05 -3.30005160e-04]
cross_val_score :  -0.0013
Model Name :  GaussianProcessRegressor
ACC :  [-2.83006532 -2.93405839 -2.89682667 -2.81148116 -2.80464065 -2.76121943
 -2.89209883 -2.93998428 -2.89902223 -2.8056807 ]
cross_val_score :  -2.8575
Model Name :  GradientBoostingRegressor
ACC :  [0.76639573 0.78145704 0.75944118 0.80059304 0.78977673 0.81367633
 0.80820224 0.73976091 0.79287657 0.79593712]
cross_val_score :  0.7848
Model Name :  HistGradientBoostingRegressor
ACC :  [0.80321931 0.82659397 0.81155505 0.84806595 0.82547871 0.85290124
 0.85262252 0.79266573 0.84468136 0.84518044]
cross_val_score :  0.8303
Model Name :  HuberRegressor
ACC :  [ -0.56268162   0.40979647   0.52271981   0.55419548   0.56462112
   0.57013145   0.56876882 -10.16626569   0.49815513   0.5235142 ]
cross_val_score :  -0.6517
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [0.16037488 0.15047839 0.10041083 0.10994701 0.16807586 0.12469973
 0.14586115 0.13277328 0.14645609 0.09511195]
cross_val_score :  0.1334
Model Name :  KernelRidge
ACC :  [0.5193498  0.53551389 0.52365888 0.5686183  0.55432283 0.5850515
 0.56266449 0.17479039 0.55377547 0.54320607]
cross_val_score :  0.5121
Model Name :  Lars
ACC :  [0.57309021 0.6037823  0.58929854 0.61835551 0.61376276 0.63310454
 0.62564234 0.2733481  0.61041461 0.60179479]
cross_val_score :  0.5743
Model Name :  LarsCV
ACC :  [0.45598277 0.59794144 0.5831976  0.61708611 0.61224435 0.62901724
 0.62142424 0.32787138 0.60776233 0.59641218]
cross_val_score :  0.5649
Model Name :  Lasso
ACC :  [0.27013117 0.26022037 0.26253942 0.27962559 0.27259016 0.28255334
 0.27549499 0.25272634 0.28488845 0.26034551]
cross_val_score :  0.2701
Model Name :  LassoCV
ACC :  [0.47943286 0.57965219 0.56948988 0.60642632 0.59849618 0.61323182
 0.60543968 0.34637352 0.59698206 0.5776042 ]
cross_val_score :  0.5573
Model Name :  LassoLars
ACC :  [-1.62553890e-05 -7.07329910e-04 -2.56646421e-04 -2.23137080e-03
 -1.58271067e-03 -1.67352910e-04 -1.52441894e-03 -5.97139391e-03
 -9.30950032e-05 -3.31257599e-04]
cross_val_score :  -0.0013
Model Name :  LassoLarsCV
ACC :  [0.45598277 0.59794144 0.5831976  0.61708611 0.61224435 0.62901724
 0.62142424 0.32787138 0.60776233 0.59641218]
cross_val_score :  0.5649
Model Name :  LassoLarsIC
ACC :  [0.57309021 0.6037823  0.58929854 0.61835551 0.61376276 0.63310454
 0.62564234 0.2733481  0.61041461 0.60179479]
cross_val_score :  0.5743
Model Name :  LinearRegression
ACC :  [0.57309021 0.6037823  0.58929854 0.61835551 0.61376276 0.63310454
 0.62564234 0.2733481  0.61041461 0.60179479]
cross_val_score :  0.5743
Model Name :  LinearSVR
ACC :  [ -0.70790397   0.37356397   0.13613569  -1.42708365 -17.58937727
  -0.22577783 -14.52328031  -5.9260493   -3.67070091   0.42018657]
cross_val_score :  -4.314
Model Name :  MLPRegressor
ACC :  [-1.85539308  0.58487962  0.45872081  0.56032472  0.58382245  0.59391195
  0.64311415  0.12990489  0.23862088  0.38277467]
cross_val_score :  0.2321
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
ACC :  [ 0.00556271 -0.00460545 -0.00375468 -0.00677412  0.01254836  0.00531381
 -0.00025009  0.02062448  0.01273303  0.00941708]
cross_val_score :  0.0051
Model Name :  OrthogonalMatchingPursuit
ACC :  [0.43456188 0.45479692 0.4455261  0.49798009 0.4831101  0.50166728
 0.49199296 0.41838419 0.47850948 0.45465748]
cross_val_score :  0.4661
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.4768082  0.58658578 0.57387706 0.6052964  0.60075626 0.61338989
 0.61264523 0.2635387  0.59663323 0.58429333]
cross_val_score :  0.5514
Model Name :  PLSCanonical
ACC :  [ 0.3293938   0.39904063  0.36190108  0.42261044  0.42487743  0.4263132
  0.40105695 -0.40512499  0.38540687  0.40171358]
cross_val_score :  0.3147
Model Name :  PLSRegression
ACC :  [0.49572201 0.51013387 0.50800906 0.55126332 0.53957629 0.55943019
 0.54623726 0.09750392 0.53723201 0.51778451]
cross_val_score :  0.4863
Model Name :  PassiveAggressiveRegressor
ACC :  [ 0.16278074  0.13800319 -0.59785685 -7.2471911  -3.59913928 -3.83665703
  0.47298609 -1.66317495 -2.2506975  -0.09809647]
cross_val_score :  -1.8519
Model Name :  PoissonRegressor
ACC :  [-1.72847328e-05 -7.53808860e-04 -2.73666023e-04 -2.41538690e-03
 -1.65273466e-03 -1.80026028e-04 -1.63976143e-03 -6.20315862e-03
 -9.84912298e-05 -3.50823755e-04]
cross_val_score :  -0.0014
Model Name :  RANSACRegressor
ACC :  [ -0.31700277   0.5056133    0.4952443    0.41636706   0.56812987
   0.53163118   0.37840194 -25.25542218   0.27373171   0.53064257]
cross_val_score :  -2.1873
Model Name :  RadiusNeighborsRegressor
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
Model Name :  RandomForestRegressor
ACC :  [0.78584815 0.78757093 0.78560455 0.81685622 0.80074032 0.83279325
 0.81464791 0.75938913 0.80798915 0.8101805 ]
cross_val_score :  0.8002
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.57307678 0.60376464 0.58928436 0.61836158 0.61376125 0.63308755
 0.62563189 0.2734906  0.61041374 0.60177534]
cross_val_score :  0.5743
Model Name :  RidgeCV
ACC :  [0.57308878 0.60360517 0.58915584 0.61841083 0.61374435 0.63293503
 0.6255355  0.27336293 0.61040242 0.60160088]
cross_val_score :  0.5742
Model Name :  SGDRegressor
ACC :  [-3.70495091e+29 -1.20635752e+26 -1.18795948e+30 -1.26885284e+28
 -1.25828047e+29 -2.45742019e+29 -3.50557631e+29 -9.11577339e+30
 -4.58303463e+28 -3.54032033e+29]
cross_val_score :  -1.1809027200761703e+30
Model Name :  SVR
ACC :  [-0.02306221 -0.0422231  -0.03813303 -0.03983621 -0.01177125 -0.02669419
 -0.039371    0.00112331 -0.01530694 -0.02011668]
cross_val_score :  -0.0255
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [ -3.16798704   0.12789122   0.57921206   0.51851559   0.57393896
   0.50086257   0.53285456 -29.67089238   0.40116787   0.53451926]
cross_val_score :  -2.907
Model Name :  TransformedTargetRegressor
ACC :  [0.57309021 0.6037823  0.58929854 0.61835551 0.61376276 0.63310454
 0.62564234 0.2733481  0.61041461 0.60179479]
cross_val_score :  0.5743
Model Name :  TweedieRegressor
ACC :  [0.47160604 0.46840914 0.46620615 0.51999677 0.49628614 0.52612615
 0.50193891 0.18805013 0.50930051 0.4784331 ]
cross_val_score :  0.4626
VotingRegressor 은 실행되지 않는다.
'''