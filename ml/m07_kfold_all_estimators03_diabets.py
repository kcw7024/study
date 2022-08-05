#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
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
ACC :  [0.37895863 0.56129935 0.21352736 0.32305154 0.11655736 0.54184305
 0.49654491 0.45886448 0.46710815 0.55381268]
cross_val_score :  0.4112
Model Name :  AdaBoostRegressor
ACC :  [0.29846857 0.47810324 0.23159245 0.13705379 0.0992133  0.38420399
 0.47872934 0.36270612 0.46918644 0.5637749 ]
cross_val_score :  0.3503
Model Name :  BaggingRegressor
ACC :  [0.19787809 0.54777091 0.06333408 0.20779464 0.09273709 0.52308377
 0.42957363 0.15110025 0.50043246 0.63006783]
cross_val_score :  0.3344
Model Name :  BayesianRidge
ACC :  [0.3982426  0.57016254 0.2277455  0.32240503 0.12392284 0.53106135
 0.49790113 0.45163697 0.45233643 0.56463077]
cross_val_score :  0.414
Model Name :  CCA
ACC :  [0.39075924 0.57215321 0.23148531 0.24955781 0.0869325  0.55720526
 0.49371084 0.39371165 0.49146006 0.54039743]
cross_val_score :  0.4007
Model Name :  DecisionTreeRegressor
ACC :  [-0.04367234  0.28621171 -0.03665795 -0.20523667 -0.43345749  0.15113217
  0.30217942 -1.17207907  0.50612824  0.32027402]
cross_val_score :  -0.0325
Model Name :  DummyRegressor
ACC :  [-0.02978824 -0.09788185 -0.12532538 -0.01554421 -0.01906559 -0.00047405
 -0.05021567 -0.00029285 -0.01880038 -0.02123992]
cross_val_score :  -0.0379
Model Name :  ElasticNet
ACC :  [-0.02231432 -0.08908167 -0.11909187 -0.00800309 -0.01355746  0.00779837
 -0.04144508  0.00829386 -0.01322307 -0.01063373]
cross_val_score :  -0.0301
Model Name :  ElasticNetCV
ACC :  [0.37887097 0.45300676 0.20715733 0.30538845 0.17985612 0.46016161
 0.42969906 0.44307521 0.35512516 0.54689837]
cross_val_score :  0.3759
Model Name :  ExtraTreeRegressor
ACC :  [-0.46652592 -0.47095033  0.0586654  -0.98935652 -0.80782142 -0.37359442
 -0.0649477  -0.0672249  -0.02532429  0.14059822]
cross_val_score :  -0.3066
Model Name :  ExtraTreesRegressor
ACC :  [0.3332795  0.56100893 0.2819555  0.18893402 0.21394229 0.46941763
 0.5047534  0.25211289 0.4646559  0.61030317]
cross_val_score :  0.388
Model Name :  GammaRegressor
ACC :  [-0.02156336 -0.08059526 -0.13387839 -0.0096745  -0.01265587  0.00546521
 -0.03545457  0.0049623  -0.0146165  -0.01120753]
cross_val_score :  -0.0309
Model Name :  GaussianProcessRegressor
ACC :  [-14.92716635 -17.03175065  -9.01727215 -31.59364211 -17.55386372
  -9.56814858  -6.36479385  -7.90353562  -5.70218935  -9.36499746]
cross_val_score :  -12.9027
Model Name :  GradientBoostingRegressor
ACC :  [0.26619237 0.59979347 0.07186242 0.1235182  0.1928846  0.49436667
 0.34316535 0.22083127 0.53808532 0.52889575]
cross_val_score :  0.338
Model Name :  HistGradientBoostingRegressor
ACC :  [ 0.09623172  0.57658635  0.17149075 -0.05569848  0.22258966  0.4396107
  0.43372459  0.23623098  0.50937958  0.48128882]
cross_val_score :  0.3111
Model Name :  HuberRegressor
ACC :  [0.37139044 0.58833966 0.18434249 0.31395524 0.10382723 0.54337783
 0.51726695 0.44674197 0.46879852 0.53793978]
cross_val_score :  0.4076
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [0.24847814 0.49777108 0.14643462 0.22100704 0.10169608 0.27381397
 0.14055281 0.20778172 0.35084096 0.4989864 ]
cross_val_score :  0.2687
Model Name :  KernelRidge
ACC :  [-3.79817541 -2.59062515 -4.82954714 -4.17223648 -4.24309705 -4.31945258
 -3.24125965 -4.17520482 -2.99233333 -4.3921477 ]
cross_val_score :  -3.8754
Model Name :  Lars
ACC :  [0.39120858 0.57325291 0.23536277 0.32731035 0.1018405  0.54116032
 0.48049996 0.45231258 0.46862395 0.56025769]
cross_val_score :  0.4132
Model Name :  LarsCV
ACC :  [0.38191008 0.54599778 0.21609749 0.30977848 0.1018405  0.53429439
 0.49281396 0.46080148 0.41890158 0.55908772]
cross_val_score :  0.4022
Model Name :  Lasso
ACC :  [0.26094353 0.30602967 0.09835265 0.23604722 0.21585484 0.29741807
 0.25539394 0.35395338 0.26125569 0.38633544]
cross_val_score :  0.2672
Model Name :  LassoCV
ACC :  [0.38937852 0.54921865 0.21808792 0.31768843 0.10250424 0.53679832
 0.50209132 0.45045758 0.46685269 0.56014849]
cross_val_score :  0.4093
Model Name :  LassoLars
ACC :  [0.29153023 0.35740496 0.11359073 0.25555644 0.2259582  0.34434593
 0.31735687 0.38954022 0.30488228 0.43568123]
cross_val_score :  0.3036
Model Name :  LassoLarsCV
ACC :  [0.38967087 0.54599778 0.21419587 0.31707215 0.10246683 0.54116032
 0.5016725  0.45081363 0.46862395 0.56025769]
cross_val_score :  0.4092
Model Name :  LassoLarsIC
ACC :  [0.38280068 0.54392965 0.21336343 0.31344576 0.14564962 0.52795753
 0.50165458 0.45582838 0.44846728 0.55535252]
cross_val_score :  0.4088
Model Name :  LinearRegression
ACC :  [0.39120858 0.57325291 0.23536277 0.32731035 0.1018405  0.54116032
 0.5016725  0.45231258 0.46862395 0.56025769]
cross_val_score :  0.4153
Model Name :  LinearSVR
ACC :  [-0.56188608 -0.0899172  -0.98163169 -0.63707448 -0.38712854 -0.3800373
 -0.1701559  -0.45475669 -0.41841678 -0.28917562]
cross_val_score :  -0.437
Model Name :  MLPRegressor
ACC :  [-3.2521527  -2.19322486 -4.35892202 -3.821734   -3.30721918 -3.08595584
 -2.34963006 -3.69108656 -2.68424823 -3.23332658]
cross_val_score :  -3.1978
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
ACC :  [ 0.07148735  0.11358505 -0.067059    0.05909346  0.10433992  0.1615512
  0.12163116  0.14686651  0.07527485  0.21151295]
cross_val_score :  0.0998
Model Name :  OrthogonalMatchingPursuit
ACC :  [ 0.22965953  0.45583541 -0.26066129  0.1481988   0.17051055  0.21065959
  0.14691104  0.28894749  0.13318636  0.36784029]
cross_val_score :  0.1891
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.31943886 0.5635688  0.20926831 0.33023768 0.10107312 0.5567581
 0.49800557 0.46761202 0.47172961 0.54864766]
cross_val_score :  0.4066
Model Name :  PLSCanonical
ACC :  [-0.98801083 -0.84073281 -0.92496001 -3.37143827 -2.36002751 -1.08500151
 -1.51219    -1.61612615 -0.43910133 -1.96398248]
cross_val_score :  -1.5102
Model Name :  PLSRegression
ACC :  [0.39706826 0.58591339 0.25311311 0.35633691 0.12208038 0.50996531
 0.47046748 0.3994111  0.46504529 0.57034052]
cross_val_score :  0.413
Model Name :  PassiveAggressiveRegressor
ACC :  [0.39388685 0.46535346 0.17755416 0.33254255 0.15839054 0.47629147
 0.45728076 0.46648613 0.35351006 0.55802901]
cross_val_score :  0.3839
Model Name :  PoissonRegressor
ACC :  [0.28917737 0.28290592 0.13381132 0.24617538 0.15893844 0.34583939
 0.28999537 0.32240245 0.24488524 0.41512058]
cross_val_score :  0.2729
Model Name :  RANSACRegressor
ACC :  [ 0.00555213  0.40295589  0.11782625 -0.28111447  0.06938636  0.40843557
  0.02837925  0.07135619 -0.27448476  0.12456014]
cross_val_score :  0.0673
Model Name :  RadiusNeighborsRegressor
ACC :  [-0.02978824 -0.09788185 -0.12532538 -0.01554421 -0.01906559 -0.00047405
 -0.05021567 -0.00029285 -0.01880038 -0.02123992]
cross_val_score :  -0.0379
Model Name :  RandomForestRegressor
ACC :  [0.33700029 0.59110994 0.25727009 0.13391164 0.1711511  0.53006578
 0.46593498 0.26769976 0.49632346 0.62965967]
cross_val_score :  0.388
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.35440595 0.40002024 0.18463558 0.28985213 0.17797337 0.43170814
 0.39580299 0.41864992 0.31078573 0.51582573]
cross_val_score :  0.348
Model Name :  RidgeCV
ACC :  [0.40214851 0.55748296 0.22973309 0.32669911 0.14294344 0.52313063
 0.49426208 0.45840261 0.44040165 0.5713199 ]
cross_val_score :  0.4147
Model Name :  SGDRegressor
ACC :  [0.35995336 0.3799568  0.17963025 0.25968147 0.15797118 0.43649041
 0.37503641 0.4083071  0.28763276 0.51256351]
cross_val_score :  0.3357
Model Name :  SVR
ACC :  [ 0.03535914  0.17659506 -0.10602062  0.01604077  0.12929003  0.16331893
  0.17645461  0.14473503  0.06752607  0.24392059]
cross_val_score :  0.1047
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [0.37041761 0.57313403 0.17960417 0.24978382 0.12008874 0.53437402
 0.5188269  0.4570874  0.44546643 0.54199369]
cross_val_score :  0.3991
Model Name :  TransformedTargetRegressor
ACC :  [0.39120858 0.57325291 0.23536277 0.32731035 0.1018405  0.54116032
 0.5016725  0.45231258 0.46862395 0.56025769]
cross_val_score :  0.4153
Model Name :  TweedieRegressor
ACC :  [-0.02373023 -0.09119529 -0.12059787 -0.00981333 -0.01493277  0.00635776
 -0.04365895  0.00638933 -0.0146467  -0.01269603]
cross_val_score :  -0.0319
VotingRegressor 은 실행되지 않는다.
'''