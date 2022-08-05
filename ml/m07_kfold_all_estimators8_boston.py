#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
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
import numpy as np


#1. 데이터
datasets = load_boston()
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
ACC :  [0.68794273 0.3213958  0.71075454 0.68677906 0.66861061 0.62711997
 0.84196564 0.68254668 0.73550764 0.66131201]
cross_val_score :  0.6624
Model Name :  AdaBoostRegressor
ACC :  [0.89322171 0.77407172 0.73307874 0.81197351 0.83411201 0.73769697
 0.80124381 0.73454805 0.79817692 0.77607377]
cross_val_score :  0.7894
Model Name :  BaggingRegressor
ACC :  [0.84751312 0.8824871  0.8192125  0.78716465 0.79137628 0.64338585
 0.86448759 0.82106222 0.86168091 0.86155454]
cross_val_score :  0.818
Model Name :  BayesianRidge
ACC :  [0.7012691  0.35444278 0.71997786 0.68104084 0.65233474 0.64146607
 0.81642384 0.68359    0.73060171 0.62631398]
cross_val_score :  0.6607
Model Name :  CCA
ACC :  [0.71521108 0.24238451 0.72002655 0.69271962 0.49758871 0.62833632
 0.81245614 0.67933089 0.61293701 0.64153732]
cross_val_score :  0.6243
Model Name :  DecisionTreeRegressor
ACC :  [0.65749957 0.67643878 0.75722475 0.53561736 0.47674882 0.641539
 0.81243738 0.78772792 0.69760495 0.84236859]
cross_val_score :  0.6885
Model Name :  DummyRegressor
ACC :  [-6.81880414e-02 -2.76487950e-03 -1.44238932e-01 -5.88903445e-04
 -2.19968984e-02 -3.88829506e-03 -9.13236757e-02 -1.68612726e-02
 -2.62354458e-03 -4.17226033e-05]
cross_val_score :  -0.0353
Model Name :  ElasticNet
ACC :  [0.66109764 0.27583568 0.67759458 0.63806298 0.62123543 0.66210638
 0.73847399 0.68673324 0.70611768 0.58097612]
cross_val_score :  0.6248
Model Name :  ElasticNetCV
ACC :  [0.64244359 0.24716146 0.6611242  0.61566545 0.61646823 0.65575199
 0.71111384 0.67739986 0.70071604 0.56751996]
cross_val_score :  0.6095
Model Name :  ExtraTreeRegressor
ACC :  [0.82917536 0.83847894 0.81478142 0.6148901  0.82976838 0.09815478
 0.74117124 0.80806746 0.47722968 0.63567548]
cross_val_score :  0.6687
Model Name :  ExtraTreesRegressor
ACC :  [0.91835976 0.77285586 0.85481265 0.83419028 0.86961025 0.66877706
 0.91870748 0.89346247 0.91090099 0.8056108 ]
cross_val_score :  0.8447
Model Name :  GammaRegressor
ACC :  [-8.89105904e-02 -3.07930863e-03 -1.02040998e-01 -7.09934015e-04
 -1.96944605e-02 -4.04166153e-03 -1.13199335e-01 -1.54536393e-02
 -2.56067127e-03 -4.29572030e-05]
cross_val_score :  -0.035
Model Name :  GaussianProcessRegressor
ACC :  [ -5.95982606 -10.17012977  -6.57215643  -5.31233152  -6.25411504
  -8.04083372  -7.08269877  -3.82169369  -5.35890263  -4.95856199]
cross_val_score :  -6.3531
Model Name :  GradientBoostingRegressor
ACC :  [0.90329306 0.84111183 0.8665784  0.78397009 0.8658554  0.79227034
 0.88151284 0.88383002 0.87833512 0.90611631]
cross_val_score :  0.8603
Model Name :  HistGradientBoostingRegressor
ACC :  [0.90312138 0.63919922 0.84639099 0.82332049 0.80594355 0.72251533
 0.87911249 0.88181111 0.85092605 0.857982  ]
cross_val_score :  0.821
Model Name :  HuberRegressor
ACC :  [0.63525464 0.30195086 0.66098702 0.47552479 0.5812717  0.32699521
 0.54124771 0.57956629 0.63699451 0.38085564]
cross_val_score :  0.5121
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [ 0.60263264 -0.29439515  0.68729528  0.25552932  0.71226426  0.18648896
  0.54271693  0.55971729  0.62751678  0.43982133]
cross_val_score :  0.432
Model Name :  KernelRidge
ACC :  [0.64530498 0.41659265 0.66721571 0.63712406 0.643324   0.46867269
 0.85654443 0.61053149 0.73032305 0.63697346]
cross_val_score :  0.6313
Model Name :  Lars
ACC :  [0.6812543  0.32542843 0.72840672 0.66863546 0.62900817 0.6330711
 0.85053344 0.69453457 0.74701467 0.67329687]
cross_val_score :  0.6631
Model Name :  LarsCV
ACC :  [0.67249953 0.39315774 0.72722061 0.67383387 0.64068191 0.63607372
 0.85083269 0.69378497 0.74775253 0.6625868 ]
cross_val_score :  0.6698
Model Name :  Lasso
ACC :  [0.65039273 0.26323254 0.66576341 0.63035339 0.6148276  0.65827104
 0.7159767  0.67996525 0.70134589 0.56936442]
cross_val_score :  0.6149
Model Name :  LassoCV
ACC :  [0.67763474 0.29530252 0.68737796 0.65142282 0.63432239 0.6669316
 0.75945821 0.68587873 0.71730421 0.58902912]
cross_val_score :  0.6365
Model Name :  LassoLars
ACC :  [-6.81880414e-02 -2.76487950e-03 -1.44238932e-01 -5.88903445e-04
 -2.19968984e-02 -3.88829506e-03 -9.13236757e-02 -1.68612726e-02
 -2.62354458e-03 -4.17226033e-05]
cross_val_score :  -0.0353
Model Name :  LassoLarsCV
ACC :  [0.69481156 0.34532369 0.73069566 0.6947581  0.66937723 0.63607372
 0.84695599 0.68519409 0.74770911 0.66898623]
cross_val_score :  0.672
Model Name :  LassoLarsIC
ACC :  [0.69545592 0.33875614 0.72896455 0.69373668 0.669359   0.60415192
 0.84068249 0.67499448 0.74698141 0.67180396]
cross_val_score :  0.6665
Model Name :  LinearRegression
ACC :  [0.69545592 0.32752172 0.72840672 0.69451941 0.65977303 0.6330711
 0.85027748 0.68918687 0.74701467 0.67329687]
cross_val_score :  0.6699
Model Name :  LinearSVR
ACC :  [0.51281394 0.44118742 0.18818918 0.3566513  0.60644888 0.39511555
 0.20038612 0.40891676 0.50793505 0.4576382 ]
cross_val_score :  0.4075
Model Name :  MLPRegressor
ACC :  [0.50725168 0.27912353 0.54883545 0.60067898 0.68907557 0.58905417
 0.55237217 0.61285733 0.56769971 0.50613367]
cross_val_score :  0.5453
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
ACC :  [0.08637665 0.03565738 0.3490361  0.08938431 0.40320485 0.39054143
 0.15515341 0.22634598 0.23159783 0.11179314]
cross_val_score :  0.2079
Model Name :  OrthogonalMatchingPursuit
ACC :  [0.5045642  0.29590776 0.49805486 0.49098778 0.55381402 0.55871817
 0.4999458  0.56871372 0.61082628 0.47744117]
cross_val_score :  0.5059
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.71139633 0.34792073 0.66270276 0.65713481 0.6035213  0.58126602
 0.79964007 0.61688744 0.68552886 0.58959507]
cross_val_score :  0.6256
Model Name :  PLSCanonical
ACC :  [-0.87573964 -6.26008147 -2.12012053 -1.41219756 -4.00876569 -4.02797026
 -1.62843099 -1.13017908 -2.50207896 -1.97096621]
cross_val_score :  -2.5937
Model Name :  PLSRegression
ACC :  [0.66210632 0.4019656  0.75940161 0.67225152 0.65365267 0.51944944
 0.80708547 0.61657132 0.73723287 0.63523103]
cross_val_score :  0.6465
Model Name :  PassiveAggressiveRegressor
ACC :  [ 0.21062811 -0.98247856  0.05910537  0.22227882  0.21702236  0.05903212
 -0.1049126  -0.01134024  0.31036225  0.24594445]
cross_val_score :  0.0226
Model Name :  PoissonRegressor
ACC :  [0.72161894 0.50701335 0.66701809 0.74571127 0.7071869  0.70230873
 0.77425328 0.72677471 0.84520103 0.69951041]
cross_val_score :  0.7097
Model Name :  RANSACRegressor
ACC :  [ 0.63005361  0.07101631  0.7347886   0.41502244  0.4316665  -0.42956103
  0.60372281  0.56570532  0.61235285  0.46167173]
cross_val_score :  0.4096
Model Name :  RadiusNeighborsRegressor
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
Model Name :  RandomForestRegressor
ACC :  [0.88535628 0.84176152 0.82911101 0.72656216 0.82280765 0.76983617
 0.89518726 0.85213515 0.8801568  0.89070925]
cross_val_score :  0.8394
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.69863245 0.35882352 0.72841603 0.69087485 0.66275173 0.63253648
 0.83980795 0.68402025 0.74218273 0.65630234]
cross_val_score :  0.6694
Model Name :  RidgeCV
ACC :  [0.69630151 0.33432301 0.72875519 0.69420374 0.66114823 0.63368249
 0.84877954 0.68846035 0.74644534 0.67073522]
cross_val_score :  0.6703
Model Name :  SGDRegressor
ACC :  [-2.81194464e+26 -2.72932558e+26 -2.55104364e+25 -8.48897280e+26
 -3.33133982e+26 -7.20322885e+26 -4.26399693e+26 -1.43977876e+26
 -1.92412928e+25 -7.10092812e+26]
cross_val_score :  -3.781703278521882e+26
Model Name :  SVR
ACC :  [0.00524791 0.00774743 0.36912985 0.05545012 0.38376774 0.33103306
 0.1148391  0.21024132 0.19001052 0.07972074]
cross_val_score :  0.1747
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [0.59726287 0.37638345 0.74127513 0.63834895 0.77202845 0.4503236
 0.85537014 0.62462071 0.69091717 0.6008415 ]
cross_val_score :  0.6347
Model Name :  TransformedTargetRegressor
ACC :  [0.69545592 0.32752172 0.72840672 0.69451941 0.65977303 0.6330711
 0.85027748 0.68918687 0.74701467 0.67329687]
cross_val_score :  0.6699
Model Name :  TweedieRegressor
ACC :  [0.61857514 0.29533546 0.6374853  0.60208409 0.57902448 0.6365338
 0.71848152 0.63684089 0.72414203 0.59432855]
cross_val_score :  0.6043
VotingRegressor 은 실행되지 않는다.
'''