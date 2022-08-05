import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping

encording_columns = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
                    'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
                    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
                    'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

non_encording_columns = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond',
                         'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
                         'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                         'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea',
                         'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                         'MiscVal','MoSold','YrSold']



#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임  3

###################### IQR 이용해서 train_set에서 이상치나온 행 삭제########################
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        
    return multiple_outliers
        
Outliers_to_drop = detect_outliers(train_set, 2, ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'])


train_set.loc[Outliers_to_drop]


train_set = train_set.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_set.shape

print(train_set)

#################긁어온거####################################

num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',
                   'FullBath','YearBuilt','YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

for col in cols_fillna : 
    train_set[col].fillna('None', inplace=True)
    test_set[col].fillna('None', inplace=True)

total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

train_set.fillna(train_set.mean(), inplace=True)
test_set.fillna(test_set.mean(), inplace=True)

total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum()) # 0 0 출력시 결측치 확인 끝

id_test = test_set['Id']

to_drop_num = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg

for df in [train_set, test_set] :
    df.drop(cols_to_drop, inplace=True, axis = 1)
    
# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 

# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']

for df in [train_set, test_set]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4 

train_set.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
test_set.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)


##############################긁어온거끝################################

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=99)

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)



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
ACC :  [0.84028226 0.86146095 0.83184795 0.88314804 0.86137563 0.81106507
 0.84895104 0.89717631 0.83575447 0.87802803]
cross_val_score :  0.8549
Model Name :  AdaBoostRegressor
ACC :  [0.81445018 0.82604539 0.81834549 0.81227412 0.8614389  0.75416794
 0.84118493 0.82330338 0.78772866 0.84547485]
cross_val_score :  0.8184
Model Name :  BaggingRegressor
ACC :  [0.85876674 0.89323001 0.8521955  0.87460853 0.86295364 0.86951203
 0.84303834 0.86275656 0.84213247 0.89399199]
cross_val_score :  0.8653
Model Name :  BayesianRidge
ACC :  [0.83975309 0.85981606 0.83200829 0.88449306 0.86207372 0.80874472
 0.85007428 0.89815326 0.83602174 0.87910909]
cross_val_score :  0.855
Model Name :  CCA
ACC :  [ 0.09511557 -0.15939798  0.33269468  0.17417454  0.01165556  0.04334574
  0.33389608 -0.07239382  0.30428667  0.22269497]
cross_val_score :  0.1286
Model Name :  DecisionTreeRegressor
ACC :  [0.72549154 0.59185575 0.76354849 0.79610919 0.55804612 0.82305246
 0.75453073 0.75121254 0.79230208 0.79625594]
cross_val_score :  0.7352
Model Name :  DummyRegressor
ACC :  [-1.66698683e-02 -4.01809786e-02 -2.83539607e-05 -6.00201611e-03
 -5.95794764e-02 -5.96056987e-03 -3.10499491e-02 -1.71888256e-05
 -2.33565907e-04 -5.87324509e-03]
cross_val_score :  -0.0166
Model Name :  ElasticNet
ACC :  [0.81151456 0.83247473 0.80369129 0.86841286 0.84936091 0.75645095
 0.8273276  0.89412079 0.82975792 0.86921228]
cross_val_score :  0.8342
Model Name :  ElasticNetCV
ACC :  [0.68216096 0.67921593 0.63989334 0.72390136 0.69006449 0.60018371
 0.63124518 0.77439361 0.73947247 0.74168845]
cross_val_score :  0.6902
Model Name :  ExtraTreeRegressor
ACC :  [0.76676972 0.7191421  0.73054869 0.7054958  0.81611257 0.69892106
 0.76735781 0.66805122 0.76898962 0.80553121]
cross_val_score :  0.7447
Model Name :  ExtraTreesRegressor
ACC :  [0.87404576 0.87901333 0.86381364 0.8549619  0.86621403 0.86866359
 0.84028898 0.84406627 0.84499338 0.89066132]
cross_val_score :  0.8627
Model Name :  GammaRegressor
ACC :  [-2.06926501e-02 -3.91855499e-02 -3.39838166e-05 -6.61532247e-03
 -5.77798579e-02 -6.81780721e-03 -3.66795071e-02 -1.69302290e-05
 -2.99005967e-04 -6.59276183e-03]
cross_val_score :  -0.0175
Model Name :  GaussianProcessRegressor
ACC :  [-6.71411336 -7.15711744 -5.68941237 -6.16659609 -6.10414444 -6.70735529
 -5.48398783 -6.18078998 -5.06158718 -5.41395633]
cross_val_score :  -6.0679
Model Name :  GradientBoostingRegressor
ACC :  [0.87964075 0.90260589 0.87095891 0.91574931 0.89277292 0.85682171
 0.8562897  0.88363956 0.89050004 0.89960273]
cross_val_score :  0.8849
Model Name :  HistGradientBoostingRegressor
ACC :  [0.88060089 0.89736864 0.83600294 0.90746642 0.87124638 0.85524457
 0.84055802 0.86235915 0.84627561 0.89169504]
cross_val_score :  0.8689
Model Name :  HuberRegressor
ACC :  [0.82989832 0.86200537 0.81180501 0.87037394 0.86809993 0.80560014
 0.83188172 0.89503838 0.82346251 0.86711285]
cross_val_score :  0.8465
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [0.69362784 0.71154289 0.71222531 0.78423062 0.81353511 0.74492432
 0.72864745 0.74295141 0.75573942 0.8147518 ]
cross_val_score :  0.7502
Model Name :  KernelRidge
ACC :  [0.84024561 0.8631651  0.83128233 0.88210853 0.86188282 0.81600972
 0.84749642 0.89549462 0.83497773 0.87715895]
cross_val_score :  0.855
Model Name :  Lars
ACC :  [0.84018198 0.85983315 0.83262145 0.88522178 0.86021316 0.80989248
 0.85020423 0.89738406 0.83573214 0.87849158]
cross_val_score :  0.855
Model Name :  LarsCV
ACC :  [0.84014219 0.85983315 0.83241368 0.88395587 0.86093788 0.81158745
 0.85011718 0.89767124 0.83556146 0.87837724]
cross_val_score :  0.8551
Model Name :  Lasso
ACC :  [0.84017865 0.85983516 0.83261419 0.88521157 0.86022734 0.8098813
 0.85020268 0.89739284 0.83573685 0.87849705]
cross_val_score :  0.855
Model Name :  LassoCV
ACC :  [0.74721126 0.76848477 0.74635931 0.83000127 0.78593243 0.65598862
 0.76251208 0.85631063 0.79236972 0.81161613]
cross_val_score :  0.7757
Model Name :  LassoLars
ACC :  [0.8401495  0.86000992 0.8325499  0.88505199 0.86058423 0.81001727
 0.85023082 0.89749796 0.83563126 0.87845497]
cross_val_score :  0.855
Model Name :  LassoLarsCV
ACC :  [0.84014219 0.85983315 0.83241368 0.88395587 0.86093788 0.80989248
 0.85011718 0.89767124 0.83556146 0.87837724]
cross_val_score :  0.8549
Model Name :  LassoLarsIC
ACC :  [0.83945731 0.86257597 0.83211591 0.88499719 0.86062638 0.81159228
 0.85023629 0.89810205 0.8347583  0.87822619]
cross_val_score :  0.8553
Model Name :  LinearRegression
ACC :  [0.84018198 0.85983315 0.83262145 0.88522178 0.86021316 0.80989248
 0.85020423 0.89738406 0.83573214 0.87849158]
cross_val_score :  0.855
Model Name :  LinearSVR
ACC :  [0.69529793 0.67756019 0.65656323 0.72655308 0.70587618 0.59140844
 0.65583311 0.78784925 0.74651659 0.73828069]
cross_val_score :  0.6982
Model Name :  MLPRegressor
ACC :  [0.42763257 0.41465155 0.38686182 0.43977669 0.39775555 0.40474622
 0.37530559 0.49489603 0.41934243 0.45935646]
cross_val_score :  0.422
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
ACC :  [-0.0551662  -0.00466613 -0.00982826 -0.03306848 -0.01377742 -0.00061136
 -0.07592445 -0.01538771 -0.0084421  -0.03000867]
cross_val_score :  -0.0247
Model Name :  OrthogonalMatchingPursuit
ACC :  [0.71000037 0.58498052 0.62407307 0.68789477 0.60726604 0.67171431
 0.6642465  0.699235   0.6535551  0.61762601]
cross_val_score :  0.6521
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.82705162 0.84468065 0.79073375 0.83299887 0.84701011 0.78134836
 0.82661086 0.88616099 0.81947864 0.86353002]
cross_val_score :  0.832
Model Name :  PLSCanonical
ACC :  [-2.19537296 -3.35337154 -1.5239517  -1.81445164 -2.74323025 -2.23746319
 -1.70068142 -2.57767327 -1.36988724 -1.53481614]
cross_val_score :  -2.1051
Model Name :  PLSRegression
ACC :  [0.82468519 0.85488935 0.81638384 0.86184729 0.87096383 0.80570215
 0.84535028 0.88728211 0.82929232 0.87125023]
cross_val_score :  0.8468
Model Name :  PassiveAggressiveRegressor
ACC :  [0.66508858 0.61227302 0.37698253 0.61418465 0.51037383 0.20237917
 0.46747558 0.34748576 0.74038311 0.74504362]
cross_val_score :  0.5282
Model Name :  PoissonRegressor
ACC :  [-1.91790316e-02 -4.10356640e-02 -3.22470164e-05 -6.56399173e-03
 -6.06870710e-02 -6.54786138e-03 -3.47583159e-02 -1.75978694e-05
 -2.78958191e-04 -6.48370813e-03]
cross_val_score :  -0.0176
Model Name :  RANSACRegressor
ACC :  [0.81415525 0.81205268 0.7983245  0.80718875 0.84210533 0.79386382
 0.80244991 0.85879283 0.78053295 0.80272341]
cross_val_score :  0.8112
Model Name :  RadiusNeighborsRegressor
ACC :  [-1.65174434e+28 -2.20011100e+28 -1.56935593e+28 -1.59610453e+28
 -1.99983671e+28 -1.95979895e+28 -1.28608832e+28 -1.66034660e+28
 -1.37153467e+28 -1.40656796e+28]
cross_val_score :  -1.670148899995296e+28
Model Name :  RandomForestRegressor
ACC :  [0.87186944 0.90408623 0.85393655 0.88113227 0.88809056 0.86450965
 0.85496164 0.87992838 0.84359682 0.89036623]
cross_val_score :  0.8732
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.84013733 0.85984793 0.83255355 0.88513663 0.86048696 0.80976764
 0.85019578 0.89749205 0.83577421 0.87858067]
cross_val_score :  0.855
Model Name :  RidgeCV
ACC :  [0.83965298 0.85978502 0.83187563 0.88439127 0.86248741 0.80852999
 0.85002212 0.89829371 0.83608979 0.87924304]
cross_val_score :  0.855
Model Name :  SGDRegressor
ACC :  [-9.39550543e+19 -1.03143653e+22 -5.41338758e+20 -8.75426903e+21
 -2.04261508e+20 -4.42731254e+21 -2.21071172e+21 -7.55053101e+21
 -9.65883534e+20 -4.13994481e+21]
cross_val_score :  -3.920257325280592e+21
Model Name :  SVR
ACC :  [-0.15384255 -0.00797605 -0.05007731 -0.10636882 -0.00047215 -0.03373285
 -0.14153026 -0.05589659 -0.0417904  -0.08296983]
cross_val_score :  -0.0675
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [0.83225247 0.85121682 0.78788953 0.8323546  0.83931928 0.73952417
 0.8142127  0.88312246 0.81915023 0.86288875]
cross_val_score :  0.8262
Model Name :  TransformedTargetRegressor
ACC :  [0.84018198 0.85983315 0.83262145 0.88522178 0.86021316 0.80989248
 0.85020423 0.89738406 0.83573214 0.87849158]
cross_val_score :  0.855
Model Name :  TweedieRegressor
ACC :  [0.79857964 0.81506851 0.76152673 0.82956446 0.82499868 0.74760321
 0.78328724 0.8795919  0.8277281  0.84888829]
cross_val_score :  0.8117
VotingRegressor 은 실행되지 않는다.

'''