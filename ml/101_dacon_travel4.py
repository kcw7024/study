from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from time import time
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
# 1. 데이터
path = './_data/travel/'  # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
# print(train_set.info())
test_set = pd.read_csv(path + 'test.csv',  # 예측에서 쓸거야!!
                       index_col=0)
# print(train_set.shape,test_set.shape) (1955, 19) (2933, 18)
# Data columns (total 19 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Age                       1861 non-null   float64  MonthlyIncome
#  1   TypeofContact             1945 non-null   object # 빈도로 메꾸기
#  2   CityTier                  1955 non-null   int64
#  3   DurationOfPitch           1853 non-null   float64 앞뒤행으로
#  4   Occupation                1955 non-null   object
#  5   Gender                    1955 non-null   object
#  6   NumberOfPersonVisiting    1955 non-null   int64
#  7   NumberOfFollowups         1942 non-null   float64
#  8   ProductPitched            1955 non-null   object
#  9   PreferredPropertyStar     1945 non-null   float64
#  10  MaritalStatus             1955 non-null   object
#  11  NumberOfTrips             1898 non-null   float64
#  12  Passport                  1955 non-null   int64
#  13  PitchSatisfactionScore    1955 non-null   int64
#  14  OwnCar                    1955 non-null   int64
#  15  NumberOfChildrenVisiting  1928 non-null   float64
#  16  Designation               1955 non-null   object
#  17  MonthlyIncome             1855 non-null   float64 직급?나이?
#  18  ProdTaken                 1955 non-null   int64
# 결측치 채우기 (클래스별 괴리가 큰 컬럼으로 평균 채우기)

# train_set = train_set.drop([190,605,1339],inplace=True)
# test_set = test_set.drop(index3,index=True)

train_set['Gender'] = train_set['Gender'].str.replace('Fe Male', 'Female')
test_set['Gender'] = test_set['Gender'].str.replace('Fe Male', 'Female')

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')[
                        'Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')[
                       'Age'].transform('mean'), inplace=True)
train_set['Age'] = np.round(train_set['Age'], 0).astype(int)
test_set['Age'] = np.round(test_set['Age'], 0).astype(int)

# print(train_set.isnull().sum()) #(1955, 19)
print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
# print(train_set.describe) #(1955, 19)
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
train_set['DurationOfPitch'] = train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch'] = test_set['DurationOfPitch'].fillna(0)

# print(train_set[train_set['NumberOfFollowups'].notnull()].groupby(['NumberOfChildrenVisiting'])['NumberOfFollowups'].mean())
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')[
                                      'NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')[
                                     'NumberOfFollowups'].transform('mean'), inplace=True)
# print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['Occupation'])['PreferredPropertyStar'].mean())
train_set['PreferredPropertyStar'].fillna(train_set.groupby(
    'Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby(
    'Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
# train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
# print(train_set['AgeBand'])
# [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# (43.8, 52.4] < (52.4, 61.0]]
combine = [train_set,test_set]
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4
# train_set = train_set.drop(['AgeBand'], axis=1)
# print(train_set[train_set['NumberOfTrips'].notnull()].groupby(['DurationOfPitch'])['PreferredPropertyStar'].mean())
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')[
                                  'NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')[
                                 'NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())
train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby(
    'MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby(
    'MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
# print(train_set.isnull().sum())
# print("================")
# print(test_set.isnull().sum())
train_set.loc[train_set['Gender'] == 'Fe Male', 'Gender'] = 'Female'
test_set.loc[test_set['Gender'] == 'Fe Male', 'Gender'] = 'Female'
cols = ['TypeofContact', 'Occupation', 'Gender',
        'ProductPitched', 'MaritalStatus', 'Designation']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col] = le.fit_transform(train_set[col])
    test_set[col] = le.fit_transform(test_set[col])
# print(train_set['TypeofContact'])


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])  # percentile 백분위
    print("1사분위 : ", quartile_1)  # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ", q2)  # 50% median과 동일
    print("3사분위 : ", quartile_3)  # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr = quartile_3-quartile_1  # 75% -25%
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) |
                    (data_out < lower_bound))


# Age_out_index= outliers(train_set['Age'])[0]
TypeofContact_out_index = outliers(train_set['TypeofContact'])[0]
CityTier_out_index = outliers(train_set['CityTier'])[0]
DurationOfPitch_out_index = outliers(train_set['DurationOfPitch'])[0]
Gender_out_index = outliers(train_set['Gender'])[0]
NumberOfPersonVisiting_out_index = outliers(
    train_set['NumberOfPersonVisiting'])[0]
NumberOfFollowups_out_index = outliers(train_set['NumberOfFollowups'])[0]
ProductPitched_index = outliers(train_set['ProductPitched'])[0]
PreferredPropertyStar_out_index = outliers(
    train_set['PreferredPropertyStar'])[0]
MaritalStatus_out_index = outliers(train_set['MaritalStatus'])[0]
NumberOfTrips_out_index = outliers(train_set['NumberOfTrips'])[0]
Passport_out_index = outliers(train_set['Passport'])[0]
PitchSatisfactionScore_out_index = outliers(
    train_set['PitchSatisfactionScore'])[0]
OwnCar_out_index = outliers(train_set['OwnCar'])[0]
NumberOfChildrenVisiting_out_index = outliers(
    train_set['NumberOfChildrenVisiting'])[0]
Designation_out_index = outliers(train_set['Designation'])[0]
MonthlyIncome_out_index = outliers(train_set['MonthlyIncome'])[0]

lead_outlier_index = np.concatenate((  # Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                    DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                    ), axis=None)

print(len(lead_outlier_index))  # 577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index:
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)

'''
id : 샘플 아이디
Age : 나이
TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급) 
DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
Occupation : 직업
Gender : 성별
NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
ProductPitched : 영업 사원이 제시한 상품
PreferredPropertyStar : 선호 호텔 숙박업소 등급
MaritalStatus : 결혼여부
NumberOfTrips : 평균 연간 여행 횟수
Passport : 여권 보유 여부 (0: 없음, 1: 있음)
PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
Designation : (직업의) 직급
MonthlyIncome : 월 급여
ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)
'''

x = train_set_clean.drop(['ProdTaken', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
                           'OwnCar','MonthlyIncome', 'NumberOfFollowups'], axis=1)  # 'NumberOfTrips',
# x = train_set_clean.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting',
                          'OwnCar','MonthlyIncome', 'NumberOfFollowups'], axis=1)  # 'NumberOfTrips',
y = train_set_clean['ProdTaken']
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
bayesian_params = {
    'learning_rate' : (0.01, 10),
    'max_depth' : (6, 16),
    # 'max_leaves' : (24, 64),
    'one_hot_max_size' : (10, 200),
    'subsample' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    'border_count' : (10, 500),
    'l2_leaf_reg' : (0.001, 10),
    'od_type' : (0.01, 50),
    'od_pval' : (0.01, 50),
    'model_size_reg' : (0.01, 50),
}


# "learning_rate": [0.20909079092170735],
# 'depth': [8],
# 'od_pval': [0.236844398775451],
# 'model_size_reg': [0.30614059763442997],
# 'l2_leaf_reg': [5.535171839105427]


def cat_hamsu(learning_rate, max_depth,
            one_hot_max_size, 
            subsample,colsample_bylevel, border_count, 
            l2_leaf_reg, od_type,od_pval,
            model_size_reg) :
    
    params = {
        'n_estimators' : 500, 
        'learning_rate': 0.02,
        'max_depth' : int(round(max_depth)), # 무조건 정수형
        # 'max_leaves' : int(round(max_leaves)),
        'subsample' : max(min(subsample,1),0), # 0~1사이의 값
        'one_hot_max_size' : max(min(one_hot_max_size,1),0),
        'colsample_bylevel' : max(min(colsample_bylevel,1),0),
        'border_count' : max(min(border_count,1),0),
        'l2_leaf_reg' : max(min(l2_leaf_reg,1),0),
        'od_type' : max(min(od_type,1),0),
        'od_pval' : max(min(od_pval,1),0),
        'model_size_reg' : max(min(model_size_reg,1),0),
    }

    #  * :: 여려개의 인자를 받겠다
    # ** :: 키워드 받겠다(딕셔너리형태)

    model = CatBoostClassifier(**params)

    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test, y_test)],
              #eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)

    return results

cat_bo = BayesianOptimization(f=cat_hamsu,
                              pbounds=bayesian_params,
                              random_state=123
                              )

cat_bo.maximize(init_points=5, n_iter=50)

print(cat_bo.max)

'''
{'target': 0.9060052219321149, 
'params': {'border_count': 349.08501863083245, 
'colsample_bylevel': 0.8910430434208152, 
'l2_leaf_reg': 7.257150375665365, 
'learning_rate': 8.568833865283933,
'max_depth': 11.617215964627862,
'model_size_reg': 15.195634438598251,
'od_pval': 41.992138779304966,
'od_type': 37.421178963755835, 
'one_hot_max_size': 101.50153950289088, 
'subsample': 0.9403029928332964}}


# model = CatBoostClassifier(
#     n_estimators=500,
#     learning_rate=0.02,
#     colsample_bylevel=[max(min(0.9663807206030328, 1), 0)],
#     colsample_bytree=max(min(0.7402837722085994, 1), 0),
#     max_bin=max(int(round(176.5547753662509)), 100),
#     max_depth=int(round(15.620931151329858)),
#     min_child_weight=int(round(1.3470658368579622)),
#     reg_alpha=max(3.0664881741733976, 0),
#     reg_lambda=max(9.426306524716228, 0),
#     subsample=max(min(0.6927562163656839, 1), 0)
# )


# #3. 훈련
# import time
# start = time.time()
# model.fit(x_train, y_train)
# end = time.time()

# #4. 평가, 예측
# y_predict = model.predict(x_test)
# print("accuracy_score :" , accuracy_score(y_test, y_predict))

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
# print("걸린시간 : ", round(end-start, 2))



# accuracy_score : 0.9649122807017544
# 최적 튠 ACC :  0.9649122807017544
# 걸린시간 :  3.24