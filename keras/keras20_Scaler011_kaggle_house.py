#kaggle 집값~ 문제풀이!!123123122
#https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import numpy as np
import datetime as dt
import pandas as pd
from collections import Counter
import datetime as dt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#데이터 경로 정의

path = './_data/kaggle_house/'  # 경로 정의
train_set = pd.read_csv(path + 'train.csv')
#print(train_set)
#print(train_set.shape) #(1460, 81)
test_set = pd.read_csv(path + 'test.csv')

#print(test_set)
#print(test_set.shape) #(1459, 80)


#1. 데이터 가공 시작
#아마도?이상치제거, 결측치제거
#타켓은 Saleprice

#1-1. 데이터 확인

#수치형변수와 범주형변수를 확인해본다(단순확인, 연산아님)
#numerical_feats = train_set.dtypes[train_set.dtypes != "object"].index
#print("수치형 features : ", len(numerical_feats)) # total : 38
#categorical_feats = train_set.dtypes[train_set.dtypes == "object"].index
#print("범주형 features : ", len(categorical_feats)) # total : 43

#확인
#print(train_set[numerical_feats].columns)
#print(train_set[categorical_feats].columns)

##### 정상 확인 #####

#1-1. 이상치 탐색 및 제거 (유효한수치들 외의 이상치들을 처리해주는 과정 : 원래시각화하면서 분류할 데이터를 구분하던디 안해도되서 안함)
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)  # 백분위율 계산
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step)
                              | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


Outliers_to_drop = detect_outliers(train_set, 2, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                                                  'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                                                  'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                                                  'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                                                  'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                                                  'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                                                  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                                                  'MiscVal', 'MoSold', 'YrSold'])

train_set.loc[Outliers_to_drop]


train_set = train_set.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

#print(train_set.shape) # (1338, 81) : 수치형 features에서 122개의 이상치가 제거됨

#1-2. 범주형 feature 작업

# 각범주형의 데이터타입을 확인해보자(걍 단순 확인)
#for catg in list(categorical_feats) :
#    print(train_set[catg].value_counts())
#    print('-'*30)

#시각화 작업을 하면서 데이터끼리의 상관관계 및 밀접한 데이터를 확인해서 그걸 기반으로
#타켓 컬럼인 saleprice 에 영향을 주는컬럼들을 정리하는듯한데 일단 당장 할필요없으니 안해 걍 알고있기
#어쨌든 시각화 작업 후에 결과로는 SalePrice에 영향을 많이 끼친 변수는
#'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual', 'CentralAir', 'Electrical', 'KitchenQual', 'SaleType'

#1-3. Saleprice와 관련이 큰 변수와 아닌변수를 분리한다.

num_strong_corr = ['SalePrice', 'OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageCars',
                   'FullBath', 'YearBuilt', 'YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual', 'CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle',
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation',
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                  'SaleCondition']

#1-4. 결측치 처리

#해당 컬럼에 있는 결측치를 None으로 처리해준다
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
               'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2',
               'MSZoning', 'Utilities']

for col in cols_fillna:
    train_set[col].fillna('None', inplace=True)
    test_set[col].fillna('None', inplace=True)

#None으로 처리된 결측치 처리정도를 확인(확인용~~~~~)
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(5))

#여기까지하면 약 (330, 300) 정도의 결측치가 남음.
#남아있는 결측치들은 수치형 변수라서 평균값으로 작업
train_set.fillna(train_set.mean(), inplace=True)
test_set.fillna(test_set.mean(), inplace=True)

#확인한다.
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head(5)

#print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum()) - (0,0)으로 결측치 완전 처리 끝!!!!!!!!!!!!!


#1-5. 유의하지 않다고 판단되는 변수를 삭제.
#SalePrice와 상관관계가 약한변수를 모두 삭제
id_test = test_set['Id']

to_drop_num = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg

for df in [train_set, test_set]:
    df.drop(cols_to_drop, inplace=True, axis=1)

#확인. 잘찍힘
#print(train_set.head())

#1-6.변수들의 범주들을 그룹화한다.

# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV']

# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor',
              'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN']

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']


#1-6_2. 각범주별 수치형 변환 실행
for df in [train_set, test_set]:

    df['MSZ_num'] = 1
    df.loc[(df['MSZoning'].isin(msz_catg2)), 'MSZ_num'] = 2
    df.loc[(df['MSZoning'].isin(msz_catg3)), 'MSZ_num'] = 3

    df['NbHd_num'] = 1
    df.loc[(df['Neighborhood'].isin(nbhd_catg2)), 'NbHd_num'] = 2
    df.loc[(df['Neighborhood'].isin(nbhd_catg3)), 'NbHd_num'] = 3

    df['Cond2_num'] = 1
    df.loc[(df['Condition2'].isin(cond2_catg2)), 'Cond2_num'] = 2
    df.loc[(df['Condition2'].isin(cond2_catg3)), 'Cond2_num'] = 3

    df['Mas_num'] = 1
    df.loc[(df['MasVnrType'] == 'Stone'), 'Mas_num'] = 2

    df['ExtQ_num'] = 1
    df.loc[(df['ExterQual'] == 'TA'), 'ExtQ_num'] = 2
    df.loc[(df['ExterQual'] == 'Gd'), 'ExtQ_num'] = 3
    df.loc[(df['ExterQual'] == 'Ex'), 'ExtQ_num'] = 4

    df['BsQ_num'] = 1
    df.loc[(df['BsmtQual'] == 'Gd'), 'BsQ_num'] = 2
    df.loc[(df['BsmtQual'] == 'Ex'), 'BsQ_num'] = 3

    df['CA_num'] = 0
    df.loc[(df['CentralAir'] == 'Y'), 'CA_num'] = 1

    df['Elc_num'] = 1
    df.loc[(df['Electrical'] == 'SBrkr'), 'Elc_num'] = 2

    df['KiQ_num'] = 1
    df.loc[(df['KitchenQual'] == 'TA'), 'KiQ_num'] = 2
    df.loc[(df['KitchenQual'] == 'Gd'), 'KiQ_num'] = 3
    df.loc[(df['KitchenQual'] == 'Ex'), 'KiQ_num'] = 4

    df['SlTy_num'] = 2
    df.loc[(df['SaleType'].isin(SlTy_catg1)), 'SlTy_num'] = 1
    df.loc[(df['SaleType'].isin(SlTy_catg3)), 'SlTy_num'] = 3
    df.loc[(df['SaleType'].isin(SlTy_catg4)), 'SlTy_num'] = 4

#1-7.기존 범주형 변수와 새로 만들어진 수치형 변수 역시 유의하지않은것은 삭제
train_set.drop(['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual', 'CentralAir', 'Electrical',
               'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis=1, inplace=True)
test_set.drop(['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual', 'CentralAir', 'Electrical',
              'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis=1, inplace=True)


#1-8. x와 y 를 정의해준다
x = train_set.drop('SalePrice', axis=1)
#print(x)
#print(x.columns)
#print(x.shape) #(1338, 12)

y = train_set['SalePrice']
#print(y)
#print(y.shape) #(1338, )


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.99, random_state=777
)


#scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=12))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # test로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)

#RMSE 함수정의, 사용


def RMSE(y_test, y_predict):  # mse에 루트를 씌운다.
    return np.sqrt(mean_squared_error(y_test, y_predict))


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



#5.csv로 내보낸다
#result = pd.read_csv(path + 'sample_submission.csv', index_col=0)
#index_col=0 의 의미 : index col을 없애준다.

#y_summit = model.predict(test_set)
#print(y_summit)
#print(y_summit.shape)  # (1459, 1)


#result['SalePrice'] = y_summit

#result 에서 지정해준 submission의 count 값에 y_summit값을 넣어준다.

#.to_csv() 를 사용해서 sampleSubmission.csv를 완성

#2
#result = abs(result) #절대값처리.... 인데 이걸로하면 안되는디
#result.to_csv(path + 'sample_submission.csv', index=True)



'''

1.기존대로 작업했을때 결과값

loss :  [1475942144.0, 1475942144.0]
RMSE :  38417.99043999383
r2스코어 :  0.16461147205704563

2.MinMaxScarler

loss :  [315371744.0, 315371744.0]
RMSE :  17758.70981587976
r2스코어 :  0.8214984300254002

2.StandardScaler

loss :  [456983040.0, 456983040.0]
RMSE :  21377.161472017877
r2스코어 :  0.741345943384845

'''