# logistic regression :: 논리회귀 , 이진분류에만 사용!!!! regression + sigmoid

from calendar import EPOCH
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes, load_boston
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter

import numpy as np
import pandas as pd

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0


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



x = torch.Tensor(x.values)
y = torch.Tensor(y.values)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=123)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())
print(x_train.shape)

# DateLoader 시작
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x, y 를 합쳐준다.
test_set = TensorDataset(x_test, y_test) 

# print(train_set) # <torch.utils.data.dataset.TensorDataset object at 0x000002A121CDBF70>
# print("="*30, "train_set[0]") 
# print(train_set[0])
# print("="*30, "train_set[0][0]")
# print(train_set[0][0])
# print("="*30, "train_set[0][1]")
# print(len(train_set[0][1]))
# print("="*30, "train_set 총 갯수")
# print(len(train_set)) # 398

# x, y  배치 결합
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

# print(train_loader) # <torch.utils.data.dataloader.DataLoader object at 0x000002B437F0E910>
# print("="*30, "train_loader[0]") 
# print(train_loader[0]) # ERROR
# print("="*30, "train_loader[0][0]")
# print(train_loader[0][0])
# print("="*30, "train_loader[0][1]")
# print(len(train_loader[0][1]))
# print("="*30, "train_loader 총 갯수")
# print(len(train_loader)) # 398


# model class 화 
# class ()안에는 상위클래스만 넣을수있음
class Model(nn.Module) :
    def __init__(self, input_dim, output_dim) :
        # super().__init__() # super 사용시 불러온 모듈의 함수와 변수 모두 사용하겠다
        super(Model, self).__init__() # 위와같음
        
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_size): 
        x = self.linear1(input_size)                
        x = self.relu(x)                
        x = self.linear2(x)
        x = self.relu(x)                                
        x = self.linear3(x)
        x = self.relu(x)                                
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x 
    
    
model = Model(12, 1).to(DEVICE)

#3. 컴파일, 훈련

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



def train(model, criterion, optimizer, loader):
    # model.train()

    total_loss = 0
    
    for x_batch, y_batch in loader :
        optimizer.zero_grad()
        hypothesis = model(x_batch) # batch 단위대로 들어감
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # 회당 나오는 loss를 더해줌(누적)
        
    return total_loss / len(loader)

EPOCHS = 200

for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch %  10 == 0 : 
        print('epoch : {}, loss :{}'.format(epoch, loss)) 
    
       
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0

    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
        
    return total_loss
    
loss = evaluate(model, criterion, test_loader)
print('최종 LOSS : ', loss)

y_pred = model(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
print('R2 : ', score)


'''
최종 LOSS :  1434670548992.0
R2 :  -6.208534606438516

'''



