# 실습
from logging import warning
from msilib.schema import Binary
from tabnanny import verbose
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from collections import Counter

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

#print(train_set.Pclass.value_counts())


# print(train_set.isnull().sum()) #결측치를 전부 더한다
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# imputer = KNNImputer()
# imputer.fit(train_set)
# data2 = imputer.transform(train_set)
# print(data2)
# print(train_set.isnull().sum()) # 없어졌는지 재확인

#print(train_set.columns)



#train_set = pd.DataFrame(train_set)



#=================컬럼별로 이상치 확인하고 제거해주기===============

def detect_outliers(df,n,features):
    outlier_indices = []  
    for col in features: 
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5*IQR  
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)           
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k , v in outlier_indices.items() if v > n)
        
    return multiple_outliers
    
Outliers_to_drop = detect_outliers(train_set,2,["Age",'SibSp','Parch','Fare'])

print(train_set.loc[Outliers_to_drop])
train_set.loc[Outliers_to_drop]
print(train_set.loc[Outliers_to_drop])
print(train_set.shape)
train_set = train_set.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
print(train_set.shape)
print(train_set)


Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)

# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

print(train_set.shape)



#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x = np.array(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2. 모델

xg = XGBClassifier(n_estimators = 100,
                  learning_rate = 0.1,
                  max_depth = 3,
                  gamma = 1,
                  min_child_weight = 1,
                  subsample = 1,
                  colsample_bytree = 0.5,
                  colsample_bylevel = 1,
                  colsample_bynode = 1,
                  reg_alpha = 0,
                  reg_lambda = 0)

lg = LGBMClassifier(num_iterations = 100,
                    learning_rate= 0.1,
                    max_depth=1,
                    min_data_in_leaf = 20,
                    num_leaves=31,
                    boosting = 'gbdt',
                    bagging_fraction = 1.0,
                    feature_fraction = 1.0,
                    lambda_l1 = 0.0                                        
)

cat = CatBoostClassifier(
                    max_depth=10,
                    learning_rate=0.01,
                    n_estimators=100,
                    eval_metric='Accuracy',
                    loss_function='MultiClass',
                    verbose=False)

model = VotingClassifier(  
        estimators = [('XG', xg), ('LG', lg), ('CAT', cat)],
        voting = 'soft'  # hard 도 있다.
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
# print("보팅 결과 : ", round(score, 4))

# 보팅 결과 :  0.9912

classifiers = [cat, xg, lg]

for model2 in classifiers :
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2))

print("보팅 결과 : ", round(score, 4))

'''
CatBoostClassifier 정확도 : 1.0000
XGBClassifier 정확도 : 0.9825
LGBMClassifier 정확도 : 0.9825
보팅 결과 :  1.0


파라미터 튜닝 후

CatBoostClassifier 정확도 : 0.8023
XGBClassifier 정확도 : 0.8136
LGBMClassifier 정확도 : 0.7797
보팅 결과 :  0.8023

'''

