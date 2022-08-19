from sklearn.datasets import load_boston, load_iris, load_breast_cancer, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from collections import Counter

from sklearn.pipeline import make_pipeline

#1. 데이터


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
    x, y, train_size=0.8, random_state=1234    
)

Kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델

model = make_pipeline(StandardScaler(), 
                      LogisticRegression()
                      )

model.fit(x_train, y_train)

print("기본 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=Kfold, scoring='accuracy')
print("기본 CV : ", np.delete(scores, 1, axis =0))
print("기본 CV 나눈 값 : ", np.mean(scores))


########################################### PolynomialFeature 후

pf = PolynomialFeatures(degree=2, include_bias=False)
# pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)
print(xp.shape) # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, train_size=0.8, random_state=1234    
)

#2. 모델

model = make_pipeline(StandardScaler(), 
                      LogisticRegression()
                      )

model.fit(x_train, y_train)

print("폴리 스코어 : ", model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=Kfold, scoring='accuracy')
print("폴리 CV : ", scores)
print("폴리 CV 나눈 값 : ", np.mean(scores))

'''
(881,)
기본 스코어 :  0.7796610169491526
기본 CV :  [0.81560284 0.78014184 0.82978723 0.81428571]
기본 CV 나눈 값 :  0.7997365754812563

(881, 44)
폴리 스코어 :  0.7853107344632768
폴리 CV :  [0.82978723 0.82269504 0.82978723 0.85106383 0.80714286]
폴리 CV 나눈 값 :  0.8280952380952382

'''