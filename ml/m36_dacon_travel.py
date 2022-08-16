from statistics import median
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import warnings
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv')

#print(train_set.shape, test_set.shape)    # (1955, 19) (2933, 18) (2933, 2)

#1-1. 결측치 처리

print(train_set.isnull().sum())

train_set['Age'] = train_set['Age'].fillna(median, inplace=True)
train_set['DurationOfPitch'] = train_set['DurationOfPitch'].fillna(median, inplace=True)
train_set['NumberOfFollowups'] = train_set['NumberOfFollowups'].fillna(median, inplace=True)
train_set['PreferredPropertyStar'] = train_set['PreferredPropertyStar'].fillna(median, inplace=True)
train_set['NumberOfTrips'] = train_set['NumberOfTrips'].fillna(median, inplace=True)
train_set['NumberOfChildrenVisiting'] = train_set['NumberOfChildrenVisiting'].fillna(median, inplace=True)
train_set['MonthlyIncome'] = train_set['MonthlyIncome'].fillna(median, inplace=True)

test_set['Age'] = test_set['Age'].fillna(median, inplace=True)
test_set['DurationOfPitch'] = test_set['DurationOfPitch'].fillna(median, inplace=True)
test_set['NumberOfFollowups'] = test_set['NumberOfFollowups'].fillna(median, inplace=True)
test_set['PreferredPropertyStar'] = test_set['PreferredPropertyStar'].fillna(median, inplace=True)
test_set['NumberOfTrips'] = test_set['NumberOfTrips'].fillna(median, inplace=True)
test_set['NumberOfChildrenVisiting'] = test_set['NumberOfChildrenVisiting'].fillna(median, inplace=True)
test_set['MonthlyIncome'] = test_set['MonthlyIncome'].fillna(median, inplace=True)

# print(train_set.isnull().sum())
# print(test_set.isnull().sum())

#1-2. Object type 라벨인코딩

le = LabelEncoder()
cols = train_set.columns
cols = np.array(cols)

for i in cols:
      if train_set[i].dtype == 'object':
        train_set[i] = le.fit_transform(train_set[i])
        test_set[i] = le.fit_transform(test_set[i])

# print(train_set.info())
# print(test_set.info())

x = train_set.drop(['ProdTaken'], axis=1)
y = train_set['ProdTaken']

x = np.array(x)
y = np.array(y)
y = y.reshape(-1, 1)

test_set = np.array(test_set)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
              'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],
              #'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              #'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              #'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              #'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              #'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10]
              } 

#2. 모델 

model = RandomizedSearchCV(
                           XGBClassifier(random_state=123), 
                           parameters, 
                           cv=kfold, 
                           n_jobs=8
                          )

model.fit(x_train, y_train, 
          early_stopping_rounds=10,
          eval_set=[(x_train, y_train), (x_test, y_test)]
          )

print("스코어 :: ", model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, model.predict(x_test))
print("결과 :: ", score)

# 스코어 ::  0.8618925831202046
# 결과 ::  0.8618925831202046

# 5. Submit
y_submit = model.predict(test_set)

submission = pd.read_csv(path+'submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(path + 'submission_4.csv', index = True)

# 스코어 ::  0.8567774936061381
# 결과 ::  0.8567774936061381


