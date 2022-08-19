import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#1. 데이터
path = './_data/travel/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv')

#1-1. 결측치 처리 (평균값으로 채워준다)
mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome','DurationOfPitch']
for col in mean_cols:
    train_set[col] = train_set[col].fillna(train_set[col].mean())

mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome','DurationOfPitch']
for col in mean_cols:
    test_set[col] = test_set[col].fillna(test_set[col].mean())

#1-2. 문자형 결측치 처리 (Unknwon으로 채워준다)    
train_set.TypeofContact = train_set.TypeofContact.fillna("Unknown")  
print(train_set.isnull().sum())
#확인 완료
#object type 컬럼 확인하기.
object_columns = train_set.columns[train_set.dtypes == 'object']
print('list : ', list(object_columns)) 
# list :  ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

test_set.TypeofContact = test_set.TypeofContact.fillna("Unknown")  
print(test_set.isnull().sum())
#확인 완료
#object type 컬럼 확인하기.
object_columns = test_set.columns[test_set.dtypes == 'object']
print('list : ', list(object_columns)) 
# list :  ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

train_set = train_set.copy()
for o_col in object_columns:
  encoder = LabelEncoder()
  encoder.fit(train_set[o_col])
  train_set[o_col] = encoder.transform(train_set[o_col])

print(train_set)

test_set = test_set.copy()

for o_col in object_columns:
  encoder = LabelEncoder()
  encoder.fit(test_set[o_col])
  test_set[o_col] = encoder.transform(test_set[o_col])

print(test_set)


scaler = MinMaxScaler()

train_set = train_set.copy()
test_set = test_set.copy()
scaler.fit(train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(train_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
scaler.fit(test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test_set[['Age', 'DurationOfPitch', 'MonthlyIncome']])
print(train_set)
print(test_set)

x = train_set.drop(['ProdTaken'], axis=1)
y = train_set['ProdTaken']


x_train, x_test, y_train, y_test = train_test_split(
  x, y, train_size=0.8, random_state=123
)


#2. 모델
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

#3. 훈련
from sklearn.metrics import accuracy_score

model.fit(x_train, y_train)

#4. 평가, 예측
score = accuracy_score(y_test, model.predict(x_test))
print("결과 : ", score)

#5. 제출
y_submit = model.predict(test_set)
submission = pd.read_csv(path+'submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(path + 'submission_6.csv', index = True)

# 결과 :  0.8388746803069054