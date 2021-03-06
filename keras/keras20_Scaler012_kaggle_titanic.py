#[실습]
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#datasets.descibe()
#datasets.info()
#datasets.isnull().sum()

#pandas의 y라벨의 종류가 무엇인지 확인하는 함수 쓸것
#numpy에서는 np.unique(y, return_counts=True) 


#데이터 경로 정의

path = './_data/kaggle_titanic/'

train_set = pd.read_csv(path + 'train.csv', index_col=0)
#print(train_set.shape) #(891, 12)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
#print(test_set.shape) #(418, 11)


#1. 데이터

#print(train_set.Pclass.value_counts())
# 3    491
# 1    216
# 2    184
# Name: Pclass, dtype: int64

#각 등급별의 생존 비율을 확인
Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize=True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
#print(f"Percentage of Pclass 1 who survived: {Pclass1}")
#print(f"Percentage of Pclass 2 who survived: {Pclass2}")
#print(f"Percentage of Pclass 3 who survived: {Pclass3}")
# 결과값
# Percentage of Pclass 1 who survived: 62.96296296296296
# Percentage of Pclass 2 who survived: 47.28260869565217
# Percentage of Pclass 3 who survived: 24.236252545824847

#여자와 남자의 생존 비율을 확인
female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
#print(f"Percentage of females who survived: {female}")
#print(f"Percentage of males who survived: {male}")
# Percentage of females who survived: 74.20382165605095
# Percentage of males who survived: 18.890814558058924

# 결측치 처리
# 각 컬럼의 결측치를 처리해준다
train_set = train_set.fillna({"Embarked": "S"}) # S 값을 넣어준다
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean()) # 평균값으로 넣어준다

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True) # pandas의 원핫인코딩사용
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean()) # 평균값으로 넣어준다
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape) #(891, 10) (418, 8)


# x와 y 변수 지정

x = train_set.drop(['Survived'], axis=1)  
#print(x)
#print(x.columns)
#print(x.shape) # (891, 8)
y = train_set['Survived'] 
#print(y)
#print(y.shape) # (891,)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=8, activation='linear'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
              #loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
              optimizer='adam', 
              metrics=['accuracy'] 
              )

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=64,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )


# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#print(y_predict)
y_predict = y_predict.round(0) #predict 반올림처리 소수점처리 
#print(y_predict)


y_summit = model.predict(test_set)

#print(y_summit)
#print(y_summit.shape) # (418, 1)
y_summit = y_summit.round()
df = pd.DataFrame(y_summit) 
#print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y_summit = oh.fit_transform(df)
#print(y_summit)
y_summit = np.argmax(y_summit, axis= 1)

#submission_set = pd.read_csv(path + 'gender_submission.csv', index_col=0)

#print(submission_set)

#submission_set['Survived'] = y_summit
#print(submission_set)


#submission_set.to_csv(path + 'gender_submission.csv', index = True)


acc = accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 

#model.summary()


'''

1. 스케일러 하기전

loss :  [0.40614479780197144, 0.8435754179954529]
acc스코어 :  0.8435754189944135

2. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  [0.4298607110977173, 0.8156424760818481]
acc스코어 :  0.8156424581005587

3. Standard Scaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 처곧를 보장할 수 없다.)

loss :  [0.5796253085136414, 0.8100558519363403]
acc스코어 :  0.8100558659217877

4. MaxAbsSacler (절대값이 0~1 사이에 매핑되도록 하는 것. 양수데이터로만 구성된 특징 
데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)

loss :  [0.47396472096443176, 0.7821229100227356]
acc스코어 :  0.7821229050279329

5. RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
* IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.

loss :  [0.5629486441612244, 0.8044692873954773]
acc스코어 :  0.8044692737430168


'''
