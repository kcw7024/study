#[실습]
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


# 2. 모델

# model = Sequential()
# model.add(Dense(100, input_dim=8, activation='linear'))
# model.add(Dense(200, activation='sigmoid'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(1, activation='sigmoid'))


input = Input(shape=(8,))
dense1 = Dense(100, activation='sigmoid')(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(200, activation='linear')(dense3)
dense5 = Dense(100, activation='sigmoid')(dense4)
output = Dense(1)(dense5)

model = Model(inputs = input, outputs = output)

model.summary()

import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
print(date) #2022-07-07 17:21:36.266674 : 현재시간

filepath = './_ModelCheckPoint/k25_12/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
                      )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1                 
                 )

end_time = time.time() - start_time


#4. 평가, 예측
print(('#'*70) + '1.기본출력')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)


'''
loss :  23.0333194732666
r2 score :  0.7180569176045343

'''