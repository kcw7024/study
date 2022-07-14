#kaggle bike 문제풀이!!
#https://www.kaggle.com/competitions/bike-sharing-demand/data


from pickletools import optimize
import numpy as np
import seaborn as sns #데이터 시각화할때 사용. 여기선 사용안했음 
import datetime as dt
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터

path = './_data/bike_sharing/' #경로 정의
train_set = pd.read_csv(path + 'train.csv', # 훈련함수를 정의, 데이터를 삽입
                        #index_col=0
                        ) 
#print(train_set)
#print(train_set.shape) #(10886, 11)

test_set = pd.read_csv(path + 'test.csv', #예측에서 사용
                       #index_col=0
                       )

#print(test_set)
#print(test_set.shape) #(6493, 9)

#train_set = train_set.dropna()
#test_set = test_set.fillna(0)


#1. 데이터 가공 시작
#1-1. datetime 을 object에서 datetime 자료형으로 변환한다
train_set['datetime'] = pd.to_datetime(train_set['datetime'])

#1-2. 자료형으로 변환한datetime을 년/월/일/시간으로 나누어 컬럼을 생성해준다
train_set['year'] = train_set['datetime'].dt.year
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour

#1-3. 가공전의datetime과 day, year을 삭제해준다 
# 숫자형으로 나타나는 컬럼을 제외하고 나머지는 제거한다.
train_set.drop(['datetime','day','year'], inplace=True, axis=1)
# month, hour 은 범주형*으로 변경
# 데이터프레임타입을 맞추기위해 month와 hour타입을 변경해준다
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')

#1-4. season과 weather를 더미로만든 가변수로 변환해준다.
#One-Hot Encoding 작업. pandas의 get_dummies를 사용하여 가변수 부여
train_set = pd.get_dummies(train_set, columns=['season', 'weather'])

#1-5. test_set과 컬럼수를 맞추기 위해 test_set에 없는 컬럼은 삭제
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
#1-5_2.temp와 atemp의 상관관계가 높고, 의미가 비슷하기때문에 temp만 사용,
#atemp를 drop, 제거해준다.
train_set.drop('atemp', inplace=True, axis=1)

#test_set 셋팅(train_set과 동일하게 진행해준다.)
test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour

test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')

test_set = pd.get_dummies(test_set, columns=['season', 'weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

#print(test_set)



x = train_set.drop(['count'], axis=1) #count라는 컬럼을 drop한다.
#print(x)
#print(x.columns)
print(x.shape) #(10886, 15)

y = train_set['count']
#print(y)
#print(y.shape) #(10886, 16)




x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.99, random_state=777
    )



scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

print(x_train.shape, x_test.shape) # (10777, 15) (109, 15)

x_train = x_train.reshape(10777, 15, 1)
x_test = x_test.reshape(109, 15, 1)


#2. 모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(15,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#model.summary()

import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1500, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
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
loss :  4299.3759765625
r2 score :  0.8055021605205419

loss :  6162.22998046875
r2 score :  0.7212292116397986


'''