#Dacon 쇼핑몰 문제

import numpy as np
import datetime as dt
import pandas as pd
from collections import Counter
import datetime as dt
from pyparsing import col
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from keras.layers import BatchNormalization
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


#데이터 경로 정의

path = './_data/shopping/'  # 경로 정의
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
#print(train_set) 
#print(train_set.shape) #(6255, 13)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
#print(test_set)
#print(test_set.shape) #(180, 12)


#1. 데이터 전처리

#결측치 처리
train_set = train_set.fillna(0)
test_set = test_set.fillna(0)
#print(train_set) #결측치 처리완료


data = pd.concat([train_set, test_set])
print(data)

#날짜데이터를 숫자로 바꿔주고 년/월/일로 쪼개준다.

data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
print(data) #확인.

# test_set['Date'] = pd.to_datetime(test_set['Date'])
# test_set['year'] = test_set['Date'].dt.year
# test_set['month'] = test_set['Date'].dt.month
# test_set['hour'] = test_set['Date'].dt.day
# # print(test_set) #확인.


data = data.drop(columns=['Date'])
print(data)

train_set = data[:len(train_set)]
test_set = data[len(train_set):]

print(train_set)
print(test_set)


x = train_set.drop(columns=['Weekly_Sales'])
y = train_set['Weekly_Sales']


x_train, x_test, y_train, y_test = train_test_split(
     x, y, train_size=0.99, random_state=777
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

print(x_train.shape) #(6192, 13)

#2. 모델 구성

# model = RandomForestRegressor(n_estimator=100, max_features=10, oob_score=True)
# model.fit(x_train, y_train)

model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(200, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(300, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(200, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(200, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(300, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(200, activation='swish'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=128)

model.compile(loss='mse', optimizer='adam')

earlyStopping=EarlyStopping(monitor='loss',patience=10, mode='auto', verbose=1,restore_best_weights=True)

model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping], epochs=100, batch_size=128, verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # test로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)

#RMSE 함수정의, 사용

def RMSE(y_test, y_predict):  # mse에 루트를 씌운다.
    return np.sqrt(mean_squared_error(y_test, y_predict))


rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#5.csv로 내보낸다
result = pd.read_csv(path + 'sample_submission.csv', index_col=0)
#index_col=0 의 의미 : index col을 없애준다.

#test_set = test_set.astype(np.float32)
y_summit = model.predict(test_set)
#print(y_summit)
print(y_summit.shape)  # (1459, 1)


result['Weekly_Sales'] = y_summit

#result 에서 지정해준 submission의 count 값에 y_summit값을 넣어준다.

#.to_csv() 를 사용해서 sample_submission.csv를 완성

#2
#result = abs(result) #절대값처리.... 인데 이걸로하면 안되는디
result.to_csv(path + 'sample_submission.csv', index=True)





'''

loss :  33649528832.0
RMSE :  183438.0710104603

loss :  27267203072.0
RMSE :  165127.8334376112

loss :  24211890176.0
RMSE :  155601.70919475352

loss :  24073287680.0
RMSE :  155155.67259426

loss :  27762049024.0
RMSE :  166619.45825564663

loss :  23707613184.0
RMSE :  153972.76323819187

Randomforest 사용후 

loss :  0.8970531771426602
RMSE :  174983.71835076745

loss :  0.910689218072747
RMSE :  162983.30219609317

loss :  0.9104498499855576
RMSE :  163201.56753344138

loss :  30833950720.0
RMSE :  175595.98262320893


'''