#kaggle 집값~ 문제풀이!!
#https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

import numpy as np
import datetime as dt
import pandas as pd
from collections import Counter
import datetime as dt
from pyparsing import col
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


#데이터 경로 정의

path = './_data/shopping/'  # 경로 정의
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
#print(train_set) 
#print(train_set.shape) #(6255, 13)
test_set = pd.read_csv(path + 'test.csv', index_col=0)  #  

#print(test_set)
#print(test_set.shape) #(180, 12)


#1. 데이터 전처리

#결측치 처리
train_set = train_set.fillna(0)
test_set = test_set.fillna(0)
#print(train_set) #결측치 처리완료

#문자형식을 숫자로 변환하고 월만 빼와서 새로만들어준다.
def get_month(date) : 
    month = date[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)
test_set['Month'] = test_set['Date'].apply(get_month)


#print(train_set)

# Date 칼럼의 휴일정보를 변경 해준다.
def holiday_to_number(isholiday):
    if isholiday == True:
        number = 1
    else:
        number = 0
    return number

train_set['NumberHoliday'] = train_set['IsHoliday'].apply(holiday_to_number)
test_set['NumberHoliday'] = test_set['IsHoliday'].apply(holiday_to_number)

train_set = train_set.drop(columns=['Date','IsHoliday'])
test_set = test_set.drop(columns=['Date','IsHoliday'])

x = train_set.drop(columns=['Weekly_Sales'])
y = train_set['Weekly_Sales']



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.99, random_state=777
)

#scaler = RobustScaler()
#scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

# print(x_train.shape)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=11))
model.add(BatchNormalization(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=128)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

earlyStopping=EarlyStopping(monitor='loss',patience=100,mode='auto', verbose=1,restore_best_weights=True)

model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping],
          epochs=2000, batch_size=128, verbose=1)


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

# loss :  207473.671875
# RMSE :  326552.4564881789

# loss :  [0.11741948872804642, 0.0]
# RMSE :  1125946.7646446743

# loss :  [0.11741948872804642, 0.0]
# RMSE :  1125888.8261547252

loss :  [70131146752.0, 147742.53125]
RMSE :  264822.8366232661

loss :  [241847025664.0, 404654.75]
RMSE :  491779.456456335

loss :  [107149369344.0, 191812.359375]
RMSE :  327336.77610361704

'''