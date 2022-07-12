#kaggle 집값~ 문제풀이!!
#https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

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
#from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from keras.layers import BatchNormalization

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
def get_year(date) : 
    year = date[7:]
    year = int(year)
    return year
def get_day(date) : 
    day = date[:2]
    day = int(day)
    return day




train_set['Month'] = train_set['Date'].apply(get_month)
test_set['Month'] = test_set['Date'].apply(get_month)
train_set['Year'] = train_set['Date'].apply(get_year)
test_set['Year'] = train_set['Date'].apply(get_year)
train_set['Day'] = train_set['Date'].apply(get_day)
test_set['Day'] = train_set['Date'].apply(get_day)


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
scaler = MinMaxScaler()
#scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 


print(x_train.shape) #(6192, 13)


#2. 모델 구성

input = Input(shape=(13, ))
dense1 = Dense(400)(input)
batch1 = BatchNormalization()(dense1)
act1 = Activation('relu')(batch1)
drp1 = Dropout(0.2)(act1)
dense2 = Dense(200)(drp1)
batch2 = BatchNormalization()(dense2)
act2 = Activation('relu')(batch2)
drp2 = Dropout(0.2)(act2)
dense3 = Dense(400)(drp2)
batch3 = BatchNormalization()(dense3)
act3 = Activation('relu')(batch3)
drp3 = Dropout(0.2)(act3)
output = Dense(1)(drp3)


model = Model(inputs = input, outputs= output)




#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=128)

model.compile(loss='mse', optimizer='adam')

earlyStopping=EarlyStopping(monitor='loss',patience=50, mode='auto', verbose=1,restore_best_weights=True)

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

loss :  [231572144128.0, 399370.21875]
RMSE :  481219.4298524854

loss :  [214719266816.0, 379795.25]
RMSE :  463378.1069933613



'''