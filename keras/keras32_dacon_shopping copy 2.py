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

#데이터 경로 정의

path = './_data/shopping/'  # 경로 정의
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
Weekly_Sales = train_set[['Weekly_Sales']]
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)

######## 년, 월 ,일 분리 ############

train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]

test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]

train_set.drop(['Date','Weekly_Sales'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
test_set.drop(['Date'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)
##########################################

# ####################원핫인코더###################

df = pd.concat([train_set, test_set])
print(df)

alldata = pd.get_dummies(df, columns=['day','Store','month', 'year', 'IsHoliday'])
print(alldata)

train_set2 = alldata[:len(train_set)]
test_set2 = alldata[len(train_set):]

print(train_set2)
print(test_set2)
# train_set = pd.get_dummies(train_set, columns=['Store','month', 'year', 'IsHoliday'])
# test_set = pd.get_dummies(test_set, columns=['Store','month', 'year', 'IsHoliday'])




###############프로모션 결측치 처리###############

train_set2 = train_set2.fillna(0)
test_set2 = test_set2.fillna(0)

print(train_set2)
print(test_set2)

##########################################

train_set2 = pd.concat([train_set2, Weekly_Sales],axis=1)
print(train_set2)

x = train_set2.drop(['Weekly_Sales'], axis=1)
y = train_set2['Weekly_Sales']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_set2 = scaler.transform(test_set2)

print(test_set2)




#2. 모델 구성

input = Input(shape=(77, ))
dense1 = Dense(400)(input)
batch1 = BatchNormalization()(dense1)
act1 = Activation('swish')(batch1)
drp1 = Dropout(0.2)(act1)
dense2 = Dense(200)(drp1)
batch2 = BatchNormalization()(dense2)
act2 = Activation('swish')(batch2)
drp2 = Dropout(0.2)(act2)
dense3 = Dense(400)(drp2)
batch3 = BatchNormalization()(dense3)
act3 = Activation('swish')(batch3)
drp3 = Dropout(0.2)(act3)
dense4 = Dense(200)(drp3)
batch4 = BatchNormalization()(dense4)
act4 = Activation('relu')(batch4)
drp4 = Dropout(0.5)(act4)
output = Dense(1)(drp4)



model = Model(inputs = input, outputs= output)




#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=128)

model.compile(loss='mse', optimizer='adam')

earlyStopping=EarlyStopping(monitor='loss',patience=50, mode='auto', verbose=1,restore_best_weights=True)

model.fit(x_train,y_train, validation_split=0.2, callbacks=[earlyStopping], epochs=2000, batch_size=128, verbose=1)


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

test_set = test_set.astype(np.float32)
y_summit = model.predict(test_set)
#print(y_summit)
print(y_summit.shape)  # (1459, 1)


result['Weekly_Sales'] = y_summit

#result 에서 지정해준 submission의 count 값에 y_summit값을 넣어준다.

#.to_csv() 를 사용해서 sample_submission.csv를 완성

#2
#result = abs(result) #절대값처리.... 인데 이걸로하면 안되는디
result.to_csv(path + 'sample_submission.csv', index=True)
