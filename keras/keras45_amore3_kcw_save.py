#컬럼 7개 이상, 앙상블 사용

from pickletools import optimize
import numpy as np
from regex import P
import seaborn as sns #데이터 시각화할때 사용. 여기선 사용안했음 
import datetime as dt
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import tensorflow as tf

#1. 데이터

path = './_data/test_amore_0718/' #경로 정의
data_amore = pd.read_csv(path + '아모레220718.csv', encoding='CP949', thousands = ',')      #아모레
data_samsung = pd.read_csv(path + '삼성전자220718.csv', encoding='CP949', thousands = ',')  #삼성

data_amore = data_amore.sort_values(by='일자', ascending=True) #오름차순 정렬
data_samsung = data_samsung.sort_values(by='일자', ascending=True) #오름차순 정렬
# print(data_amore.head(), data_samsung.head()) 정상~~~~

data_amore = data_amore.rename(columns={'Unnamed: 6':'증감량'})
data_samsung = data_samsung.rename(columns={'Unnamed: 6':'증감량'})

#필요없는거 날리기
data_amore = data_amore.drop(columns=["금액(백만)", "신용비", "외인(수량)", "외국계", "프로그램", "외인비", "전일비"]) #아모레
data_samsung = data_samsung.drop(columns=["금액(백만)", "신용비", "외인(수량)", "외국계", "프로그램", "외인비", "전일비"]) #삼성
#print(data_amore.columns, data_samsung.columns) #  ['일자', '시가', '고가', '저가', '종가', '증감량', '등락률', '거래량']
y = data_amore['종가'] #y값 지정

#print(data_amore.head())
#print(data_samsung.head())


#결측치 처리
data_amore = data_amore.fillna(0)
data_samsung = data_samsung.fillna(0)

data_amore = data_amore.loc[data_amore['일자']>="2018/05/04"]
data_samsung = data_samsung.loc[data_samsung['일자']>="2018/05/04"] 

feature_cols = ['시가', '고가', '저가', '증감량', '등락률', '거래량', '기관', '개인', '종가']
#label_cols = ['종가']

data_amore = data_amore[feature_cols]
data_samsung = data_samsung[feature_cols]

data_amore = np.array(data_amore)
data_samsung = np.array(data_samsung)

# 시계열 데이터 만드는 함수
def split_x(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)): #x_dataset(data_amore의 총 행수)
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1 
        
        if y_end_number > len(dataset):
            break
        
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1: y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 7
size = 3

x1, y1 = split_x(data_amore, time_steps, size)
x2, y2 = split_x(data_samsung, time_steps, size)

print(x1.shape, x2.shape, y1.shape, y2.shape) #(1027, 7, 8) (1027, 7, 8) (1027, 3) (1027, 3)



x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, test_size=0.2, shuffle=False
    )


# 정규화 해주기
scaler = MinMaxScaler()
#print(x1_train.shape, x1_test.shape) #(821, 7, 8) (206, 7, 8)
#print(x2_train.shape, x2_test.shape) #(821, 7, 8) (206, 7, 8)
#print(y1_train.shape, y1_test.shape) #(821, 3) (206, 3)


x1_train = x1_train.reshape(821*7,8) 
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(206*7,8)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(821*7,8)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(206*7,8)
x2_test = scaler.transform(x2_test)

# Reshape 해주기
x1_train = x1_train.reshape(821, 7, 8)
x1_test = x1_test.reshape(206, 7, 8)
x2_train = x2_train.reshape(821, 7, 8)
x2_test = x2_test.reshape(206, 7, 8)

# 2. 모델구성
# 2-1. 모델1
input1 = Input(shape=(7, 8))
dense1 = Conv1D(100, 2, activation='relu')(input1)
dense2 = LSTM(100, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense2)
output1 = Dense(100, activation='relu')(dense4)

# 2-2. 모델2
input2 = Input(shape=(7, 8))
dense11 = Conv1D(100, 2, activation='relu')(input2)
dense12 = LSTM(100, activation='relu')(dense11)
dense13 = Dense(100, activation='relu')(dense12)
dense14 = Dense(100, activation='relu')(dense13)
output2 = Dense(100, activation='relu')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(100)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
print(date) #2022-07-07 17:21:36.266674 : 현재시간

filepath = './_test/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

start_time = time.time()

EarlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)

ModelCheckpoint = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath,'amore_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
                      )

fit_log = model.fit([x1_train, x2_train], y1_train,
                    epochs=600, batch_size=64, 
                    callbacks=[EarlyStopping, ModelCheckpoint], 
                    validation_split=0.1)

end_time = start_time - time.time() 

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
#print('예측 종가 : ', predict[-3:]) 
print('예측 종가 : ', predict[-1:]) 
#print('predict: ', predict)
#print('걸린시간 : ', start_time-end_time)


# loss:  80900816.0
# 예측 종가 :  [[133816.12]] amore_0719_1647_0202-68309000.0000

# loss:  72322528.0
# 예측 종가 :  [[132051.53]] amore_0719_1655_0149-63371124.0000

# loss:  116945672.0
# 예측 종가 :  [[133099.89]] amore_0719_1721_0006-80499232.0000

# loss:  68304696.0
# 예측 종가 :  [[132656.03]] amore_0719_1742_0267-65257572.0000

# loss:  76025208.0
# 예측 종가 :  [[132203.08]] amore_0719_1755_0231-62495820.0000

# loss:  64770456.0
# 예측 종가 :  [[132751.08]] amore_0719_1923_0161-55904256.0000