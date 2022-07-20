#컬럼 7개 이상, 앙상블 사용

from pickletools import optimize
import numpy as np
import seaborn as sns #데이터 시각화할때 사용. 여기선 사용안했음 
import datetime as dt
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
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
data_amore = data_amore.drop(columns=["금액(백만)", "신용비", "개인", "기관", "외인(수량)", "외국계", "프로그램", "외인비", "전일비"]) #아모레
data_samsung = data_samsung.drop(columns=["금액(백만)", "신용비", "개인", "기관", "외인(수량)", "외국계", "프로그램", "외인비", "전일비"]) #삼성
#print(data_amore.columns, data_samsung.columns) #  ['일자', '시가', '고가', '저가', '종가', '증감량', '등락률', '거래량']

#결측치 처리
data_amore = data_amore.fillna(0)
data_samsung = data_samsung.fillna(0)
y = data_amore['시가']


data_amore = data_amore.loc[data_amore['일자']>="2018/05/04"]
data_samsung = data_samsung.loc[data_samsung['일자']>="2018/05/04"] 

feature_cols = ['시가', '고가', '저가', '종가', '증감량', '등락률', '거래량']
label_cols = ['시가']

# 시계열 데이터 만드는 함수
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

SIZE = 20
x1 = split_x(data_amore[feature_cols], SIZE)
x2 = split_x(data_samsung[feature_cols], SIZE)
y = split_x(data_amore[label_cols], SIZE)

print(x1.shape, x2.shape, y.shape) #(1016, 20, 7) (1016, 20, 7) (1016, 20, 1)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, test_size=0.2, shuffle=False
    )

scaler = MinMaxScaler()
print(x1_train.shape, x1_test.shape) #(812, 20, 7) (204, 20, 7)
print(x2_train.shape, x2_test.shape) #(812, 20, 7) (204, 20, 7)
print(y_train.shape, y_test.shape) #(812, 20, 1) (204, 20, 1)


x1_train = x1_train.reshape(812*20,7)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(204*20,7)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(812*20,7)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(204*20,7)
x2_test = scaler.transform(x2_test)

# Reshape 해주기
x1_train = x1_train.reshape(812, 20, 7)
x1_test = x1_test.reshape(204, 20, 7)
x2_train = x2_train.reshape(812, 20, 7)
x2_test = x2_test.reshape(204, 20, 7)


# 2. 모델구성
# 2-1. 모델1
# input1 = Input(shape=(20,7))
# dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
# dense2 = LSTM(128, activation='relu', name='d2')(dense1)
# dense3 = Dense(64, activation='relu', name='d3')(dense2)
# output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# # 2-2. 모델2
# input2 = Input(shape=(20, 7))
# dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
# dense12 = LSTM(128, activation='swish', name='d12')(dense11)
# dense13 = Dense(64, activation='relu', name='d13')(dense12)
# dense14 = Dense(32, activation='relu', name='d14')(dense13)
# output2 = Dense(16, activation='relu', name='out_d2')(dense14)

# from tensorflow.python.keras.layers import concatenate
# merge1 = concatenate([output1, output2], name='m1')
# merge2 = Dense(100, activation='relu', name='mg2')(merge1)
# merge3 = Dense(100, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

# model = Model(inputs=[input1, input2], outputs=[last_output])


# model.save("./amore3_model/keras46_save_model.h5")


# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import time
# # 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
# print(date) #2022-07-07 17:21:36.266674 : 현재시간

# filepath = './_ModelCheckPoint/amore3/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# start_time = time.time()

# EarlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800, restore_best_weights=True)

# ModelCheckpoint = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath="".join([filepath,'amore_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
#                       )

# fit_log = model.fit([x1_train, x2_train], y_train,
#                     epochs=500, batch_size=64, 
#                     callbacks=[EarlyStopping, ModelCheckpoint], 
#                     validation_split=0.1)

#end_time = time.time() - start_time

model = load_model("./_ModelCheckPoint/amore6/amore_0718_2154_0065-103714392.0000.hdf5")
model.load_weights("./_ModelCheckPoint/amore6_weights/keras46_save_weights1.h5")

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('19일 아모레 시가 예측 : ', predict[-1:]) 
#print('predict: ', predict)
#print('걸린 시간: ', end_time-start_time)

# 최종 주가예측 값
# loss:  120710024.0
# 19일 아모레 시가 예측 :  [[136925.42]]