
#keras18_gpu_test3파일의 summary를 확인
#summary와 time


import numpy as np
from sklearn.datasets import fetch_covtype
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터

datasets = fetch_covtype()

x, y = datasets.data, datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,) 
print(np.unique(y, return_counts=True))     #[1 2 3 4 5 6 7]
#(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)
#print(y)

#pandas로 원핫코딩작업
y = pd.get_dummies(y)
print(y)


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

# model = Sequential()
# model.add(Dense(100, input_dim=54, activation='linear'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(7, activation='softmax'))


input = Input(shape=(54,))
dense1 = Dense(100, activation='relu')(input)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(200, activation='relu')(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(300, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(200, activation='relu')(drop3)
drop4 = Dropout(0.5)(dense4)
dense5 = Dense(100, activation='relu')(drop4)
output = Dense(7, activation='softmax')(dense5)


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

filepath = './_ModelCheckPoint/k26_8/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath,'k26_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
                      )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=1,
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