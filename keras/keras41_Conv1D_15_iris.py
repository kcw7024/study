import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler



# weight의 난수
#import tensorflow as tf
#tf.random.set_seed(66) 



#1. 데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names)
x, y = datasets.data, datasets.target


#print(x.shape, y.shape) #(150, 4) (150,)
#print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 :  [0 1 2]

#1-1. 데이터 전처리!!!

#1-2. keras 를 이용한 Onehotincoding
#keras의 utils 에서 to_categorical을 이용한다
from tensorflow.keras.utils import to_categorical
y = to_categorical(y) #단어 텍스트를 정수 시퀀스로 변환한다.
#print(y)
#print(y.shape) # (150, 3) < 확인.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)


scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


print(x_train.shape, x_test.shape) # (120, 4) (30, 4)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)




#print(y_train)
#print(y_test)

# 2. 모델

model = Sequential()
model.add(Conv1D(100, 2, input_shape=(4, 1)))
model.add(Flatten())
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
model.add(Dense(3))



import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
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
loss :  0.024214833974838257
r2 score :  0.8934637541656638

Conv1D

loss :  0.011380219832062721
r2 score :  0.9497437943012601


'''