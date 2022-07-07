import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행




#print(y_train)
#print(y_test)

# 2. 모델

# model = Sequential()
# model.add(Dense(100, input_dim=4, activation='linear'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))


input = Input(shape=(4,))
dense1 = Dense(100, activation='relu')(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(100, activation='softmax')(dense4)
output = Dense(3)(dense5)

model = Model(inputs = input, outputs = output)


# sigmoid 0~1 사이
# softmax 다중분류에서 사용(마찬가지로 중간레이어에서는 사용할수없고, 아웃풋에서만 사용가능)
# 현재 예시에서 마지막 노드 값이 3가지. 마지막 노드 값은 데이터에따라 바뀐다
# 결과값의 총 합이 1이된다
# ex) 출력값 70,20,10 일때 -> [0.7,0.2,0.1]


import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
print(date) #2022-07-07 17:21:36.266674 : 현재시간

filepath = './_ModelCheckPoint/k25_5/'
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