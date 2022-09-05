from pydoc import describe
import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

# from keras.optimizers import RMSprop, SGD, Nadam

op_list = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSprop, nadam.Nadam]

print(op_list)

learning_rate = 0.00001

for optimizer in op_list :
    optimizer = optimizer(learning_rate = learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x, y, epochs=50, batch_size=1)
    loss = model.evaluate(x, y)
    y_pred = model.predict([11])
    print('loss : ', round(loss, 4), 'lr : ', learning_rate, '결과물 : ', y_pred)

# optimizer = adam.Adam(learning_rate=learning_rate)
# model.compile(loss='mse', optimizer='adam')
# model.compile(loss='mse', optimizer=optimizer)
# model.fit(x, y, epochs=50, batch_size=1)
# 4. 평가 , 예측
# loss = model.evaluate(x, y)
# y_pred = model.predict([11])
# print('loss : ', round(loss, 4), 'lr : ', learning_rate, '결과물 : ', y_pred)

'''
optimizer:  Adam loss :  2.5893 lr :  1e-05 결과물 :  [[11.430659]]
optimizer:  Adadelta loss :  2.589 lr :  1e-05 결과물 :  [[11.428485]]
optimizer:  Adagrad loss :  2.5761 lr :  1e-05 결과물 :  [[11.279323]]
optimizer:  Adamax loss :  2.5433 lr :  1e-05 결과물 :  [[11.222636]]
optimizer:  RMSprop loss :  2.5045 lr :  1e-05 결과물 :  [[11.357602]]
optimizer:  Nadam loss :  2.4438 lr :  1e-05 결과물 :  [[11.128921]]
'''


