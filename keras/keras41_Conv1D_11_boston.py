#과적합 
from pickletools import optimize
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split  # 훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import r2_score, accuracy_score


# 1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 


print(x_train.shape, y_train.shape) #(404, 13) (404,)
print(x_test.shape, y_test.shape) #(102, 13) (102,)


x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)


# 2. 모델

model = Sequential()
model.add(Conv1D(100, 2, input_shape=(13, 1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#model.summary()

import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
# print(date) #2022-07-07 17:21:36.266674 : 현재시간

# filepath = './_ModelCheckPoint/k26_1/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k26_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
#                       )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=2000, batch_size=128,
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
# print(x_test.shape)
# print(y_predict.shape)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)


'''
loss :  23.0333194732666
r2 score :  0.7180569176045343

loss :  23.97797966003418
r2 score :  0.7064936637306685

loss :  26.71713638305664
r2 score :  0.6729645902274248


'''