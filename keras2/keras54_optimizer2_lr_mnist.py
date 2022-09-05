from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
import numpy as np
from sklearn import datasets
from keras.utils import to_categorical
# import tensorflow as tf
# print(tf.__version__)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2. 모델 구성

model = Sequential()
  
model.add(Conv2D(filters = 64, kernel_size=(3,3),   # 출력 (N, 28, 28, 64) 패딩이 세임이므로
                 padding='same',
                 input_shape=(28, 28, 1))) # (batch_size: 행을 자르는 단위, rows, cols, channels) : 데이터는 항상 변동이 있을 수 있기때문에 생략 
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding='valid', #디폴트옵션  
                 activation='relu'))  
model.add(Conv2D(32, (2,2), 
                 padding='valid', activation='relu'))
model.add(Flatten())  # 출력 (N, 63)
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rate = 0.00001

optimizer = adam.Adam(learning_rate=learning_rate)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping
esp = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                    restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=128, 
                 validation_split=0.2, 
                 callbacks=[esp], verbose=1) 

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

# loss :  0.07187040150165558
# accuracy :  0.9800999760627747