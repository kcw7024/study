from gc import callbacks
from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris, load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout, Conv1D, LSTM
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터


datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

print(x_train.shape, x_test.shape)
# (398, 30) (171, 30)

x_train = x_train.reshape(398, 30)
x_test = x_test.reshape(171, 30)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)

#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

model = Sequential()
model.add(LSTM(10,input_shape=(30,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu')) 
model.add(Dense(1,activation='sigmoid'))


# model.summary() 


model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print("걸린시간 : ", end-start)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
# print(y_pred[:10])
# y_pred = np.argmax(model.predict(x_test), axis=-1)
print("loss : ", loss)
print("acc : ", acc)
# print("acc : ", accuracy_score(y_test, y_pred))

# loss :  0.0
# acc :  0.38596490025520325