from gc import callbacks
from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris, load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(tf.__version__)
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터


datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

print(x_train.shape, x_test.shape)
# (354, 13) (152, 13)

x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)

#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(13,1,1), name='input')
x = Dense(64, activation=activation, name='hidden1')(x)
x = Dense(64, activation=activation, name='hidden2')(x)
x = Dense(64, activation=activation, name='hidden3')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)    

model = Model(inputs=inputs, outputs=outputs)


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

loss, r2_score = model.evaluate(x_test,y_test)

print("걸린시간 : ", end-start)
# print("model.best_score_ :", model.best_score_)
# print("model.score :", model.score)

from sklearn.metrics import accuracy_score, r2_score

y_pred = model.predict(x_test)
# print(y_pred[:10])
# y_pred = np.argmax(model.predict(x_test), axis=-1)
print("loss : ", loss)
print("acc : ", r2_score)
# print("acc : ", accuracy_score(y_test, y_pred))

# acc :  0.9265