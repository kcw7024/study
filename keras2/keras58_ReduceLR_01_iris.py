from gc import callbacks
from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)


print(x_train.shape, x_test.shape)
# (105, 4) (45, 4)

x_train = x_train.reshape(105, 4)
x_test = x_test.reshape(45, 4)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(105, 2, 2, 1)
x_test = x_test.reshape(45, 2, 2, 1)

#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(2, 2, 1), name='input')
x = Conv2D(64, (1, 1), padding='valid',
           activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same',
#            activation=activation, name='hidden2')(x) # 13, 13, 64
# x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (1, 1), padding='valid',
           activation=activation, name='hidden3')(x) # 12, 12, 32
x = Dropout(drop)(x)
# x = Flatten()(x) # 25*25*32 = 20000
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(1, activation='sigmoid', name='outputs')(x)    

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

loss, acc = model.evaluate(x_test,y_test)

print("걸린시간 : ", end-start)
# print("model.best_score_ :", model.best_score_)
# print("model.score :", model.score)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
# print(y_pred[:10])
# y_pred = np.argmax(model.predict(x_test), axis=-1)
print("loss : ", loss)
print("acc : ", acc)
# print("acc : ", accuracy_score(y_test, y_pred))

# loss :  0.0
# acc :  0.2888889014720917