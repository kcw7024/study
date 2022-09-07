from distutils.log import fatal
from gc import callbacks
from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris
from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Input, Dropout
import keras
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  (60000, 28, 28) (10000, 28, 28)
 
# x_train = x_train.reshape(60000, 28*28*1)
# x_test = x_test.reshape(10000, 28*28*1)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

 
# x_train = x_train.reshape(60000, 28,28,1)
# x_test = x_test.reshape(10000, 28,28,1)


#2. 모델
activation = 'relu'
drop = 0.2
# optimizer = 'adam'

inputs = Input(shape=(28, 28, 1), name='input')
x = Conv2D(64, (3, 3), padding='valid',
           activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same',
#            activation=activation, name='hidden2')(x) # 13, 13, 64
# x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (2, 2), padding='valid',
           activation=activation, name='hidden3')(x) # 12, 12, 32
x = Dropout(drop)(x)
# x = Flatten()(x) # 25*25*32 = 20000
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)    

model = Model(inputs=inputs, outputs=outputs)\
    

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1, factor=0.5)

from keras.optimizers import Adam

learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])


import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start()

loss, acc = model.evaluate(x_test, y_test)
print('learning_late :: ', learning_rate)
print('loss :: ', round(loss, 4))
print('acc :: ', round(loss, 4))
print('걸린시간 :: ', round(end, 4))

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

#1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])


plt.show()












