from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout
import keras
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(64, (2, 2), padding='valid',
           activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same',
#            activation=activation, name='hidden2')(x) # 13, 13, 64
# x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding='valid',
           activation=activation, name='hidden3')(x) # 12, 12, 32
x = Dropout(drop)(x)
# x = Flatten()(x) # 25*25*32 = 20000
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)    

model = Model(inputs=inputs, outputs=outputs)


model.summary()

'''
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input (InputLayer)          [(None, 28, 28, 1)]       0

 hidden1 (Conv2D)            (None, 27, 27, 128)       640

 dropout (Dropout)           (None, 27, 27, 128)       0

 hidden2 (Conv2D)            (None, 27, 27, 64)        32832

 dropout_1 (Dropout)         (None, 27, 27, 64)        0

 hidden3 (Conv2D)            (None, 25, 25, 32)        18464

 dropout_2 (Dropout)         (None, 25, 25, 32)        0

 flatten (Flatten)           (None, 20000)             0

 hidden4 (Dense)             (None, 100)               2000100

 dropout_3 (Dropout)         (None, 100)               0

 outputs (Dense)             (None, 10)                1010

=================================================================
Total params: 2,053,046
Trainable params: 2,053,046
Non-trainable params: 0
_________________________________________________________________

MaxPooling 후

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input (InputLayer)          [(None, 28, 28, 1)]       0

 hidden1 (Conv2D)            (None, 27, 27, 128)       640

 dropout (Dropout)           (None, 27, 27, 128)       0

 hidden2 (Conv2D)            (None, 27, 27, 64)        32832

 dropout_1 (Dropout)         (None, 27, 27, 64)        0

 max_pooling2d (MaxPooling2D  (None, 13, 13, 64)       0
 )

 hidden3 (Conv2D)            (None, 11, 11, 32)        18464

 dropout_2 (Dropout)         (None, 11, 11, 32)        0

 flatten (Flatten)           (None, 3872)              0

 hidden4 (Dense)             (None, 100)               387300

 dropout_3 (Dropout)         (None, 100)               0

 outputs (Dense)             (None, 10)                1010

=================================================================
Total params: 440,246
Trainable params: 440,246
Non-trainable params: 0
_________________________________________________________________


global_average_pooling2d 적용후,
평균 pooling을 해주기때문에 연산량이 줄어든다

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input (InputLayer)          [(None, 28, 28, 1)]       0

 hidden1 (Conv2D)            (None, 27, 27, 128)       640

 dropout (Dropout)           (None, 27, 27, 128)       0

 hidden2 (Conv2D)            (None, 27, 27, 64)        32832

 dropout_1 (Dropout)         (None, 27, 27, 64)        0

 max_pooling2d (MaxPooling2D  (None, 13, 13, 64)       0
 )

 hidden3 (Conv2D)            (None, 11, 11, 32)        18464

 dropout_2 (Dropout)         (None, 11, 11, 32)        0

 global_average_pooling2d (G  (None, 32)               0
 lobalAveragePooling2D)

 hidden4 (Dense)             (None, 100)               3300

 dropout_3 (Dropout)         (None, 100)               0

 outputs (Dense)             (None, 10)                1010

=================================================================
Total params: 56,246
Trainable params: 56,246
Non-trainable params: 0
_________________________________________________________________

'''



model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='sparse_categorical_crossentropy')

import time
start = time.time()
model.fit(x_train, y_train, epochs=20, validation_split=0.4, batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print("걸린시간 : ", end-start)
# print("model.best_score_ :", model.best_score_)
# print("model.score :", model.score)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
# print(y_pred[:10])
y_pred = np.argmax(model.predict(x_test), axis=-1)

print("acc : ", accuracy_score(y_test, y_pred))

# acc :  0.9265