from pickletools import optimize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
vgg16.trainable = False  # vgg16의 레이어에 대해서는 훈련을 안시킨다.(가중치 동결)

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss, acc  = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss, acc)

# 걸린 시간 :  302.32
# loss, acc  [1.1628283262252808, 0.6105999946594238]