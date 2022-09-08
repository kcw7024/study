from ast import Mod
from pickletools import optimize
from keras.models import Model
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Dropout
from keras.applications import VGG16
from keras.datasets import cifar100
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)


input = Input(shape=(32, 32, 3))
x = VGG16(include_top=False)(input)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)


# optimizer = Adam(learning_late=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

import time
# start = time.time()
model.fit(x_train, y_train, epochs=5, callbacks=[es, reduce_lr], batch_size=128, verbose=1)
# end = time.time() - start()


loss, acc = model.evaluate(x_test, y_test)
# print('걸린시간 : ', round(end, 2))
print('loss:', round(loss, 4))
print('acc:', round(acc, 4))


