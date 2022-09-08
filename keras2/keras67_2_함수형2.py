from ast import Mod
from email.mime import base
from pickletools import optimize
from keras.models import Model
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Dropout
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar100
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape)

base_model = InceptionV3(weights='imagenet', include_top=False)
# base_model.summary()


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(102, activation='relu')(x)

output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

#1. 
# ex : base_model.layers[3] < 이런식으로 특정 레이어만 동결도 가능

# for layer in base_model.layers :    
#     layer.trainable = False

'''
Total params: 22,022,082
Trainable params: 219,298
Non-trainable params: 21,802,784
'''

#2.
# base_model.trainable = False

'''
Total params: 22,022,082
Trainable params: 219,298
Non-trainable params: 21,802,784
'''
    
model.summary()

print(base_model.layers)
# 해당 모델의 모든 layers 확인가능




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






