import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_trian = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = VGG16() # include_top=True, input_shape=(224,224,3)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg16.summary()
# vgg16.trainable=False # 가중치를 동결시킨다.
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))

model.trainable = False

model.summary()

                                        # Trainable:True / Vgg16 False / model False 
print(len(model.weights))               # 30 / 30 / 30
print(len(model.trainable_weights))     # 30 / 4 /  0



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

loss, acc = model.evaluate(x_test, y_test)

print('loss:', round(loss, 4))
print('acc:', round(acc, 4))


