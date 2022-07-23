#넘파이에서 불러와서 모델 구성
#성능 비교

import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화


#1. 데이터

#수치화한 데이터를 저장한다.
# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0]) # train x값
# np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1]) # train y값
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0]) # test x값
# np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1]) # test y값

x_train = np.load('d:/study_data/_save/_npy/keras49_6_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_6_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_6_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_6_test_y.npy')
test = np.load('d:/study_data/_save/_npy/keras49_6_test.npy')

#print(x_train)
print(x_train.shape) #(40000, 28, 28, 1)
print(y_train.shape) #(40000,)
print(x_test.shape)  #(40000, 28, 28, 1)
print(y_test.shape)  #(40000,)

#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(xy_train[0][0], xy_train[0][1]) #배치를 최대로 잡으면 이방법도 가능
hist = model.fit(x_train, y_train, 
                 epochs=10, 
                 steps_per_epoch=33, #전체데이터/batch=160/5=32
                 validation_split=0.2
                 )

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])

# loss :  0.6947375535964966
# val_accuracy :  1.0
# accuracy :  0.3030303120613098
# val_loss :  0.6914277672767639


# loss :  0.28313571214675903
# val_accuracy :  0.5997999906539917
# accuracy :  0.8977749943733215
# val_loss :  1.091827392578125