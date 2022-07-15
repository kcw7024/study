from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, LSTM, Reshape, SimpleRNN
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn import datasets
from tensorflow.keras.utils import to_categorical



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

# y는 꼭 원핫인코딩 해준다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)
#print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],      



#2. 모델 구성

model = Sequential()
# model.add(Conv2D(filters = 64, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D())                   # (14, 14, 64)
# model.add(Conv2D(32, (3, 3)))               # (12, 12, 32)     
# model.add(Conv2D(7, (3, 3)))                # (10, 10, 7)
# model.add(Flatten())                        # (N, 700)
# model.add(Dense(100, activation='relu'))    # (N, 100)
# model.add(Reshape(target_shape=(100, 1)))   # (N, 100, 1)
# model.add(Conv1D(10, 3))                    # (N, 98, 10)
# model.add(LSTM(16))                         # (N, 16) #3차원 받아들이고 2차원 아웃풋해줌
# model.add(Dense(32, activation='relu'))     # (N, 32)
# model.add(Dense(10, activation='softmax'))  # (N, 10)


model.add(Conv2D(filters = 64, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Flatten())                        # (N, 50176)
model.add(Dense(100, activation='relu'))    # (N, 100)                       
model.add(Reshape(target_shape=(100, 1)))   # (N, 100, 1)
model.add(Conv1D(10, 3, padding='same'))    # (N, 100, 10)                  
model.add(Flatten())                        # (N, 1000)
model.add(Reshape(target_shape=(1000,1)))   # (N, 1000, 1)
model.add(LSTM(16, return_sequences=True))  # (N, 16)
model.add(SimpleRNN(10))                    # (N, 10)
model.add(Reshape(target_shape=(10, 1)))    # (N, 10, 1)
model.add(Reshape(target_shape=(10,)))      # (N, 10)
#model.add(Conv1D(10, ))
# model.add(Conv1D(10, 3))                  
# model.add(LSTM(16))                         
# model.add(Dense(32, activation='relu'))     
model.add(Dense(10, activation='softmax'))  




model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        640
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023
_________________________________________________________________
flatten (Flatten)            (None, 700)               0
_________________________________________________________________
dense (Dense)                (None, 100)               70100
_________________________________________________________________
reshape (Reshape)            (None, 100, 1)            0
_________________________________________________________________
conv1d (Conv1D)              (None, 98, 10)            40
_________________________________________________________________
lstm (LSTM)                  (None, 16)                1728
_________________________________________________________________
dense_1 (Dense)              (None, 32)                544
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330
=================================================================
Total params: 93,869
Trainable params: 93,869
Non-trainable params: 0
_________________________________________________________________


max_pooling, flatten, reshape = 연산하지 않고 모양만 변형 시킨다.

'''


# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

# from tensorflow.python.keras.callbacks import EarlyStopping
# esp = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
#                 restore_best_weights=True)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, 
#                  callbacks=[esp], verbose=1) 


# #4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)



# # loss :  0.06842680275440216
# # accuracy :  0.9805999994277954


