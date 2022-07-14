# 실습!!!!
# acc 0.98 이상으로 만들어낼 것.
# conv 3개 이상 쌓기
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D
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

x_train = x_train.reshape(60000, 784, 1)
x_test = x_test.reshape(10000, 784, 1)
print(x_train.shape)
#print(np.unique(y_train, return_counts=True))

#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],      



#2. 모델 구성

model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   #(batch_size, input_dim)
# model.summary()
# (input_dim + bias) * units = summary Params 갯수(Dense 모델)


# model.add(Conv2D(filters = 64, kernel_size=(3,3),   # 출력 (N, 28, 28, 64) 패딩이 세임이므로
#                  padding='same',
#                  input_shape=(28, 28, 1))) # (batch_size: 행을 자르는 단위, rows, cols, channels) : 데이터는 항상 변동이 있을 수 있기때문에 생략 
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), 
#                  padding='valid', #디폴트옵션  
#                  activation='relu'))  
# model.add(Conv2D(32, (2,2), 
#                  padding='valid', activation='relu'))
# model.add(Flatten())  # 출력 (N, 63)
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))


model.add(Conv1D(100, 2, input_shape=(784, 1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.add(Dense(10))



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping
esp = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[esp], verbose=1) 




#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)



# loss :  0.06842680275440216
# accuracy :  0.9805999994277954

# Conv1D

# loss :  4.978883266448975
# accuracy :  0.0957999974489212


