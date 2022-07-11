from tkinter.tix import Y_REGION
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout #이미지작업(Conv2D), 데이터를 펼친다(Flatten)
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping





#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)     #(10000, 32, 32, 3) (10000, 1)

#원핫인코딩 해준다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

print(x_train.shape)
print(np.unique(y_train, return_counts=True))


#2. 모델 구성

model = Sequential()
 
model.add(Dense(64, input_shape=(3072, ))) # (batch_size, rows, cols, channels)  
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping
esp = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=128, validation_split=0.2, 
                 callbacks=[esp], verbose=1) 




#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)


# loss :  4.605686664581299
# accuracy :  0.009999999776482582


