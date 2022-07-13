from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn import datasets
from tensorflow.keras.utils import to_categorical

#Conv2D = 2D 이미지 관련
#Flatten 납작하게 하는 것 쭉 늘려서
#Maxpooling2D 큰 놈 골라내는 것.

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape, y_train.shape)   #(60000, 28, 28) (60000,)
#print(x_test.shape, y_test.shape)     #(10000, 28, 28) (10000,)

# reshape 이용해서 (60000, 28, 28)와 (28, 28, 1)을 맞추기. 순서와 위치는 그대로. 
# 하지만 모양은 바꿀 수 있음.


'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)

#print(np.unique(y_train, return_counts=True))

#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) 
# ----> 다중분류임 (unique를 찍어보았을 때, output이 10, 즉 2 이상이었으므로)





#2. 모델 구성

# model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   #(batch_size, input_dim)
# model.summary()
# (input_dim + bias) * units = summary Params 갯수(Dense 모델)



input1 = Input(shape=(28,28,1))
Conv2D_1 = Conv2D(64, 3, padding='same')(input1)
Maxp1 = MaxPooling2D()(Conv2D_1)
Conv2D_2 = Conv2D(32, 2, padding='valid', activation='relu')(Maxp1)
Conv2D_3 = Conv2D(32, 2, padding='valid', activation='relu')(Conv2D_2)
flatten = Flatten()(Conv2D_3)
dense1 = Dense(32)(flatten)
dense2 = Dense(64, activation = 'relu')(dense1)
dense3 = Dense(32, activation = 'relu')(dense2)
output1 = Dense(1, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)   

#model.summary()





#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
                restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, 
                 callbacks=[earlyStopping], verbose=1) 




#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)


