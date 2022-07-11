
import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터

datasets = load_digits()
x, y = datasets.data, datasets.target

print(x.shape, y.shape) #(1797, 64) (1797,) - 8 x 8 의 이미지가 1797장 있다. /  #원핫 인코딩으로 1797,10으로 만들어준다. 
print(np.unique(y, return_counts=True))     #[0 1 2 3 4 5 6 7 8 9]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

print(x_train.shape, x_test.shape) #(1437, 64) (360, 64)

x_train = x_train.reshape(1437, 4, 4, 4)
x_test = x_test.reshape(360, 4, 4, 4)





# 2. 모델

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(3,3), input_shape=(4,4,4), activation='linear'))
model.add(Dropout(0.25))
model.add(Conv2D(200, (2,2),activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(300, (1,1),activation='relu'))
model.add(Conv2D(200, (1,1),activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(100, (1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))



#model.summary()

import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time


#4. 평가, 예측
print(('#'*70) + '1.기본출력')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)


'''
loss :  0.009047186933457851
r2 score :  0.8935664424989282
'''