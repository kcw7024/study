
#유방암 데이터활용
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터

datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DECR)
#print(datasets.feature_names)


x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

print(x_train.shape, x_test.shape) #(455, 30) (114, 30)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)




# 2. 모델

model = Sequential()
model.add(LSTM(100, input_shape=(30, 1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))



# input = Input(shape=(30,))
# dense1 = Dense(10, activation='relu')(input)
# dense2 = Dense(20, activation='relu')(dense1)
# dense3 = Dense(30, activation='relu')(dense2)
# dense4 = Dense(20, activation='relu')(dense3)
# dense5 = Dense(10, activation='relu')(dense4)
# output = Dense(1, activation='sigmoid')(dense5)

# model = Model(inputs = input, outputs = output)


# #linear : 선형
# #sigmoid : 0 과 1사이로 값을 반환해줌  (이진분류는 아웃풋에서 무조건 sigmoid)
# # : 0.4 같은 숫자를 처리하기 위해 반올림 처리 해줘야함


import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
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

loss :  0.04654461517930031
r2 score :  0.8018690279121176


'''