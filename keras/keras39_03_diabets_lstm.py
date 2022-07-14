
#과적합 
from pickletools import optimize
from tabnanny import verbose
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split  # 훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

print(x_train.shape, x_test.shape) #(353, 10) (89, 10) 

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)
#print(x_train.shape) #(353, 5, 2, 1)
#print(np.unique(y_train, return_counts=True))



# 2. 모델

model = Sequential()
model.add(LSTM(100, input_shape=(10, 1)))
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


# input = Input(shape=(10,))
# dense1 = Dense(100, activation='relu')(input)
# dense2 = Dense(200, activation='relu')(dense1)
# dense3 = Dense(300, activation='relu')(dense2)
# dense4 = Dense(200, activation='relu')(dense3)
# dense5 = Dense(100, activation='relu')(dense4)
# output = Dense(1)(dense5)

# model = Model(inputs = input, outputs = output)

import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10000, batch_size=360,
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

loss :  3885.051025390625
r2 score :  0.41164280628692285


'''
