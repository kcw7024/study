


#과적합 
from pickletools import optimize
from tabnanny import verbose
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split  # 훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터

datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행



# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


# input = Input(shape = (8,))
# dense1 = Dense(100)(input)
# dense2 = Dense(200, activation='relu')(dense1)
# dense3 = Dense(100, activation='relu')(dense2)
# dense4 = Dense(300, activation='relu')(dense3)
# dense5 = Dense(100, activation='relu')(dense4)
# dense6 = Dense(200, activation='relu')(dense5)
# dense7 = Dense(100, activation='relu')(dense6)
# output = Dense(1)(dense7)

# model = Model(inputs = input, outputs = output)

# model.summary()
# model.save("./_save/keras23_08_save_model_California.h5")

import time
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723 : 문자열형태로 출력된다!
print(date) #2022-07-07 17:21:36.266674 : 현재시간

filepath = './_ModelCheckPoint/k26_2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath,'k26_',date,'_',filename]) # ""< 처음에 빈공간을 만들어주고 join으로 문자열을 묶어줌
                      )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=50, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
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

loss :  0.5541903972625732
r2 score :  0.576935250102395

'''
