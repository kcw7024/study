


#과적합 
from pickletools import optimize
from tabnanny import verbose
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split  # 훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)



scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행



# 2. 모델

# model = Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))


input = Input(shape = (8,))
dense1 = Dense(100)(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(300, activation='relu')(dense3)
dense5 = Dense(100, activation='relu')(dense4)
dense6 = Dense(200, activation='relu')(dense5)
dense7 = Dense(100, activation='relu')(dense6)
output = Dense(1)(dense7)

model = Model(inputs = input, outputs = output)

model.summary()
model.save("./_save/keras23_08_save_model_California.h5")

import time
# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# #restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs=200, batch_size=128,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1                 
#                  )

# end_time = time.time() - start_time

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# print('걸린시간 :', end_time)

# y_predict = model.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 : ', r2)


# import matplotlib
# import matplotlib.pyplot as plt #그려보자~
# matplotlib.rcParams['font.family'] ='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] =False

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss, val_loss 값 비교')
# plt.ylabel('loss')
# plt.xlabel('epochs') #횟수당
# #plt.legend(loc='upper right') #label 값 명칭의 위치
# plt.legend()
# plt.show()


'''

#220707, model을 변경하여 적용하고 결과비교하기


1. 모델변경전

loss :  [0.5540508031845093, 0.5540508031845093]
걸린시간 : 26.202254056930542
r2스코어 :  0.577041859504293

2. 모델변경후

loss :  [0.5874351263046265, 0.5874351263046265]
걸린시간 : 29.387880086898804
r2스코어 :  0.5515564485517552

3. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  [0.24931497871875763, 0.24931497871875763]
걸린시간 : 49.49594736099243
r2스코어 :  0.8096747983723596


'''











