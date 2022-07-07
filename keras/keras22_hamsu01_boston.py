from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input




datatests = load_boston()
x = datatests.data
y = datatests['target']


# 잘못된 방법. 전처리는 컬럼별로 해야한다.
# print(np.min(x)) #0.0
# print(np.max(x)) #711.0
# # 1. minmaxscaler
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# #minmaxscaler는 0과 1사이로 결과값을 변환
# print(x[:10])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66    
)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()

#scaler.fit(x_train)
#print(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


# print(np.min(x_train)) #0.0
# print(np.max(x_train)) #1.0
# print(np.min(x_test))
# print(np.max(x_test))


# a = 0.1
# b = 0.2
# print(a + b)


#2. 모델구성


# model = Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))


input = Input(shape=(13,))
dense1 = Dense(100)(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
dense5 = Dense(100, activation='relu')(dense4)
output = Dense(1)(dense5)

model = Model(inputs = input, outputs = output)

import time
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print("~" * 70)
# print(hist)  # <tensorflow.python.keras.callbacks.History object at 0x0000016F6E6F6F40>
# print("~" * 70)
# print(hist.history) #훈련의 결과값을 모두 볼 수 있다.
# #{'loss': [422.9698486328125, 113.11389923095703, 89.71373748779297, 101.33758544921875, 91.11957550048828, 
# #84.3913345336914, 80.44567108154297, 82.64938354492188, 74.48269653320312, 70.34647369384766, 70.6878890991211], 
# #'val_loss': [115.74527740478516, 62.94166564941406, 76.85604858398438, 66.15141296386719, 64.16177368164062, 168.2118377685547, 
# #122.81727600097656, 56.76030349731445, 72.63927459716797, 68.89613342285156, 60.90849304199219]}
# print("~" * 70)
# print(hist.history['loss']) #loss만 출력
# print("~" * 70)
# print(hist.history['val_loss']) #val_loss만 출력

print('걸린시간 :', end_time)



y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

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

loss :  [28.50311279296875, 28.50311279296875]
걸린시간 : 4.463206768035889
r2스코어 :  0.6549974126463911

2. 모델변경후

loss :  [22.783706665039062, 22.783706665039062]
걸린시간 : 4.479897737503052
r2스코어 :  0.7242252898077506

3. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  [10.881688117980957, 10.881688117980957]
걸린시간 : 4.5386621952056885
r2스코어 :  0.8682877080520573


'''











