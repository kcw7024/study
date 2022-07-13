import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)
# x의_shape = (형, 열, 몇개씩 자르는지!) :  RNN 3차원
x = x.reshape(7, 3, 1)
print(x.shape) #(7, 3, 1) 

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1))) #RNN은 2차원으로 던져준다(Flatten없이 Dense로 받을수 있음)
# #model.add(SimpleRNN(32))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))

#관련 자세한건 keras.io
#model.add(SimpleRNN(units=10, input_shape=(3, 1))) # [batch, timesteps, feature(input_dim)]
#RNN은 2차원으로 던져준다(Flatten없이 Dense로 받을수 있음)
#model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
#model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
model.add(LSTM(units=200, input_shape=(3, 1))) # [batch, timesteps, feature(input_dim)]
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(1))


model.summary()

# unit = output node의 갯수

# param 값 구하기
# DNN : unit * (feature + bias) = param
# simpleRNN :  unit * (feature+bias+unit) = param / 연산을 한번 더 한다.
# LSTM : 4 * unit * (feature+bias+unit) = param / = simpleRNN * 4
# LSTM 의 숫자 4의 의미는 cell state, input gate, output gate, forget gate


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120          #   (unit*unit) + (input_dim(feature) * unit) + (unit*b(1))
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=700, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)

