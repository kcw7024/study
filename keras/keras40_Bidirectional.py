import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU
from tensorflow.keras.layers import Bidirectional

#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #시계열데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)
# x의_shape = (형, 열, 몇개씩 자르는지!) :  RNN 3차원화 해주는것
x = x.reshape(7, 3, 1)
print(x.shape) #(7, 3, 1) 


x_predict = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(5, return_sequences=True), input_shape=(3, 1))) #양방향으로 돌아간다.
#model.add(Bidirectional(SimpleRNN(64), input_shape=(3 ,1)))
model.add(SimpleRNN(10)) 
#model.add(Bidirectional(SimpleRNN(5))) #양방향으로 돌아간다.
#RNN은 2차원으로 던져준다(Flatten없이 Dense로 받을수 있음)
#model.add(SimpleRNN(32))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = x_predict.reshape(1, 3, 1) # [[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)


'''
DNN : unit * (feature + bias) = param
simpleRNN :  unit * (feature+bias+unit) = param / 연산을 한번 더 한다.
LSTM : 4 * unit * (feature+bias+unit) = param / = simpleRNN * 4
LSTM 의 숫자 4의 의미는 cell state, input gate, output gate, forget gate
GRU : 3 * unit * (feature + bias + unit) : simpleRNN * 3
GRU 숫자 3 의미는 hidden state, reset gate, update gete
Bidirectional ( simpleRNN :  unit * (feature+bias+unit) * 2 ) = param
'''







