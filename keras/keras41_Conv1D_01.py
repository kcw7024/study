import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, LSTM, Conv1D, Flatten
# from tensorflow.keras.layers import Bidirectional

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
#model.add(LSTM(10, return_sequences=False, input_shape=(3, 1))) #양방향으로 돌아간다.
model.add(Conv1D(10, 2, input_shape=(3 ,1)))
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(1))


#model.summary() #LSTM : 517 / Conv1D : 97


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = x_predict.reshape(1, 3, 1) # [[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10]의 결과 : ', result)





