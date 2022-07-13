import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape, y.shape) #(13, 3) (13,)

x_predict = np.array([50,60,70])

x = x.reshape(13, 3, 1)

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10,  return_sequences=True, input_shape=(3,1))) # (N, 3, 1) -> (N, 3, 10)
#return_sequences : 차원이 하나 늘어난다.
model.add(LSTM(5, return_sequences=False))
model.add(Dense(100))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1) # [[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)