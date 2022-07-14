import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, LSTM, Conv1D, Flatten



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
model = Sequential()
model.add(Conv1D(5, 2, input_shape=(3, 1)))
#model.add(LSTM(units=500, input_shape=(3, 1), return_sequences=True)) # [batch, timesteps, feature(input_dim)]
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(1))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=128)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1) # [[[8],[9],[10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70]의 결과 : ', result)

# # loss :  4.804766376764746e-06
# # [50,60,70]의 결과 :  [[75.02797]]

# # loss :  0.004187827464193106
# # [50,60,70]의 결과 :  [[74.3389]]

# # loss :  9.204842353938147e-05
# # [50,60,70]의 결과 :  [[76.2696]]

# # loss :  0.07843225449323654
# # [50,60,70]의 결과 :  [[76.05879]]

# # loss :  0.11388034373521805
# # [50,60,70]의 결과 :  [[77.81378]]

# Conv1D

# loss :  1.9661972316953324e-07
# [50,60,70]의 결과 :  [[79.999214]]