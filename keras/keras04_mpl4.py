#2022.06.21 range 함수
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#.1 데이터

x = np.array([range(10)]) #범위 함수

#print(range(10))
#for i in range(10): 
#     print(i)

#print(x.shape) #(1, 10)

#x = x.T # 반전
x = np.transpose(x)

#print(x.shape) #(10, 1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]
             )

y = np.transpose(y)

#print(y.shape)


#2. 모델

#[실습]

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(3)) #out_put 갯수 


# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1700, batch_size=3)

#3. 예측 : [[9]] 예상 y값 [[10, 1.9, 0]]
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9]])
print('[9]의 예측값 :', result)

#loss :  5.0035341701004654e-05
#[9]의 예측값 : [[10.004158    1.8999274   0.01380324]]

