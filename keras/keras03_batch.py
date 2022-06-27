import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

# 실습 [6]을 예측해볼것. 

# 2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1200, batch_size=1)

# 4. 평가, 예측 

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6]) #6을 예측한다
print('6의 예측값은 : ', result)

# loss :  0.41261619329452515
# 6의 예측값은 :  [[5.9966655]]


