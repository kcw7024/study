#1. 데이터

import numpy as np

x = np.array([1,2,3]) #input 레이어에 들어가는 형태 
y = np.array([1,2,3])

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() #순차적모델
model.add(Dense(4, input_dim=1)) #add:층을 쌓아감
model.add(Dense(5)) #숫자가 하나의 노드, 4가input 5가output
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') 
#mse=평균제곱오차사용, optimizer=나온로스값율최적화
#절대값사용=mae
model.fit(x,y, epochs=360)

#4. 평가, 예측
loss = model.evaluate(x, y) #평가 
print('loss : ', loss)

result = model.predict([4]) #예측
print('4의 예측값은 : ', result)
#훈련량조절, 노드조절, 레이어조절, 등등으로 조절 가능
#타이포 파라미터 튜닝
#가장 중요한것 - 데이터 전처리 / 후에 모델링

# loss :  3.886387275997549e-06
# 4의 예측값은 :  [[3.9959924]]


 








