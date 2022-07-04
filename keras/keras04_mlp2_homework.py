from pickletools import optimize
import numpy as np
from sklearn.metrics import log_loss
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array(
             [[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]]
             )
y = np.array(
             [11,12,13,14,15,16,17,18,19,20]
             )

#x = np.transpose(x) # 방법 2

#print(x.shape) #(2,10)
#print(y.shape) #(10, )

x = x.T #x를 전치한다. 방법 1
#x = x.reshape(10,2) 방법 3 순서가 유지됨

print(x)
print(x.shape)

# [숙제] 모델을 완성하시오

'''레이어'''

model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1300, batch_size=3)

# 예측 : [[10 ,1.4, 0]]

loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10,1.4,0]])
print('[10, 1.4, 0]의 예측값 :', result)

#loss :  5.862551006430294e-06
#[10, 1.4, 0]의 예측값 : [[19.999092]]

