#예제데이터를 사용
from pickletools import optimize
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


#print(x) 
#print(y) 

#print(x.shape, y.shape)  #(20640, 8) (20640,) 
#print(datasets.feature_names)
#print(datasets.DESCR) #데이터셋에 대한 설명


#[실습] 아래를 완성할것
# 1.train 0.7
# 2. R2 0.8이상


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)


#2. 모델

model = Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=510, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

import matplotlib
import matplotlib.pyplot as plt #그려보자~
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='_', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='_', c='blue', label='val_loss')
plt.grid()
plt.title('loss, val_loss 값 비교')
plt.ylabel('loss')
plt.xlabel('epochs') #횟수당
#plt.legend(loc='upper right') #label 값 명칭의 위치
plt.legend()
plt.show()


#기존값
#loss :  0.5925426483154297

#validation 사용
#loss :  0.5953660011291504