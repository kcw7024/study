#예제데이터를 사용
from pickletools import optimize
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
    x, y, train_size=0.7, random_state=66
)


#2. 모델

model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(90))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=900, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss :  0.6370521187782288
#r2스코어 :  0.5357340087180074

