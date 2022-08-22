#예제데이터를 사용
from pickletools import optimize
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

#print(x) #8개의 피쳐
#print(y) #x를 사용한 예상 집값

#print(x.shape, y.shape)  #(506, 13) (506,) 
#print(datasets.feature_names)
#print(datasets.DESCR) #데이터셋에 대한 설명

#print(x_train.shape, y_train.shape) # (354, 13) (354,)
#print(x_test.shape, y_test.shape) # (152, 13) (152,)

#[실습] 아래를 완성할것
# 1.train 0.7
# 2. R2 0.8이상

#2. 모델

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss :  18.128032684326172
#r2스코어 :  0.7805777152060368