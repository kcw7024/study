#예제데이터를 사용
from pickletools import optimize
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#from sklearn.datasets import load_diabetes
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target


#print(x) 
#print(y) #비교값이기때문에 전처리해줄필요가 X

#print(x.shape, y.shape)  #(442, 10) (442,) 
#print(datasets.feature_names)
#print(datasets.DESCR) #데이터셋에 대한 설명


#[실습] 아래를 완성할것
# R2 0.62이상


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)


#2. 모델

model = Sequential()
model.add(Dense(300, input_dim=10))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss :  2139.052490234375
#r2스코어 :  0.6429479087939218
