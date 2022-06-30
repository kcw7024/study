#validation, 검증
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#train 훈련 test 평가 val 검증

#1. 데이터 
x = np.array(range(1, 17))
y = np.array(range(1, 17))

#[실습] train_test_split로만 나눠라.
#10:3:3으로 나눌것

#일단 10:6으로 나눠준다.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=66
    )



#print(x_train, y_train, x_test, y_test, x_val, y_val)


#2. 모델

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25 # train데이터의 25%를 validation으로 사용한다
          )
#train으로 훈련, valdata로 검증

#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


#loss < val_loss 가 맞는것
#과적합의 이유도 있음.











