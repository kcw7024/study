from pickletools import optimize
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터

#원데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#훈련용과 테스트용으로 분리
#x_train = np.array([1,2,3,4,5,6,7]) #훈련용
#x_test = np.array([8,9,10]) #테스트용
#y_train = np.array([1,2,3,4,5,6,7])
#y_test = np.array([8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, 
    train_size=0.7, # 원하는 비율로 나눈다. test size와 train size중 하나의값만 지정해도 정상작동
    #shuffle=False, # 데이터셋을 섞는다. 
    random_state=66 # 랜던값 고정
    
    #train_test_split 함수의 기본은 shuffle=True
)

#print(x_train) #[2 7 6 3 4 8 5]
#print(x_test) #[ 1  9 10]
#print(y_train)
#print(y_test)


#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)


#loss :  0.07892820984125137
#11의 예측값 :  [[10.591753]]