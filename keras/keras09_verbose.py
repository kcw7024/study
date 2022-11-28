#예제데이터를 사용
from pickletools import optimize
from tabnanny import verbose

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split #훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import load_boston

'''


'''
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

import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#현재시간 출력(Unix timestamp로 출력됨)
start_time = time.time() #1656033228.2365258
print(start_time)
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)
#verbose = 파라미터를 사용
end_time = time.time() - start_time

print('걸린시간 : ', end_time)

'''

verbose 0 걸린시간 : 14.956664562225342 / 출력없음
verbose 1 걸린시간 : 15.13918662071228 / 기존과 같은 출력이 있음
verbose 2 걸린시간 : 14.565986156463623 / 프로그래스바가 없음
verbose 3,4,5... 걸린시간 : 13.68128252029419 / epoch만 나옴



'''

