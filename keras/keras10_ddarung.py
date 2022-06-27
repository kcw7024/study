#데이콘 따릉이 문제풀이

from pickletools import optimize
import numpy as np
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터

path = './_data/ddarung/' #경로 정의
train_set = pd.read_csv(path + 'train.csv', # 훈련함수를 정의, 데이터를 삽입
                        index_col=0
                        ) 
# print(train_set)
# print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 사용
                       index_col=0
                       )

# print(test_set)
# print(test_set.shape) #(715, 9)

## pandas제공 기본기능 ##
# print(train_set.columns) #컬럼명을 확인
# print(train_set.info()) #각 칼럼에 대한 상세정보 확인 *non-null:null값이 없다 #null=결측치
# print(train_set.describe()) #데이터 설명


# 결측치 처리 1. 제거 #

# print(train_set.isnull().sum()) #null의 갯수(합계)를 구한다
train_set = train_set.dropna() #null값이 있는 행을 삭제
#print(train_set.isnull().sum()) #null의 갯수(합계)를 구한다
#print(train_set.shape) # (1328, 10)

x = train_set.drop(['count'], axis=1) #count라는 컬럼을 drop한다.
#print(x)
#print(x.columns)
#print(x.shape) #(1459, 9)

y = train_set['count']
#print(y)
#print(y.shape) #(1459, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.99, random_state=750
    )


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2222, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) #test로 평가
print('loss : ', loss)

y_predict = model.predict(x_test)

#RMSE 함수정의, 사용
def RMSE(y_test, y_predict): #mse에 루트를 씌운다.
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 데이터 수정전, 데이터 안에 null값이 있으면 loss 값이 nan이 나옴


'''
1. train_size를 0.7->0.9로 늘렸을 경우,
random_state를 777로 늘렸을경우,
RMSE 수치가 49에서 아래와 같이 낮아졌음

loss :  2215.654541015625
RMSE :  47.07073949571291

2. 다른조건들은 동일하되, random_state를 750으로 
조정하였을때 RMSE 수치가 47에서 45로 낮아졌음

loss :  2114.784423828125
RMSE :  45.986785517117845

3. 다른조건들은 동일하되, epochs를 888로 변경하였을때
아래와 같이 미세하게 수치가 낮아졌음

#1
loss :  2070.074462890625
RMSE :  45.49806849664057
#2
loss :  2041.0472412109375
RMSE :  45.177950848307844

4. train_size=0.99, random_state=777, epochs=300

loss :  1037.4530029296875
RMSE :  32.20951999176865

5. epochs만 조절

#400
loss :  944.0969848632812
RMSE :  30.726162289416347

#100 
loss :  902.3173217773438
RMSE :  30.03860354084897

#300
loss :  901.3604736328125
RMSE :  30.02267188530714

#500
loss :  790.4541625976562
RMSE :  28.115020603301012

#510
loss :  781.6456298828125
RMSE :  27.95792743207647



6. activation 함수 사용
random_state=750, epochs=500

loss :  681.2056274414062
RMSE :  26.09991689722984

**결과값의 편차가 너무 커서 신뢰도가 낮음



'''