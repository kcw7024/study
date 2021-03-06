
#데이콘 따릉이 문제풀이

from pickletools import optimize
import numpy as np
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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

#결측치를 0값으로 넣어준다. 방법은 여러가지가있음.

#print(test_set)
#test_set = test_set.replace(to_replace=np.nan, value=0)
test_set = test_set.fillna(0)
#결측치 처리, Nan값이 있는곳에 지정한 value값을 넣어준다
#test_set.head()


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



#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행



#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=9))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

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

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



# 22.06.27

#1 to_read()를 사용해서 submission.csv를 불러온다.

#result = pd.read_csv(path + 'submission.csv', index_col=0)
#index_col=0 의 의미 : index col을 없애준다.

#y_summit = model.predict(test_set)
#print(y_summit)
#print(y_summit.shape) # (715,1)

#result['count'] = y_summit
#result 에서 지정해준 submission의 count 값에 y_summit값을 넣어준다.

#.to_csv() 를 사용해서 submission.csv를 완성

#2
#result.to_csv(path + 'submission.csv', index=True)


'''

1. 스케일러 하기전

loss :  [774.9049682617188, 774.9049682617188]
RMSE :  27.83711613177755
r2스코어 :  0.8225411082448723

2. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  [521.676025390625, 521.676025390625]
RMSE :  22.840228233805437
r2스코어 :  0.8805323939540031

3. Standard Scaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 처곧를 보장할 수 없다.)

loss :  [837.9401245117188, 837.9401245117188]
RMSE :  28.947195431731217
r2스코어 :  0.8081056142078956

4. MaxAbsSacler (절대값이 0~1 사이에 매핑되도록 하는 것. 양수데이터로만 구성된 특징 
데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)

loss :  [636.8639526367188, 636.8639526367188]
RMSE :  25.236164014835538
r2스코어 :  0.8541535155654829

5. RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
* IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.

loss :  [3401.5986328125, 3401.5986328125]
RMSE :  58.323226858002165
r2스코어 :  0.8461163619215499


'''
