
#데이콘 따릉이 문제풀이

from pickletools import optimize
import numpy as np
import pandas as pd #csv 파일 사용시 주로 사용함
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

print(x_train.shape, x_test.shape) # (1314, 9) (14, 9)

x_train = x_train.reshape(1314, 3, 3, 1)
x_test = x_test.reshape(14, 3, 3, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(100, kernel_size=(1,1), activation='relu', input_shape=(3,3,1)))
model.add(Conv2D(200, (1,1),activation='relu'))
model.add(Conv2D(100, (1,1),activation='relu'))
model.add(Conv2D(300, (1,1),activation='relu'))
model.add(Conv2D(400, (1,1),activation='relu'))
model.add(Conv2D(100, (1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(1))


import time

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time


#4. 평가, 예측
print(('#'*70) + '1.기본출력')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)


'''
loss :  809.9908447265625
r2 score :  0.8145061942737434

'''