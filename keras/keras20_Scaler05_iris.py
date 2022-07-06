import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler



# weight의 난수
#import tensorflow as tf
#tf.random.set_seed(66) 



#1. 데이터

datasets = load_iris()
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target


#print(x.shape, y.shape) #(150, 4) (150,)
#print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 :  [0 1 2]

#1-1. 데이터 전처리!!!

#1-2. keras 를 이용한 Onehotincoding
#keras의 utils 에서 to_categorical을 이용한다
from tensorflow.keras.utils import to_categorical
y = to_categorical(y) #단어 텍스트를 정수 시퀀스로 변환한다.
#print(y)
#print(y.shape) # (150, 3) < 확인.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)


#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행




#print(y_train)
#print(y_test)

# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=4, activation='linear'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))


# sigmoid 0~1 사이
# softmax 다중분류에서 사용(마찬가지로 중간레이어에서는 사용할수없고, 아웃풋에서만 사용가능)
# 현재 예시에서 마지막 노드 값이 3가지. 마지막 노드 값은 데이터에따라 바뀐다
# 결과값의 총 합이 1이된다
# ex) 출력값 70,20,10 일때 -> [0.7,0.2,0.1]




# 3. 컴파일, 훈련
model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
              loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
              optimizer='adam', 
              metrics=['accuracy'] 
              )


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time



# # 4. 평가, 예측

# 첫번째 방법
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss )
# print('acc : ', acc)

# 두번째 방법
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
#print('accuracy : ', results[1])

print('걸린시간 :', end_time)
#print("#" * 80)
#print(y_test[:5])
#print("#" * 80)
y_pred = model.predict(x_test[:5])
#print(y_pred)
#print("#"*15 + "pred" + "#"*15)

'''
################################################################################
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
################################################################################
[[4.9661531e-07 9.9999928e-01 2.7846119e-07]
 [1.6254176e-06 9.9993074e-01 6.7580280e-05]
 [5.0330500e-06 9.9961519e-01 3.7977946e-04]
 [1.0000000e+00 1.0639704e-08 2.1361866e-29]
 [8.9578879e-07 9.9999309e-01 5.9722643e-06]]
################################################################################

'''

# 2. argmax 사용
# y_pred = np.argmax(y_test, axis =1)
# #print(y_test)
# y_pred = to_categorical(y_pred)
# #print(y_pred)
# acc2 = accuracy_score(y_test, y_pred)
# print("acc : ", acc2)


#풀이해주신것
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
#print(y_predict)
y_test = np.argmax(y_test, axis=1)
#print(y_test)

acc = accuracy_score(y_test, y_predict)
print("acc 스코어 : ", acc)




'''

1. 스케일러 하기전

loss :  0.05754564702510834
걸린시간 : 15.824750185012817
acc 스코어 :  1.0

2. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  0.3455403745174408
걸린시간 : 15.768676042556763
acc 스코어 :  0.96666666666666675

3. Standard Scaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 처곧를 보장할 수 없다.)

loss :  1.0016261339187622
걸린시간 : 16.1255145072937
acc 스코어 :  0.9

4. MaxAbsSacler (절대값이 0~1 사이에 매핑되도록 하는 것. 양수데이터로만 구성된 특징 
데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)

loss :  0.04309871047735214
걸린시간 : 15.906254053115845
acc 스코어 :  1.0

5. RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
* IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.

loss :  2.0109810829162598
걸린시간 : 16.117822647094727
acc 스코어 :  0.9


'''
