
import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))     #[0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)


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



# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=13, activation='linear'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
              loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
              optimizer='adam', 
              metrics=['accuracy'] 
              )

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=550, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )


# # 4. 평가, 예측

# 첫번째 방법
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss )
# print('acc : ', acc)

# 두번째 방법
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


#print("#" * 80)
#print(y_test[:5])
#print("#" * 80)
y_pred = model.predict(x_test)
#print(y_pred)
#print("#"*15 + "pred" + "#"*15)


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

loss :  0.2294347584247589
accuracy :  0.8888888955116272
acc 스코어 :  0.8888888888888888

2. MinMaxScaler (모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는경우엔 변환된 값이 매우 좁은 범위로 압축 될 수 있음. 
MinMaxSacler역시 아웃라이어의 존재에 매우 민감.)

loss :  0.18783718347549438
accuracy :  0.9722222089767456
acc 스코어 :  0.9722222222222222

3. Standard Scaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 처곧를 보장할 수 없다.)

loss :  0.08004675805568695
accuracy :  0.9722222089767456
acc 스코어 :  0.9722222222222222

4. MaxAbsSacler (절대값이 0~1 사이에 매핑되도록 하는 것. 양수데이터로만 구성된 특징 
데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)

loss :  0.0006286170100793242
accuracy :  1.0
acc 스코어 :  1.0

5. RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
* IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.

loss :  0.06003196910023689
accuracy :  0.9722222089767456
acc 스코어 :  0.9722222222222222

'''
