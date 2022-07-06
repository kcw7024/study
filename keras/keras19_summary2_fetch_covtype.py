#keras18_gpu_test3파일의 summary를 확인
#summary와 time


import numpy as np
from sklearn.datasets import fetch_covtype
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import time


#1. 데이터

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,) 
print(np.unique(y, return_counts=True))     #[1 2 3 4 5 6 7]
#(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)
#print(y)

#pandas로 원핫코딩작업
y = pd.get_dummies(y)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=54, activation='linear'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
              loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
              optimizer='adam', 
              metrics=['accuracy'] 
              )

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, 
                 batch_size=64,
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
print('accuracy : ', results[1])

print('걸린시간 :', end_time)

# print("#" * 80)
# print(y_test[:5])
# print("#" * 80)
#y_predict = model.predict(x_test)
#print(y_predict)
# print("#"*15 + "pred" + "#"*15)


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
y_predict = tf.argmax(y_predict, axis=1)
#print(y_predict)
y_test = tf.argmax(y_test, axis=1)
#print(y_test)

acc = accuracy_score(y_test, y_predict)
print("acc 스코어 : ", acc)

model.summary()

'''

epochs = 2

GPU
걸린시간 : 27.442648887634277

CPU
걸린시간 : 11.278942823410034

epochs = 100

CPU
걸린시간 : 564.0292217731476

GPU
걸린시간 : 1335.0220313072205


07/06

CPU
걸린시간 : 97.43256640434265

GPU
걸린시간 : 176.20923447608948


batchsize 128
Total params: 408,007
Trainable params: 408,007
batchsize 64


'''