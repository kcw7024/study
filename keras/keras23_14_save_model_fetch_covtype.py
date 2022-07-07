
#keras18_gpu_test3파일의 summary를 확인
#summary와 time


import numpy as np
from sklearn.datasets import fetch_covtype
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


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


#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

# 2. 모델

# model = Sequential()
# model.add(Dense(100, input_dim=54, activation='linear'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(7, activation='softmax'))


input = Input(shape=(54,))
dense1 = Dense(100, activation='relu')(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(300, activation='relu')(dense4)
dense6 = Dense(200, activation='relu')(dense5)
dense7 = Dense(300, activation='relu')(dense6)
dense8 = Dense(200, activation='relu')(dense7)
dense9 = Dense(100, activation='relu')(dense8)
output = Dense(7, activation='softmax')(dense9)


model = Model(inputs = input, outputs = output)


model.summary()
model.save("./_save/keras23_014_save_model_fetch_covtype.h5")

# 3. 컴파일, 훈련
# model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
#               loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
#               optimizer='adam', 
#               metrics=['accuracy'] 
#               )

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs=10, 
#                  batch_size=64,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1                 
#                  )

# end_time = time.time() - start_time

# # # 4. 평가, 예측

# # 첫번째 방법
# # loss, acc = model.evaluate(x_test, y_test)
# # print('loss : ', loss )
# # print('acc : ', acc)

# # 두번째 방법
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])

# print('걸린시간 :', end_time)

# # print("#" * 80)
# # print(y_test[:5])
# # print("#" * 80)
# #y_predict = model.predict(x_test)
# #print(y_predict)
# # print("#"*15 + "pred" + "#"*15)


# # 2. argmax 사용
# # y_pred = np.argmax(y_test, axis =1)
# # #print(y_test)
# # y_pred = to_categorical(y_pred)
# # #print(y_pred)
# # acc2 = accuracy_score(y_test, y_pred)
# # print("acc : ", acc2)


# #풀이해주신것
# from sklearn.metrics import accuracy_score

# y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict, axis=1)
# #print(y_predict)
# y_test = tf.argmax(y_test, axis=1)
# #print(y_test)

# acc = accuracy_score(y_test, y_predict)
# print("acc 스코어 : ", acc)

# #model.summary()



'''

#220707, model을 변경하여 적용하고 결과비교하기


1. 모델변경전

loss :  0.3983382284641266
accuracy :  0.8331798911094666
걸린시간 : 367.76929783821106
acc 스코어 :  0.8331798662685128

2. 모델변경후

loss :  0.3451073169708252
accuracy :  0.8587988018989563
걸린시간 : 363.52514386177063
acc 스코어 :  0.8587988261920949

3.  RobustScaler (아웃라이어의 영향을 최소화 한 기법. 중앙값(median)과 IQR(interquartile range)를 사용하기 때문에 
StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있음.
* IQR = Q3 - Q1 : 25퍼센타일과 75퍼센타일의 값들을 다룸.

loss :  0.1786353439092636
accuracy :  0.9284183979034424
걸린시간 : 375.988792181015
acc 스코어 :  0.9284183712985035


'''



