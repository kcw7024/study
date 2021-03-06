import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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
scaler = MaxAbsScaler()
#scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행




#print(y_train)
#print(y_test)

# 2. 모델

# model = Sequential()
# model.add(Dense(100, input_dim=4, activation='linear'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))


input = Input(shape=(4,))
dense1 = Dense(100, activation='relu')(input)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(100, activation='softmax')(dense4)
output = Dense(3)(dense5)

model = Model(inputs = input, outputs = output)

model.summary()
model.save("./_save/keras23_011_save_model_iris.h5")

# sigmoid 0~1 사이
# softmax 다중분류에서 사용(마찬가지로 중간레이어에서는 사용할수없고, 아웃풋에서만 사용가능)
# 현재 예시에서 마지막 노드 값이 3가지. 마지막 노드 값은 데이터에따라 바뀐다
# 결과값의 총 합이 1이된다
# ex) 출력값 70,20,10 일때 -> [0.7,0.2,0.1]




# 3. 컴파일, 훈련
# model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
#               loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
#               optimizer='adam', 
#               metrics=['accuracy'] 
#               )


# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

# import time

# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
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
# #print('accuracy : ', results[1])

# print('걸린시간 :', end_time)
# #print("#" * 80)
# #print(y_test[:5])
# #print("#" * 80)
# y_pred = model.predict(x_test[:5])
# #print(y_pred)
# #print("#"*15 + "pred" + "#"*15)

# '''
# ################################################################################
# [[0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]]
# ################################################################################
# [[4.9661531e-07 9.9999928e-01 2.7846119e-07]
#  [1.6254176e-06 9.9993074e-01 6.7580280e-05]
#  [5.0330500e-06 9.9961519e-01 3.7977946e-04]
#  [1.0000000e+00 1.0639704e-08 2.1361866e-29]
#  [8.9578879e-07 9.9999309e-01 5.9722643e-06]]
# ################################################################################

# '''

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
# y_predict = np.argmax(y_predict, axis=1)
# #print(y_predict)
# y_test = np.argmax(y_test, axis=1)
# #print(y_test)

# acc = accuracy_score(y_test, y_predict)
# print("acc 스코어 : ", acc)




'''

#220707, model을 변경하여 적용하고 결과비교하기


1. 모델변경전

loss :  0.06984784454107285
걸린시간 : 15.93791151046753
acc 스코어 :  0.9666666666666667

2. 모델변경후

loss :  5.910277843475342
걸린시간 : 13.698678016662598
acc 스코어 :  0.36666666666666664

3. MaxAbsSacler (절대값이 0~1 사이에 매핑되도록 하는 것. 양수데이터로만 구성된 특징 
데이터셋에서는 MinMax와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.)

loss :  5.3726983070373535
걸린시간 : 14.124694108963013
acc 스코어 :  0.5333333333333333


'''






