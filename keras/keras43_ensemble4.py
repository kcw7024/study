import numpy as np
from regex import X
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Conv2D, LSTM, SimpleRNN
from sklearn.metrics import r2_score
import tensorflow as tf

#1. 데이터

x_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가(0~99), 하이닉스 종가(301~400)

x = np.transpose(x_datasets)

print(x.shape) # (100, 2) (100, 3)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301)) # 환율

from sklearn.model_selection import train_test_split

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
     x, y1, y2, train_size=0.7, random_state=66
)

#print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)
#print(x1_test.shape, x2_test.shape, y_test.shape) #(30, 2) (30, 3) (30,)

#2. 모델
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1 
input1 = Input(shape=(2, ))
dense1 = Dense(100, activation='relu', name='k1')(input1)
dense2 = Dense(200, activation='relu', name='k2')(dense1)
dense3 = Dense(300, activation='relu', name='k3')(dense2)
output1 = Dense(1, activation='relu')(dense3)
output2 = Dense(1, activation='relu')(dense3)


from tensorflow.python.keras.layers import concatenate, Concatenate   # 여러개의 범위 또는 여러개의 텍스트 문자열을 하나의 텍스트 문자열로 연결하는 함수

#각모델들의 아웃풋을 합쳐서 레이어가 만들어짐
# Concatenate
#merge1 = concatenate([output1, output2, output3], name='mg1')

#merge1 = tf.keras.layers.Concatenate()([output1, output2, output3])

# 방법1
# merge2 = Dense(200, activation='relu', name='mg2')(merge1)
# merge3 = Dense(300, name='mg3')(merge2)
# last_output1 = Dense(1)(merge3)
# last_output2 = Dense(1)(merge3)

#merge2 = Dense(200, activation='relu', name='mg2')(merge1)
#merge3 = Dense(300, name='mg3')(merge2)
#last_output = Dense(1)(merge3)


# # 방법2
# #2-4. y모델1
# output41 = Dense(32)(last_output)
# output42 = Dense(10)(output41)
# last_output2 = Dense(1)(output42)

# #2-5. y모델2
# output51 = Dense(32)(last_output)
# output52 = Dense(10)(output51)
# last_output3 = Dense(1)(output52)


model = Model(inputs=input1, outputs=[output1, output2])


#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train], epochs=1000, batch_size=128)
 
#4. 평가, 예측
loss1 = model.evaluate([x_test], y1_test)
loss2 = model.evaluate([x_test], y2_test)

y_predict = model.predict([x_test])

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])


# print('predict :', y1_predict)
# print('predict :', y2_predict)
print('loss1 : ', loss1)
print('loss2 : ', loss2)
print('R2_1 : ', r2_1)
print('R2_2 : ', r2_2)

# loss1 :  [3211143.75, 10.949295997619629, 3211132.75]
# loss2 :  [3237061.75, 3236343.0, 719.123291015625]
# R2_1 :  0.9874778297571414
# R2_2 :  0.17756421428600033


#3-1. 컴파일, 훈련2

# loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])

# y_predict = model.predict([x1_test, x2_test, x3_test])
# y_predict = np.array(y_predict)
# y_test = np.array([y1_test, y2_test])

# #print(y_test.shape, y_predict.shape) #(2, 30) (2, 30, 1)

# y_predict = y_predict.reshape(2, 30)
# y_test = y_test.reshape(2, 30)

# r2 = r2_score(y_test, y_predict)
# print('loss :', loss)
# print('r2스코어 :', r2)



'''

loss1 :  [3240023.5, 0.014194067567586899, 3240023.5]
loss2 :  [3240040.5, 3240040.5, 0.006768553983420134]
R2_1 :  0.9999837664578737
R2_2 :  0.9999922590441598



'''





