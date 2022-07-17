import numpy as np
from regex import X
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Conv2D, LSTM, SimpleRNN
from sklearn.metrics import r2_score
import tensorflow as tf

#1. 데이터

x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가(0~99), 하이닉스 종가(301~400)
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)])   # 추가정보

x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape) # (100, 2) (100, 3)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301)) # 환율

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
     x1, x2, x3, y1, y2, train_size=0.7, random_state=66
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
output1 = Dense(100, activation='relu', name='out_model1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3, ))
dense11 = Dense(110, activation='relu', name='k11')(input2)
dense12 = Dense(120, activation='relu', name='k12')(dense11)
dense13 = Dense(130, activation='relu', name='k13')(dense12)
dense14 = Dense(140, activation='relu', name='k14')(dense13)
output2 = Dense(100, activation='relu', name='out_model2')(dense14)

#2-3. 모델3
input3 = Input(shape=(2, ))
dense21 = Dense(110, activation='relu', name='k21')(input3)
dense22 = Dense(120, activation='relu', name='k22')(dense21)
dense23 = Dense(130, activation='relu', name='k23')(dense22)
dense24 = Dense(140, activation='relu', name='k24')(dense23)
output3 = Dense(100, activation='relu', name='out_model3')(dense24)

from tensorflow.python.keras.layers import concatenate, Concatenate   # 여러개의 범위 또는 여러개의 텍스트 문자열을 하나의 텍스트 문자열로 연결하는 함수

#각모델들의 아웃풋을 합쳐서 레이어가 만들어짐
# Concatenate
#merge1 = concatenate([output1, output2, output3], name='mg1')

merge1 = tf.keras.layers.Concatenate()([output1, output2, output3])

# 방법1
# merge2 = Dense(200, activation='relu', name='mg2')(merge1)
# merge3 = Dense(300, name='mg3')(merge2)
# last_output1 = Dense(1)(merge3)
# last_output2 = Dense(1)(merge3)

merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, name='mg3')(merge2)
last_output = Dense(1)(merge3)


# 방법2
#2-4. y모델1
output41 = Dense(32)(last_output)
output42 = Dense(10)(output41)
last_output2 = Dense(1)(output42)

#2-5. y모델2
output51 = Dense(32)(last_output)
output52 = Dense(10)(output51)
last_output3 = Dense(1)(output52)


model = Model(inputs=[input1, input2, input3], outputs=[last_output2, last_output3])


model.summary()

'''

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 2)]          0
__________________________________________________________________________________________________
kcw11 (Dense)                   (None, 11)           33          input_1[0][0]
__________________________________________________________________________________________________
kcw1 (Dense)                    (None, 1)            3           input_1[0][0]
__________________________________________________________________________________________________
kcw12 (Dense)                   (None, 12)           144         kcw11[0][0]
__________________________________________________________________________________________________
kcw2 (Dense)                    (None, 2)            4           kcw1[0][0]
__________________________________________________________________________________________________
kcw13 (Dense)                   (None, 13)           169         kcw12[0][0]
__________________________________________________________________________________________________
kcw3 (Dense)                    (None, 3)            9           kcw2[0][0]
__________________________________________________________________________________________________
kcw14 (Dense)                   (None, 14)           196         kcw13[0][0]
__________________________________________________________________________________________________
out_model1 (Dense)              (None, 10)           40          kcw3[0][0]
__________________________________________________________________________________________________
out_model2 (Dense)              (None, 10)           150         kcw14[0][0]
__________________________________________________________________________________________________
mg1 (Concatenate)               (None, 20)           0           out_model1[0][0]
                                                                 out_model2[0][0]
__________________________________________________________________________________________________
mg2 (Dense)                     (None, 2)            42          mg1[0][0]
__________________________________________________________________________________________________
mg3 (Dense)                     (None, 3)            9           mg2[0][0]
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
last (Dense)                    (None, 1)            4           mg3[0][0]
==================================================================================================
Total params: 803
Trainable params: 803
Non-trainable params: 0
__________________________________________________________________________________________________

'''


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, batch_size=128)
 
#4. 평가, 예측
loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)

y_predict = model.predict([x1_test, x2_test, x3_test])

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





