import numpy as np
from regex import X
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score

#1. 데이터

x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가(0~99), 하이닉스 종가(301~400)
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # 금리

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
     x1, x2, y, train_size=0.7, random_state=66
)

#print(x1_train.shape, x2_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)
#print(x1_test.shape, x2_test.shape, y_test.shape) #(30, 2) (30, 3) (30,)

#2. 모델
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1 
input1 = Input(shape=(2, ))
dense1 = Dense(100, activation='relu', name='kcw1')(input1)
dense2 = Dense(200, activation='relu', name='kcw2')(dense1)
dense3 = Dense(300, activation='relu', name='kcw3')(dense2)
output1 = Dense(100, activation='relu', name='out_model1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3, ))
dense11 = Dense(110, activation='relu', name='kcw11')(input1)
dense12 = Dense(120, activation='relu', name='kcw12')(dense11)
dense13 = Dense(130, activation='relu', name='kcw13')(dense12)
dense14 = Dense(140, activation='relu', name='kcw14')(dense13)
output2 = Dense(100, activation='relu', name='out_model2')(dense14)


from tensorflow.python.keras.layers import concatenate, Concatenate   # 여러개의 범위 또는 여러개의 텍스트 문자열을 하나의 텍스트 문자열로 연결하는 함수

#각모델들의 아웃풋을 합쳐서 레이어가 만들어짐
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)


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
model.fit([x1_train, x2_train], y_train, epochs=15000, batch_size=128)
 
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('predict :', y_predict)
print('loss : ', loss)
print('결과 : ', r2)


'''

loss :  0.17710603773593903
결과 :  0.9997974680456038

loss :  15.21438217163086
결과 :  0.9825994589684286

loss :  1.5179283618927002
결과 :  0.9982639181872793

loss :  0.03583908826112747
결과 :  0.9999590266439743

loss :  0.07896331697702408
결과 :  0.9999097033957715

loss :  0.0016117180930450559
결과 :  0.999998153189266

'''





