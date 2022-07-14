
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, GRU


a = np.array(range(1, 101))

x_predict = np.array(range(96, 106))

size = 5  # x는 4개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)   

x = bbb[:, :-1]
y = bbb[:, -1]

ccc = split_x(x_predict, 4) # 4개씩 잘라준다

print(x, y)
print(x.shape, y.shape)   

# x = x.reshape(96, 4, 1) 

print(ccc)

#2. 모델구성

model = Sequential()
model.add(Dense(100, input_shape=(4, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=128)




#4. 평가,예측
loss = model.evaluate(x, y)
x_pred = ccc
result = model.predict(x_pred)
print('loss : ', loss)
print('결과 : ', result)


'''

LSTM

loss :  0.00020310225954744965
결과 : 

[[ 99.93752 ]
 [100.755486]
 [101.56585 ]
 [102.338715]
 [103.08333 ]
 [103.80364 ]
 [104.51217 ]]
 
 
DNN 

loss :  0.0006726059946231544
결과 :  [[100.000824]
 [101.00105 ]
 [102.0013  ]
 [103.00153 ]
 [104.001755]
 [105.002   ]
 [106.002235]]
 
loss :  0.0005023134290240705
결과 :  [[100.05436]
 [101.05624]
 [102.0581 ]
 [103.05997]
 [104.06183]
 [105.06371]
 [106.06557]]

loss :  0.00044799596071243286
결과 :  [[100.04672]
 [101.04763]
 [102.04853]
 [103.04943]
 [104.05034]
 [105.05124]
 [106.05214]]

'''