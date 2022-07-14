
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

x = x.reshape(96, 4, 1) 

print(ccc)

#2. 모델구성

model = Sequential()
model.add(LSTM(units=100, input_shape=(4,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=128)




#4. 평가,예측
loss = model.evaluate(x, y)
x_pred = ccc.reshape(7, 4, 1)
result = model.predict(x_pred)
print('loss : ', loss)
print('결과 : ', result)


'''


loss :  0.00020310225954744965
결과 : 

[[ 99.93752 ]
 [100.755486]
 [101.56585 ]
 [102.338715]
 [103.08333 ]
 [103.80364 ]
 [104.51217 ]]
 
loss :  0.00031687048613093793
결과 :  [[ 99.93952]
 [100.86   ]
 [101.7048 ]
 [102.53304]
 [103.32476]
 [104.07534]
 [104.78636]]


'''