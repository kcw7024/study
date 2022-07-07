#과적합 
from imp import load_module
from pickletools import optimize
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행

# 2. 모델

model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
#model.summary()

import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5'
                      )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=50, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1                 
                 )

end_time = time.time() - start_time

model.save('./_save/kares24_3_save_model.h5')

#4. 평가, 예측
print(('*'*70) + '1.기본출력')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)

print(('*'*70) + '2. load_model 출력')

model2 = load_model('./_save/kares24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2 스코어 : ', r2)


print(('*'*70) + '3. ModelCheckpoint 출력')

model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3)
print('r2 스코어 : ', r2)

'''

**********************************************************************1.기본출력
4/4 [==============================] - 0s 3ms/step - loss: 26.5202
loss :  26.520246505737305
r2 score :  0.6753746543634477
**********************************************************************2. load_model 출력
4/4 [==============================] - 0s 2ms/step - loss: 26.5202
loss2 :  26.520246505737305
r2 스코어 :  0.6753746543634477
**********************************************************************3. ModelCheckpoint 출력
4/4 [==============================] - 0s 2ms/step - loss: 26.5202
loss3 :  26.520246505737305
r2 스코어 :  0.6753746543634477


'''