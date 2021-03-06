#과적합 
from pickletools import optimize
from tabnanny import verbose
from tensorflow.python.keras.models import Sequential
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

#model.save("./_save/keras23_1_save_model.h5")
#model = load_model("./_save/keras23_3_save_model.h5")
#model.save_weights("./_save/keras23_5_save_weights1.h5")

#현재 실습에서 해당~ 
#fit 다음에 save,
#load는 compile, 모델 살리고 load

#model.load_weights("./_save/keras23_5_save_weights1.h5") #훈련시키기전의 내용이라 결과값이 구림: 앞에서 save 위치를그렇게 지정해줘서~save위치에따라 다름
#model.load_weights("./_save/keras23_5_save_weights2.h5")



import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 전에 나오는 최적값을 가져온다

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath='./_ModelCheckPoint/kerase24_ModelCheckPoint.hdf5'
                      )

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1                 
                 )

end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 score : ' , r2)

