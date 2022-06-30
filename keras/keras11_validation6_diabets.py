#예제데이터를 사용
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72 #72 
                                                    )


#2. 모델구성

model = Sequential()
model.add(Dense(200, input_dim=10))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(180))
model.add(Dense(170))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, epochs=195, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#loss :  2139.052490234375
#r2스코어 :  0.6429479087939218

# validation 사용
#loss :  2206.111572265625
#r2스코어 :  0.6659035907133195