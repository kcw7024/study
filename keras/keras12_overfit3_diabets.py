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
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

import matplotlib
import matplotlib.pyplot as plt #그려보자~
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='_', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='_', c='blue', label='val_loss')
plt.grid()
plt.title('loss, val_loss 값 비교')
plt.ylabel('loss')
plt.xlabel('epochs') #횟수당
#plt.legend(loc='upper right') #label 값 명칭의 위치
plt.legend()
plt.show()


#기존값
#loss :  2139.052490234375

#validation 사용