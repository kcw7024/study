#과적합 
from pickletools import optimize
from tabnanny import verbose

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split  # 훈련용과 테스트용 분리하는 모듈
from sklearn.datasets import load_boston

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

# 2. 모델

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

import time
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=1,
                 validation_split=0.2,
                 verbose=1                 
                 )

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#print("~" * 70)
#print(hist)  # <tensorflow.python.keras.callbacks.History object at 0x0000016F6E6F6F40>
#print("~" * 70)
#print(hist.history) #훈련의 결과값을 모두 볼 수 있다.
#{'loss': [422.9698486328125, 113.11389923095703, 89.71373748779297, 101.33758544921875, 91.11957550048828, 
# 84.3913345336914, 80.44567108154297, 82.64938354492188, 74.48269653320312, 70.34647369384766, 70.6878890991211], 
#'val_loss': [115.74527740478516, 62.94166564941406, 76.85604858398438, 66.15141296386719, 64.16177368164062, 168.2118377685547, 
# 122.81727600097656, 56.76030349731445, 72.63927459716797, 68.89613342285156, 60.90849304199219]}
#print("~" * 70)
#print(hist.history['loss']) #loss만 출력
#print("~" * 70)
#print(hist.history['val_loss']) #val_loss만 출력

print('걸린시간 :', end_time)

import matplotlib
import matplotlib.pyplot as plt #그려보자~
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss, val_loss 값 비교')
plt.ylabel('loss')
plt.xlabel('epochs') #횟수당
#plt.legend(loc='upper right') #label 값 명칭의 위치
plt.legend()
plt.show()


#기존결과값
#loss :  18.128032684326172

#validation 사용
