#유방암 데이터활용
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DECR)
#print(datasets.feature_names)


x = datasets['data']
y = datasets['target']
#print(x.shape, y.shape) #(569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

# 2. 모델

model = Sequential()
model.add(Dense(10, input_dim=30, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#linear : 선형
#sigmoid : 0 과 1사이로 값을 반환해줌  (이진분류는 아웃풋에서 무조건 sigmoid)
# : 0.4 같은 숫자를 처리하기 위해 반올림 처리 해줘야함


import time
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', 'mse'] 
              )
#accuracy
#이진분류에서 사용
#정확도, 평가지표를 판단할때 사용 #loss에 정확도도 표기


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 검증중의 가장 최소값을 가져온다.
#False로 지정하면 마지막 값을 가져온다


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1200, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#print('걸린시간 :', end_time)

y_predict = model.predict(x_test)

####### 과제1. accuracy_score 완성할 것.
# #from sklearn.metrics import r2_score #결정계수, 회귀모델에서 사용
# from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test, y_predict) 
# print('r2스코어 : ', acc) 
#print(y_predict)
# relu는 히든에서만 사용가능, 아웃풋에서는 사용할수 없음. 성능이 80퍼 이상. 좋은 활성화함수



'''
#결과값
1. r2스코어 :  0.6159392330460675
2. r2스코어 :  0.6143527328408744
3. r2스코어 :  0.6256625727047345

#활성화 함수 추가한뒤에 결과값
#r2스코어 :  0.8196528959683043

#accuracy 사용시
#loss값과 metrics의 값도 함께 보여준다.
#loss :  [0.2095879167318344, 0.9473684430122375] - loss값, metrics값

'''