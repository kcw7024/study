#유방암 데이터활용
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
import time
from sklearn.metrics import classification_report

#1. 데이터

datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DECR)
#print(datasets.feature_names)


x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) #(569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

# 2. 모델

model = Sequential()
model.add(Dense(10, input_dim=30, activation='linear'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# #linear : 선형
# #sigmoid : 0 과 1사이로 값을 반환해줌  (이진분류는 아웃풋에서 무조건 sigmoid)
# # : 0.4 같은 숫자를 처리하기 위해 반올림 처리 해줘야함



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy','mse'] 
              )
#accuracy
# #이진분류에서 사용
# #정확도, 평가지표를 판단할때 사용 #loss에 정확도도 표기


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 검증중의 가장 최소값을 가져온다.
#False로 지정하면 마지막 값을 가져온다


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print('걸린시간 :', end_time)

y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                
y_predict = np.where(y_predict > 0.5, 1 , 0)   

#print(y_predict)

print(classification_report(y_test, y_predict))
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)  


'''
**이진분류는 마지막 activation이 sigmoid
**이진분류는 binary_crossentropy < 무조건~

activation (활성화함수, y값을 제한시킨다.)


# 과제 1을 수행하기 위해 accuracy_score 작업 부터 한다.

1. accuracy_score 처리를 해주려고 하는데 계속 value 에러가 났다.
2. 에러메세지 확인해보니 이진분류를 확인하는 모델인데 y_test와 y_predict의 값이 달라서 비교할수가없었다.
3. flatten()와 np.where를 이용하여 변환하는 방법을 참고한다.
4. 처리후 accuracy_score 적용해주니 잘 돌아간다.

# 결과값 확인해봄

loss :  0.280094176530838
accuracy : 0.9035087823867798
mse :  0.07709896564483643
accyracy 스코어 :  0.9035087719298246


# activation 은 relu를 사용할때 값이 더 좋아졌다.
# Earlystopping을 사용하는데도 여러번 에포를 돌려도 작업이 중지되지 않는점이 궁금하다. 흠

'''


