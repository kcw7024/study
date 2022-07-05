#유방암 데이터활용
import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('GPU 됨')
    aaa = 'GPU'
else :
    print('GPU 안됨')
    aaa = 'CPU'
    
 


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
model.add(Dense(500, input_dim=30, activation='linear'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='linear'))
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
earlyStopping = EarlyStopping(monitor='var_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)
#restore_best_weights=True로 하게되면 Earlystopping 검증중의 가장 최소값을 가져온다.
#False로 지정하면 마지막 값을 가져온다


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print(aaa, '걸린시간 :', end_time)

y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                
y_predict = np.where(y_predict > 0.5, 1 , 0)   

#print(y_predict)

print(classification_report(y_test, y_predict))
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)  


'''



'''


