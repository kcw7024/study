
#유방암 데이터활용
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


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



#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()

scaler.fit(x_train)
#print(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) #x_train이작업된 범위에 맞춰서 진행


# 2. 모델

# model = Sequential()
# model.add(Dense(10, input_dim=30, activation='linear'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))



input = Input(shape=(30,))
dense1 = Dense(10, activation='relu')(input)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output = Dense(1, activation='sigmoid')(dense5)

model = Model(inputs = input, outputs = output)



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

hist = model.fit(x_train, y_train, epochs=200, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

end_time = time.time() - start_time

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('걸린시간 :', end_time)

y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                
y_predict = np.where(y_predict > 0.5, 1 , 0)   

#print(y_predict)

#print(classification_report(y_test, y_predict))
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)  



'''

#220707, model을 변경하여 적용하고 결과비교하기


1. 모델변경전

loss :  [0.3431629240512848, 0.9035087823867798, 0.0979343056678772]
걸린시간 : 9.873746395111084
acc 스코어 :  0.9035087719298246

2. 모델변경후

loss :  [0.9275791049003601, 0.8070175647735596, 0.16021843254566193]
걸린시간 : 8.52377700805664
acc 스코어 :  0.8070175438596491

3. Standard Scaler (평균을 제거하고 데이터를 단위 분산으로 조정, 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 
변환된 데이터의 확산은 매우 달라짐. 때문에 이상치가 있는경우에는 균형잡힌 처곧를 보장할 수 없다.)

loss :  [0.147552028298378, 0.9473684430122375, 0.0392722487449646]
걸린시간 : 8.350560426712036
acc 스코어 :  0.9473684210526315


'''





