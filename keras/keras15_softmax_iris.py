import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report


#1. 데이터

datasets = load_iris()
print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y = datasets.target


#print(x.shape, y.shape) #(150, 4) (150,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

# 2. 모델

model = Sequential()
model.add(Dense(10, input_dim=4, activation='linear'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy','mse'] 
              )

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=300, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )

# # 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                
y_predict = np.where(y_predict > 0.5, 1 , 0)   

#print(y_predict)

print(classification_report(y_test, y_predict))
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)  
