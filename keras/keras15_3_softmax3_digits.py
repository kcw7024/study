import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine, load_digits


#1. 데이터

datasets = load_digits()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(1797, 64) (1797,) - 8 x 8 의 이미지가 1797장 있다. /  #원핫 인코딩으로 1797,10으로 만들어준다. 
print(np.unique(y, return_counts=True))     #[0 1 2 3 4 5 6 7 8 9]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)


# 2. 모델

model = Sequential()
model.add(Dense(100, input_dim=64, activation='linear'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(#loss='binary_crossentropy', #음수가 나올수 없다. (이진분류에서 사용)
              loss='categorical_crossentropy',#다중분류에서는 loss는 이것만 사용한다(당분간~)
              optimizer='adam', 
              metrics=['accuracy'] 
              )

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='var_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1                 
                 )


# # 4. 평가, 예측

# 첫번째 방법
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss )
# print('acc : ', acc)

# 두번째 방법
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


#print("#" * 80)
#print(y_test[:5])
#print("#" * 80)
y_pred = model.predict(x_test[:5])
#print(y_pred)
#print("#"*15 + "pred" + "#"*15)


# 2. argmax 사용
# y_pred = np.argmax(y_test, axis =1)
# #print(y_test)
# y_pred = to_categorical(y_pred)
# #print(y_pred)
# acc2 = accuracy_score(y_test, y_pred)
# print("acc : ", acc2)


#풀이해주신것
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
#print(y_predict)
y_test = np.argmax(y_test, axis=1)
#print(y_test)

acc = accuracy_score(y_test, y_predict)
print("acc 스코어 : ", acc)

import matplotlib.pyplot as plt

plt.gray()
plt.matshow(datasets.images[0])
plt.show()

'''
loss :  0.10082201659679413
accuracy :  0.9861111044883728
acc 스코어 :  0.9861111111111112

'''