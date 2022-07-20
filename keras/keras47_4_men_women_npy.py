#본인사진으로 predict 하시오
#d:/study_data/_data/image/안에 넣고

#넘파이 불러와서 모델링~
#넘파이 불러와서 모델링~
import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화


#1. 데이터

#수치화한 데이터를 저장한다.
# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0]) # train x값
# np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1]) # train y값
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0]) # test x값
# np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1]) # test y값

x_train = np.load('d:/study_data/_save/_npy/keras47_4_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_4_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_4_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_4_test_y.npy')
mbin = np.load('d:/study_data/_save/_npy/keras47_4_mbin.npy')

print(x_train)
print(x_train.shape) #(500, 150, 150, 1)
print(y_train.shape) #(500,)
print(x_test.shape)  #(500, 150, 150, 1)
print(y_test.shape)  #(500,)

#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #흑백이라 이진분류 ~

#3. 컴파일, 훈련

model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#배치를 최대로 잡으면 이방법도 가능
hist = model.fit(x_train, y_train, 
                 epochs=10, batch_size=15, 
                 validation_split=0.2
                 ) 

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']




print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])


#4. 평가, 훈련

y_predict = model.predict(mbin)
print(y_predict)

# loss :  0.6947375535964966
# val_accuracy :  1.0
# accuracy :  0.3030303120613098
# val_loss :  0.6914277672767639

# import matplotlib
# import matplotlib.pyplot as plt #그려보자~
# matplotlib.rcParams['font.family'] ='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] =False

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='_', c='red', label='loss')
# plt.plot(hist.history['val_accuracy'], marker='_', c='blue', label='val_accuracy')
# plt.plot(hist.history['accuracy'], marker='_', c='green', label='accuracy')
# plt.plot(hist.history['val_loss'], marker='_', c='black', label='val_loss')
# plt.grid()
# plt.title('뇌사진 비교')
# plt.ylabel('loss')
# plt.xlabel('epochs') #횟수당
# #plt.legend(loc='upper right') #label 값 명칭의 위치
# plt.legend()
# plt.show()



# loss :  6.485952042112331e-08
# val_accuracy :  0.6099137663841248
# accuracy :  1.0
# val_loss :  7.33774995803833


############ 마동석사진
# loss :  0.001224910607561469
# val_accuracy :  0.5732758641242981
# accuracy :  0.9994600415229797
# val_loss :  4.813855171203613
# [[0.00015712]]