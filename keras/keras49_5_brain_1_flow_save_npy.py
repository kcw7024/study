import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split #이미지데이터를 수치화


#1. 데이터


train_datagen = ImageDataGenerator(
    rescale=1./255,             #스케일링
    horizontal_flip=True,       #수평으로 뒤집어준다
    vertical_flip=True,         #수직으로 뒤집어준다 
    width_shift_range=0.1,      #가로로 움직이는 범위          
    height_shift_range=0.1,     #세로로 움직이는 범위
    rotation_range=5,           #이미지 회전           
    zoom_range=1.2,             #임의 확대/축소 범위
    shear_range=0.7,            #임의 전단 변환 (shearing transformation) 범위 #짜부~
    fill_mode='nearest'         #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식    
)

#평가데이터이기때문에 이미지 증폭은 하면 X
test_datagen = ImageDataGenerator(
    rescale=1./255
)


xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/',
    target_size=(150, 150),
    batch_size=500, 
    class_mode='binary', #0 또는 1만 나오는 수치라서
    color_mode='grayscale',
    shuffle=False           
) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=500, #y값 범위지정
    class_mode='binary', #0 또는 1만 나오는 수치라서
    color_mode='grayscale',
    shuffle=False           
) #Found 120 images belonging to 2 classes.


augument_size = 10000 #증폭
#batch_size = 64

randidx = np.random.randint(xy_train, size=augument_size) #(60000)

x_augmented = xy_train[randidx].copy()
y_augmented = xy_train[randidx].copy()


#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

x_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[0]
#y_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[1]


x_train, y_train, x_test, y_test = train_test_split(
        xy_train, xy_test, test_size=0.2, shuffle=False
)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))



#수치화한 데이터를 저장한다.
np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=x_train) # train x값
np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=y_train) # train y값
np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=x_test) # test x값
np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=y_test) # test y값



#2. 모델
'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #흑백이라 이진분류 ~

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(xy_train[0][0], xy_train[0][1]) #배치를 최대로 잡으면 이방법도 가능
hist = model.fit_generator(xy_train, epochs=200, 
                    steps_per_epoch=33, #전체데이터/batch=160/5=32
                    validation_data=xy_test,
                    validation_steps=4, #생성기에서 만들어 낼 단계(샘플 배치)의 총 개수. 보통은 검증 데이터셋의 샘플 수를 배치 크기로 나눈 값을 갖습니다.
                    )

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


print('loss : ', loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])

# loss :  0.6947375535964966
# val_accuracy :  1.0
# accuracy :  0.3030303120613098
# val_loss :  0.6914277672767639

import matplotlib
import matplotlib.pyplot as plt #그려보자~
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='_', c='red', label='loss')
plt.plot(hist.history['val_accuracy'], marker='_', c='blue', label='val_accuracy')
plt.plot(hist.history['accuracy'], marker='_', c='green', label='accuracy')
plt.plot(hist.history['val_loss'], marker='_', c='black', label='val_loss')
plt.grid()
plt.title('뇌사진 비교')
plt.ylabel('loss')
plt.xlabel('epochs') #횟수당
#plt.legend(loc='upper right') #label 값 명칭의 위치
plt.legend()
plt.show()
'''