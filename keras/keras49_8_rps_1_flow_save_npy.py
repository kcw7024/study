from grpc import AuthMetadataContext
from tensorflow.keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split



train_datagen = ImageDataGenerator(
    #rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_data = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/rps/',
    target_size=(150, 150),
    batch_size=2000, 
    class_mode='binary', #0 또는 1만 나오는 수치라서
    shuffle=False           
) #Found 1027 images belonging to 2 classes.



x = xy_data[0][0]
y = xy_data[0][1]

#print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
     x, y, train_size=0.7, shuffle=True
    )


print(x_train.shape) #(50000, 32, 32, 3) 

augument_size = 3000 #증폭
batch_size = 64

randidx = np.random.randint(x_train.shape[0], size=augument_size) #(60000)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

x_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[0]
#y_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[1]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# xy_train  = test_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=False)
# xy_test = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

print(x_train.shape, y_train.shape) #(4400, 150, 150, 3) (4400,)
print(x_test.shape, y_test.shape) #(600, 150, 150, 3) (600,)


np.save('d:/study_data/_save/_npy/keras49_8_train_x.npy', arr=x_train) # train x값
np.save('d:/study_data/_save/_npy/keras49_8_train_y.npy', arr=y_train) # train y값
np.save('d:/study_data/_save/_npy/keras49_8_test_x.npy', arr=x_test) # test x값
np.save('d:/study_data/_save/_npy/keras49_8_test_y.npy', arr=y_test) # test y값



# #2. 모델
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=(28, 28, 1), activation='relu'))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련

# model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# #배치를 최대로 잡으면 이방법도 가능
# hist = model.fit_generator(xy_train, epochs=5, steps_per_epoch=len(xy_train),validation_data=xy_test, validation_steps=4) 

# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']


# print('loss : ', loss[-1])
# print('val_accuracy : ', val_accuracy[-1])
# print('accuracy : ', accuracy[-1])
# print('val_loss : ', val_loss[-1])



# loss :  0.06617587804794312
# val_accuracy :  0.8828125
# accuracy :  0.9775800108909607
# val_loss :  0.4464709460735321