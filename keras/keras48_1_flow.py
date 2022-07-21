from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100 #증폭

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) #(784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape) # (100, 28, 28, 1)
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) #(100,)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x
    np.zeros(augument_size), # y 
    batch_size=augument_size,
    shuffle=True
)#.next()

#.next() 사용시 앞의 shape를 건너뛴다.

# print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000246025CA790>
# print(x_data[0]) # 
# print(x_data[0][0].shape)  #next()를사용후 (28, 28, 1)
# print(x_data[0][1].shape)  #next()를사용후 (28, 28, 1)

#.next() 사용하지 않을때

print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000246025CA790>
print(x_data[0]) # 첫번째 배치, x와 y가 모두 포함
print(x_data[0][0].shape) #x값 (100, 28, 28, 1)   
print(x_data[0][1].shape) #y값 (100,) 
            

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
#   plt.imshow(x_data[0][i],cmap='gray') #next 사용할때, 첫번째 shape을 건너뛴다.
    plt.imshow(x_data[0][0][i],cmap='gray') # next 사용하지 않을때.
plt.show()

