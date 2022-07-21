from grpc import AuthMetadataContext
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

augument_size = 40000 #증폭
randidx = np.random.randint(x_train.shape[0], size=augument_size) #(60000)
# x_train.shape[0] 범위내에서 augment_size 만큼 정수값을 뽑아준다
#randint 균일 분포의 정수 난수(랜덤) 생성 (최소값, 최대값, 조건) 사이에서 생성해준다.
#x_train.shape #(60000, 28, 28)
print(x_train.shape[0]) # 60000
print(randidx)          # [20014 40476  4736 ... 53470 50713 47713] 랜덤으로 
print(np.min(randidx), np.max(randidx)) # 5 59997 랜덤으로 뽑은 40000개

print(type(randidx)) # <class 'numpy.ndarray'>

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) #(40000, 28, 28)
print(y_augmented.shape) #(40000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)


#이미지 변경
x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[0] #x값
print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,) 


# print(x_train[0].shape) # (28, 28)
# print(x_train[0].reshape(28*28).shape) #(784,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape) # (40000, 28, 28, 1)
# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape) #(100,)

# x_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x
#     np.zeros(augument_size), # y 
#     batch_size=augument_size,
#     shuffle=True
# )#.next()

# #.next() 사용시 앞의 shape를 건너뛴다.

# # print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000246025CA790>
# # print(x_data[0]) # 
# # print(x_data[0][0].shape)  #next()를사용후 (28, 28, 1)
# # print(x_data[0][1].shape)  #next()를사용후 (28, 28, 1)

# #.next() 사용하지 않을때

# print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000246025CA790>
# print(x_data[0]) # 첫번째 배치, x와 y가 모두 포함
# print(x_data[0][0].shape) #x값 (100, 28, 28, 1)   
# print(x_data[0][1].shape) #y값 (100,) 
            

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
# #   plt.imshow(x_data[0][i],cmap='gray') #next 사용할때, 첫번째 shape을 건너뛴다.
#     plt.imshow(x_data[0][0][i],cmap='gray') # next 사용하지 않을때.
# plt.show()

