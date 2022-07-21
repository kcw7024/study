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

augument_size = 40 #증폭
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
import time
start_time = time.time()
print("시작 : ", start_time)
x_augmented = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, save_to_dir='d:/study_data/_temp/', shuffle=False).next()[0] #x값
end_time = time.time() - start_time

print(augument_size, "개 증폭에 걸린 시간 : ",  round(end_time, 3), "초")

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,) 

