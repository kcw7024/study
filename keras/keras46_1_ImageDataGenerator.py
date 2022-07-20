import numpy as np
from keras.preprocessing.image import ImageDataGenerator #이미지데이터를 수치화

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
    'd:/_data/image/brain/train/',
    target_size=(150, 150),
    batch_size=15, 
    class_mode='binary', #0 또는 1만 나오는 수치라서
    #color_mode='grayscale',
    shuffle=False           
) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=15, #y값 범위지정
    class_mode='binary', #0 또는 1만 나오는 수치라서
    #color_mode='grayscale',
    shuffle=False           
) #Found 120 images belonging to 2 classes.

print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002D4F9755F70>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)
 
#print(xy_train[0])
#print(xy_train[31][0].shape) #(5, 150, 150, 3) / 컬러
print(xy_train[10][0])
print(xy_train[10][1])
#print(xy_train[31][2]) #error

#print(xy_train[10][0].shape, xy_train[10][1].shape)

# print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>


