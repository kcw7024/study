
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from keras.applications import ResNet101, ResNet50
from keras.optimizers import Adam
import time
from sklearn.model_selection import train_test_split
#####
import warnings
warnings.filterwarnings('ignore')

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

test_datagen = ImageDataGenerator(rescale=1./255)

xy_data = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/horse_or_human/',
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

augument_size = 5000 #증폭
batch_size = 64

randidx = np.random.randint(x_train.shape[0], size=augument_size) #(60000)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

x_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[0]
y_augmented  = train_datagen.flow(x_augmented, y_augmented, batch_size=augument_size, shuffle=False).next()[1]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#2. 모델
res50 = ResNet50(weights='imagenet', include_top=False,
              input_shape=(150, 150, 3))
res50.trainable = True

model = Sequential()
model.add(res50)
model.add(GlobalAvgPool2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(x_train,y_train, epochs=100, validation_steps=5, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print("loss, acc : ", loss)

# 걸린 시간 :  2185.44
# loss, acc :  [0.004341119900345802, 1.0]