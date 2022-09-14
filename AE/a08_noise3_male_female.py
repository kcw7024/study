import numpy as np
from keras.datasets import mnist


x_train = np.load('d:/study_data/_save/_npy/keras47_4_train_x.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_4_test_x.npy')
test = np.load('d:/study_data/_save/_npy/keras47_4_test.npy')

# print(x_train.shape, x_test.shape, test.shape)
# (2316, 150, 150, 3)
# (993, 150, 150, 3)

# x_train = x_train.reshape(2316, 67500)
# x_test = x_test.reshape(993, 67500)

x_train = x_train.reshape(2316, 150, 150, 3)
x_test = x_test.reshape(993, 150, 150, 3)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
test_noised = test + np.random.normal(0, 0.1, size=test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
test_noised = np.clip(test_noised, a_min=0, a_max=1)


from keras.models import Sequential, Model
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, 
#                     input_shape=(67500, ),
#                     activation='relu'))
#     model.add(Dense(units=67500, activation='sigmoid'))
#     return model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,input_shape=(150, 150, 3),kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(32,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(3,(2,2),padding='same',activation='sigmoid'))
    return model


# # model = autoencoder(hidden_layer_size=64)
model = autoencoder(hidden_layer_size=154) # PCA의 95% 성능
# model = autoencoder(hidden_layer_size=331) # PCA의 99% 성능

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=10)
output = model.predict(x_test_noised)
pred = model.predict(test_noised)

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5, m1), (ax6, ax7, ax8, ax9, ax10, m2), 
      (ax11, ax12, ax13, ax14, ax15, m3)) = \
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

m1.imshow(test[0].reshape(150, 150, 3))
    
# 노이즈가 들어간 이미지를 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
m2.imshow(test_noised[0].reshape(150, 150, 3))

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3))
    if i ==0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
m3.imshow(pred[0].reshape(150, 150, 3))
    
plt.tight_layout()
plt.show()

