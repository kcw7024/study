import numpy as np
from keras.datasets import mnist



(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784, ))

encoded = Dense(64, activation='relu')(input_img)
# 중간레이어에서 1차적으로 축소하며 중요한 특성만 남긴다(필요없는것들을 제거)
# encoded = Dense(1064, activation='relu')(input_img) # 정확도가 높음
# encoded = Dense(16, activation='relu')(input_img) # 형태정도만 유지가 됨
# encoded = Dense(1, activation='relu')(input_img) # 절망편
decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded) # mse, 잡데이터가 많이남음
# decoded = Dense(784, activation='linear')(encoded) # mse, 결과 최악 
# decoded = Dense(784, activation='tanh')(encoded) # mse, 절망



autoencoder = Model(input_img, decoded)

# autoencoder.summary()
# 입력과 출력이 같은구조
# 첫번째 노드에서 > 중간노드 > 아웃풋노드 순으로 축소되었다가 다시 원래대로 돌아옴

'''
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 784)]             0

 dense (Dense)               (None, 64)                50240

 dense_1 (Dense)             (None, 784)               50960

=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
'''

# 준 지도학습임.
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10

plt.figure(figsize=(20, 4))
for i in range(n) : 
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()




