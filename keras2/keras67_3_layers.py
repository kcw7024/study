from ast import Mod
from email.mime import base
from pickletools import optimize
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Dropout
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar100
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))


#1. 
# model.trainable = False
'''
Total params: 17
Trainable params: 0
Non-trainable params: 17
'''

#2.
# for layer in model.layers : 
#     layer.trainable = False

'''
Total params: 17
Trainable params: 0
Non-trainable params: 17
'''

# model.summary()

# model.layers[0].trainable = False # Dense
'''
Total params: 17
Trainable params: 11
Non-trainable params: 6
'''
# model.layers[1].trainable = False # Dense_1
'''
Total params: 17
Trainable params: 9
Non-trainable params: 8
'''
model.layers[2].trainable = False # Dense_2

'''
Total params: 17
Trainable params: 14
Non-trainable params: 3

'''

model.summary()

print(model.layers)

# [<keras.layers.core.dense.Dense object at 0x000002576CF34A00>, 
#  <keras.layers.core.dense.Dense object at 0x0000025773FBCBB0>, 
#  <keras.layers.core.dense.Dense object at 0x00000257132A8D90>]

