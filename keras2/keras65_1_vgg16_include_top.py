import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

# model = VGG16() # include_top=True, input_shape=(224,224,3)

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model.summary()

print(len(model.weights)) # 32 
print(len(model.trainable_weights)) # 32 

#################### include_top = True #####################

'''
#1. FC layer :: 원래것을 그대로 사용.
#2. input_shape=(224, 224, 3) 고정값, > 바꿀 수 없다.

Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0

 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
.................................................................
 flatten (Flatten)           (None, 25088)             0

 fc1 (Dense)                 (None, 4096)              102764544

 fc2 (Dense)                 (None, 4096)              16781312

 predictions (Dense)         (None, 1000)              4097000

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________

'''

#################### include_top = False #####################

'''
#1. FC layer 원래값 삭제 -> 커스터마이징 가능
#2. input_shape=(32, 32, 3) 바꿀수 있음, 커스터마이징 가능

Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0

 block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
.................................................................
Flatten 하단 연산이 없어짐.

=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
'''
