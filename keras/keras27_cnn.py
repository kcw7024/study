from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten #이미지작업(Conv2D), 데이터를 펼친다(Flatten)

model = Sequential()

#model.add(Dense(units=10, input_shape=(3, ))) #input_shape=(10,10,3))) #(batch_size, input_dim )
#model.summary()

# (input_dim + bias) * units = summary Param 갯수 (Dense모델)

model.add(Conv2D(filters = 10, kernel_size=(3, 3), input_shape = (8, 8, 1))) # (batch_size, rows, colums, channels) # 출력 (None, 6, 6, 10)
# 예제기준(Conv2D)
# Conv2D 에서의 노드수(10)
# kernerl_size = 이미지 크롭 규격
# input_shape = 5x5, 흑백 이미지
# 4+D tensor with shape: batch_shape + (channels, rows, cols) if data_format='channels_first' 
# 4+D tensor with shape: batch_shape + (rows, cols, channels) if data_format='channels_last'
# filters = output node의 수 
model.add(Conv2D(7, (2, 2), activation='relu')) # 출력 (None, 5, 5, 7)
model.add(Flatten()) # (None, 175)   
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN모델)
 

''' 

Model: "sequential" 10 (2,2) (5,5,1)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
=================================================================
Total params: 50
Trainable params: 50
Non-trainable params: 0
_________________________________________________________________


Model: "sequential" 10 (3,3) (10,10,1)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 8, 8, 10)          100
=================================================================
Total params: 100
Trainable params: 100
Non-trainable params: 0
_________________________________________________________________

'''





