import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()


print(model.weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[1.0971547 , 0.26039648, 0.2719581 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
        numpy=array([[ 0.9131527 , -0.41186094],
       [ 0.4909122 ,  0.9507556 ],
       [ 0.4117149 , -0.62540996]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
       <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
       numpy=array([[0.78589976], [0.43447065]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print("="*80)
print(model.trainable_weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.3586756 , -0.00230706,  0.11630988]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
numpy=array([[ 0.10275936,  0.25741386],
       [ 0.23606777,  0.89341986],
       [-0.9606107 , -0.70817095]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
       <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32,
       numpy=array([[1.2218786],
       [0.6670815]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''

print("="*80)
print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 6

model.trainable = False

print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 0
print("="*80)
print(model.trainable_weights) # []


model.summary()

'''
=================================================================
Total params: 17
Trainable params: 0
Non-trainable params: 17
_________________________________________________________________

'''
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=1, epochs=100) # loss값이 고정되어있음/ 훈련하지 않았다

y_pred = model.predict(x)
print(y_pred[:3])

'''
[[1.6645715]
 [3.329143 ]
 [4.993715 ]]
'''


