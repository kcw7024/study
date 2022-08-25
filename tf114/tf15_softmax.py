import numpy as np
import tensorflow as tf
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],  # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],  # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# 2. 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
# == 이렇게 하나의 레이어

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# == model.add(Dense(3, activation='sotfmax', input_dim=4))

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = 'categorical_crossentropy'

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)
 

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))

# accuracy = accuracy_score(y_data, y_predict)
# print("ACC :: ", accuracy)
# mae = mean_absolute_error(y_data, y_predict)
# print("mae :: ", mae) 

 
sess.close()
        