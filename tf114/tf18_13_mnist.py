#실습
#DNN으로 구성!

import tensorflow as tf 
import numpy as np
import keras
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(123)


#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


print(x_train.shape, x_test.shape)

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

w2 = tf.Variable(tf.random_normal([784, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')

hypothesis = tf.nn.softmax(tf.matmul(x, w2) + b2)

#print(w2) # <tf.Variable 'weight2:0' shape=(784, 10) dtype=float32_ref>
#print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))


# train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 400
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        # print(epochs, "loss :: ", cost_val, "\n", hy_val)
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
# y_test = y_test.values

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print("ACC :: ", acc)

sess.close()

# ACC ::  0.1306
# ACC ::  0.1324 (Adam)