from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.set_random_seed(123)

# 1. 데이터

datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (506, 13) (506,)

y = y.reshape(-1, 1)
# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 80]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80]), name='bias1')

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80, 100]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]), name='bias2')

hidden_layer2 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 70]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([70]), name='bias3')

hidden_layer3 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([70, 80]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80]), name='bias4')

hidden_layer4 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias5')

hypothesis = tf.matmul(hidden_layer4, w5) + b5

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

# 3-1. 컴파일

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 100
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


# 4. 평가, 예측

y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)
sess.close()