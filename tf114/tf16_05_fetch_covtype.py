from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, fetch_covtype
import numpy as np
import tensorflow as tf
import pandas as pd
tf.set_random_seed(123)

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (581012, 54) (581012,)

y = y.reshape(-1, 1)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, stratify=y
)


# 2. 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
w = tf.Variable(tf.random_normal([54, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
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
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)

sess.close()

