from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
import numpy as np
import pandas as pd
import tensorflow as tf
tf.set_random_seed(123)

datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (178, 13) (178,)

# y = y.reshape(-1, 1)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_train)

# 2. 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.Variable(tf.random_normal([13, 50]), name='weight1')
b1 = tf.Variable(tf.random_normal([50]), name='bias1')

hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([50, 80]), name='weight2')
b2 = tf.Variable(tf.random_normal([80]), name='bias2')

hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

w3 = tf.Variable(tf.random_normal([80, 3]), name='weight3')
b3 = tf.Variable(tf.random_normal([3]), name='bias3')

hypothesis = tf.nn.softmax(tf.matmul(hidden2, w3) + b3)


# == model.add(Dense(3, activation='sotfmax', input_dim=4))

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# loss = 'categorical_crossentropy'

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

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
y_test = y_test.values

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print("ACC :: ", acc)

sess.close()
