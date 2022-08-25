from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
import pandas as pd
tf.set_random_seed(123)

datasets = load_iris()

x = datasets.data
y = datasets.target

print(type(x))
# <class 'numpy.ndarray'>
print(x.shape, y.shape)
# (150, 4) (150,)
# y = np.array(y)

# y = y.reshape(-1, 1)

y = pd.get_dummies(y)
print(y.head(14))
# (150, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, stratify=y
)

# 2. 모델 구성

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)

# 4. 평가, 예측
#
y_predict = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
y_test = y_test.values
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_predict)
print("ACC :: ", accuracy)
# mae = mean_absolute_error(np.argmax(y_test), np.argmax(hy_val))
# print("mae :: ", mae)


sess.close()
