from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(1234)

# 1. 데이터

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
# (442, 10) (442,)

y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 100]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]), name='bias1')

hidden_layer1 = tf.nn.relu(tf.compat.v1.matmul(x, w1) + b1)  # matmul :: 행렬곱 함수

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 80]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80]), name='bias2')

hidden_layer2 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)  # matmul :: 행렬곱 함수

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80, 100]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]), name='bias3')

hidden_layer3 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer2, w3) + b3)  # matmul :: 행렬곱 함수

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 50]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([50]), name='bias4')

hidden_layer4 = tf.nn.relu(tf.compat.v1.matmul(hidden_layer3, w4) + b4)  # matmul :: 행렬곱 함수

w5 = tf.compat.v1.Variable(tf.compat.v1.zeros([50, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias5')

hypothesis = tf.sigmoid(tf.compat.v1.matmul(hidden_layer4, w5) + b5)  # matmul :: 행렬곱 함수


loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

# 3-1. 컴파일

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 200
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


# 4. 평가, 예측


y_predict = sess.run(hypothesis, feed_dict={x:x_test, y:y_test})

r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae: ', mae)


# r2:  -3.5697287607918504
# mae:  149.96629213483146