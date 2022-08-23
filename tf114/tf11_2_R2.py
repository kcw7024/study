import tensorflow as tf
from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3]
y_train = [1, 2, 3]
x_test = [4, 5, 6]
y_test = [4, 5, 6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='weight'))
# w = tf.compat.v1.Variable(10, dtype=tf.float32, name='weight')
sess = tf.compat.v1.Session()

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse

# learning late 를 포함한 gradient 수식을 풀어쓴것
lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - lr * gradient
update = w.assign(descent)

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print(sess.run(w))

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={
                              x: x_train, y: y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    # w_history.append(w_v[0])
    loss_history.append(loss_v)

# 실습 R2로 만들기

y_predict = x_test * w_v
print(y_predict)
# [3.99998951 4.99998689 5.99998426]

from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
r2 = r2_score(y_test, y_predict)
print("R2 :: ", r2)
mae = mean_absolute_error(y_test, y_predict)
print("mae :: ", mae)

# R2 ::  0.9999999997351949
# mae ::  1.3113021850585938e-05

sess.close()
