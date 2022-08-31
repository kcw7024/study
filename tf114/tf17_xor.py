from pickletools import optimize
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
tf.compat.v1.set_random_seed(123)


#1. 데이터

x_data = [[0,0], [0,1], [1,0], [1,1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]           # (4, 1)

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)


loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1000
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))

accuracy = accuracy_score(y_data, y_predict)
print("ACC :: ", accuracy)
mae = mean_absolute_error(y_data, y_predict)
print("mae :: ", mae)

sess.close()


