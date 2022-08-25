from sklearn.metrics import accuracy_score, mean_absolute_error
import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]  # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                    # (6, 1)



x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b)
# == model.add(Dense(1, activation='sigmoid'))

# loss = tf.reduce_mean(tf.square(hypothesis-y))  # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y) *
                       tf.log(1-hypothesis))  # binary_crossentropy
# == model.compile(loss='binary_crossentropy')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2001
for epochs in range(epoch):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0:
        print(epochs, "loss :: ", cost_val, "\n", hy_val)


# 4. 평가, 예측

y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))

accuracy = accuracy_score(y_data, y_predict)
print("ACC :: ", accuracy)
mae = mean_absolute_error(y_data, y_predict)
print("mae :: ", mae)

sess.close()

# ACC ::  0.5
# mae ::  0.5
